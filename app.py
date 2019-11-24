from flask import Flask,jsonify,request
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import sys
import turicreate as tc
sys.path.append("..")
import json
from flask_cors import CORS
from flask import request
import datetime
import json as json
from pymongo import MongoClient
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from bson import ObjectId
import math
# from flask_ngrok import run_with_ngrok

app=Flask(__name__)
CORS(app)
app.run(debug=True)
# run_with_ngrok(app)

url='mongodb+srv://test:test@cluster0-12rwi.azure.mongodb.net/test?retryWrites=true&w=majority'
db_name='shop_list'


def read_json(url,db_name,table_name):
    client = MongoClient(url)
    db = client.get_database(db_name)
    if(table_name=="customers"):
        return(db.customers)
    elif(table_name=="transactions"):
        return(db.transactions)
    elif(table_name=="itemlist"):
        return(db.itemlist)
    elif(table_name=="category"):
        return(db.category)
    elif(table_name=="rta"):
        return(db.rta)
    elif(table_name=="Recent_purchases"):
        return(db.Recent_purchases)

#functions for recommendation -->>
#To get the overall users list
def get_user():
    users_table=read_json(url,db_name,"customers")
    res=users_table.find({},{"_id":0})
    users=[]
    for i in res:
        users.append(str(i["cust_id"]))
    return users

#To get the the data for recommendation
def get_data(users):
    user_data=[]#output 1
    item_data=[]#output 2
    target_data=[]#output 3

    transactions_table=read_json(url,db_name,"transactions")

    for user in users:
        #An object to find in the table
        query={}
        query["cust_id"]=int(user)
        
        res=transactions_table.find(query,{"_id":0,"cust_id":0})#ignoring the _id and cust_id fields
        for obj in res:
            for enteries in obj["Transaction"]:
                user_data.append(str(user))
                item_data.append(str(enteries["item_id"]))
                target_data.append(len(enteries["item_transactions"]))
    return user_data,item_data,target_data
    
#Functions for prediction algorithms -->>
def calc_error(predicted,actual):
    error=0
    for i in range(0,len(actual)):
        error=error+((actual[i]-predicted[i])*(actual[i]-predicted[i]))
    error=error/len(actual)
    return math.sqrt(error)

#Prefetches the dates and quantity with corresponding to item_id in recent purchases
def prefetch(item_id_dict,item_info):
  for x in item_info:
    for y in x["Transaction"]:
      if(item_id_dict.get(y['item_id'])!=None):
        dates=[]
        quantity=[]
        item_trans = y['item_transactions']
        for z in item_trans:
          dates.append(z['date'])
          quantity.append(z['quantity'])
        item_id_dict[y['item_id']]["dates"]=dates
        item_id_dict[y['item_id']]["quantity"]=quantity
  return item_id_dict
        

def removeOutliers(frequency,threshold):
    modified_freq=[]
    modified_quantity=[]

    for freq,arr in frequency.items():
        if(len(arr)==1):
            modified_freq.append(freq)
            modified_quantity.append(arr[0])
        else:
            z=stats.zscore(arr)
            for idx in range(0,len(z)):
                if(np.isnan(z[idx])==True):
                    modified_freq.append(freq)
                    modified_quantity.append(arr[idx])
                elif(abs(z[idx])<threshold):
                    modified_freq.append(freq)
                    modified_quantity.append(arr[idx])
    return modified_freq,modified_quantity

    
def get_dates_quantity(dates,quantity,remove_outliers=0,outliers_threshold=0):
    dates_arr=[]
    frequency_distribution={} 
    for i in range(1,len(dates)):
        frequency=(dates[i]-dates[i-1]).astype('int64')
        dates_arr.append(frequency)
        frequency_distribution[frequency]=[]

    quantity=quantity[1:]
  
    if(remove_outliers==1):
        for idx in range(0,len(dates_arr)):
            frequency_distribution[dates_arr[idx]].append(quantity[idx])
            modified_dates,modified_quantity=removeOutliers(frequency_distribution,outliers_threshold)
        modified_dates=np.array(modified_dates).astype('int64')
        modified_dates=np.reshape(modified_dates,(len(modified_dates),1))
        return modified_dates,modified_quantity
    else:
        dates_arr=np.array(dates_arr).astype('int64')
        dates_arr=np.reshape(dates_arr,(len(dates_arr),1))
    return (dates_arr,quantity)

def algo(dates,quantity,gap):
    dates = np.array(dates).astype('datetime64[D]')
    #preparing frequncy array(dates_arr)
    (dates_arr , quantity) = get_dates_quantity(dates,quantity,0,1.5)

    #INITIALISING THE MODEL
    
    svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1)
    random_forest = RandomForestRegressor(n_estimators=5,random_state=10)

     #FITTING THE MODEL
    #svr_poly.fit(dates_arr,quantity)-- CURRENTLY NOT USING POLY
    svr_rbf.fit(dates_arr,quantity)
    random_forest.fit(dates_arr,quantity);

    #READING THE CURRENT TIMESTAMP TO FIND THE GAP
    predict_dates = gap
    predict_dates = np.reshape(predict_dates,(1,1))
 
    #PREDICTING FROM THE FITTED MODEL
    if predict_dates > max(dates_arr):
      maximum = max(dates_arr)[0]
      k = 0
      max_quant = 0
      for i in dates_arr:
        if (i[0] == maximum):
          if (quantity[k] > max_quant):
            max_quant = quantity[k]
        k += 1
      return(round(max_quant))

    rbf= svr_rbf.predict(dates_arr)
    rf=random_forest.predict(dates_arr)#rf=Random Forest
  
    rounded_rbf=[]
    rounded_rf=[]

    for i in range(0,len(rbf)):
        rounded_rbf.append(round(rbf[i]))
        rounded_rf.append(round(rf[i]))
    
    error_rbf=calc_error(rounded_rbf,quantity)
    error_rf=calc_error(rounded_rf,quantity)
    #print(error_rbf,error_rf) -->> ERROR PRINTING
    if(error_rbf<=error_rf):
        return svr_rbf.predict(predict_dates)[0]
    else:
        return random_forest.predict(predict_dates)[0]


@app.route('/ml/recommend',methods=['GET'])
#Main function for recommendation
def recommend():

    user_id = request.args.get('userid')
    users=get_user()
    #users=[25]
    user_data,item_data,target_data=get_data(users)

    user_arr=[]
    user_arr.append(str(user_id))

    sf = tc.SFrame({'user_id':user_data,'item_id':item_data,'frequency':target_data})
    m = tc.item_similarity_recommender.create(sf,target="frequency",similarity_type='cosine')
    #recom=m.recommend(users,k=10) UNCOMMENT IF want to test for all users
    recom=m.recommend(user_arr,k=10)
    output={}
    output["item_id"]=[]

    for items in recom["item_id"]:
      output["item_id"].append(items)

    return json.dumps(output)


@app.route('/ml/predict',methods=['GET'])
def predict():
  userid = request.args.get('userid')
  transaction =read_json(url,db_name,"transactions")
  recent_purchases = read_json(url,db_name,"Recent_purchases")#Getting the rta table

  # itemlist = db.itemlist
  user_dict={}
  user_dict["cust_id"]=int(userid)
  item_info = transaction.find(user_dict,{"Transaction.item_transactions.date":1, "Transaction.item_transactions.quantity":1,"Transaction.item_id":1,"_id":0})
  itemDetails = recent_purchases.find(user_dict,{'_id':0})#Mongo query

  output = []
  item_id_dict={}#Stores the item and dates and quantity array
  item_info_dict=[] #stores the avg , last_date and item_id
  
  for item in itemDetails:
      for one_item in item['recents']:
        item_obj_dict={}
        item_id_dict[one_item["item_id"]]={}
        item_obj_dict["item_id"]=one_item["item_id"]
        item_obj_dict["avg"]=one_item["avg"]
        item_obj_dict["last_date"]=one_item["last_date"]
        item_info_dict.append(item_obj_dict)

  item_id_dict=prefetch(item_id_dict,item_info)        
  for one_item in item_info_dict:
    avg = one_item['avg'] #Fetch the avg of an item for a particular user
    datetimeobj = datetime.datetime.now()
    date = datetimeobj.strftime("%Y") + "-" +datetimeobj.strftime("%m") + "-" + datetimeobj.strftime("%d")
        
    last_date_of_purchase=one_item['last_date']
        
    t = (datetime.datetime.strptime(date,"%Y-%m-%d") - datetime.datetime.strptime(last_date_of_purchase,"%Y-%m-%d"))
    t = t.days
    avg=math.ceil(avg)
    if(avg !=0 and ((avg)-2)<=t and t<=(avg+3)):
      item_pred = {}
      itemid = one_item['item_id']
      item_dict=item_id_dict.get(itemid)
      
      if(len(item_dict["dates"])>2 and len(item_dict["quantity"])>2):
        ans = algo(dates=item_dict["dates"],quantity=item_dict["quantity"],gap=t)
        dictionary = dict({'item_id' : itemid})

            # itemName = itemlist.find( dictionary, {'item_name':1 ,'item_id':1, '_id':0})
            
        item_pred['itemID'] = itemid
            # for name in itemName['item_name']:
        item_pred['itemName'] = "Test_items"
        item_pred['Quantity'] = round(ans)
        output.append(item_pred)
        
        # else:
        #   print("Hello")
        #   customer_dict={}
        #   customer_dict["cust_id"]=user
        #   info_dict={}
        #   info_dict["recent.item_id"]=one_item["item_id"]
        #   recent_transactions.update(customer_dict,{'$pull':info_dict})
  json_output=json.dumps(output)
  return json_output

if __name__=='__main__':
    app.run()

