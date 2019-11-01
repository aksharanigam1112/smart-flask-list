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

app=Flask(__name__)
CORS(app)

def create_data_dummy(data):
    data_dummy = data.copy()
    data_dummy['purchase_dummy'] = 1
    return data_dummy

def split_data(data):
    '''
    Splits dataset into training and test set.
    
    Args:
        data (pandas.DataFrame)
        
    Returns
        train_data (tc.SFrame)
        test_data (tc.SFrame)
    '''
    train, test = train_test_split(data, test_size = .2)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data

def model(train_data,name,user_id,item_id,target,users_to_recommend,n_rec,n_display):
    if name=='popularity':
        model=tc.popularity_recommender.create(train_data,user_id=user_id,item_id=item_id,target=target)
    elif name=='cosine':
        model = tc.item_similarity_recommender.create(train_data, user_id=user_id, item_id=item_id, target=target,similarity_type='cosine')
    elif name == 'pearson':
        model = tc.item_similarity_recommender.create(train_data,user_id=user_id,  item_id=item_id,  target=target,similarity_type='pearson')
    recom = model.recommend(users=users_to_recommend, k=n_rec)
    recom.print_rows(n_display)
    return model

@app.route('/ml/recommend',methods=['GET'])
def recommned():
    customers = pd.read_csv('./data/recommend_1.csv') 
    transactions = pd.read_csv('./data/trx_data.csv')

    transactions['products'] = transactions['products'].apply(lambda x: [int(i) for i in x.split("|")])
    transactions.head(2).set_index('customerId')['products'].apply(pd.Series).reset_index()

    data = pd.melt(transactions.set_index('customerId')['products'].apply(pd.Series).reset_index(), 
               id_vars=['customerId'],
                 value_name='products') \
        .dropna().drop(['variable'], axis=1) \
        .groupby(['customerId', 'products']) \
        .agg({'products': 'count'}) \
        .rename(columns={'products': 'purchase_count'}) \
        .reset_index() \
        .rename(columns={'products': 'productId'})


    data['productId']=data['productId'].astype(np.int64)


    data_dummy = create_data_dummy(data)

    df_matrix = pd.pivot_table(data, values='purchase_count', index='customerId', columns='productId')

#df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())
    df_matrix_norm=df_matrix

# create a table for input to the modeling  
    d = df_matrix_norm.reset_index() 
    d.index.names = ['scaled_purchase_freq'] 
    data_norm = pd.melt(d, id_vars=['customerId'], value_name='scaled_purchase_freq').dropna()
    
    train_data, test_data = split_data(data)
    train_data_dummy, test_data_dummy = split_data(data_dummy)
    train_data_norm, test_data_norm = split_data(data_norm)

# constant variables to define field names include:
    user_id = 'customerId'
    item_id = 'productId'
    users_to_recommend = list(customers[user_id])
    n_rec = 10 # number of items to recommend
    n_display = 30 # to display the first few rows in an output dataset


    final_model = tc.item_similarity_recommender.create(tc.SFrame(data_norm),user_id=user_id, item_id=item_id,target='scaled_purchase_freq', similarity_type='cosine')

    recom = final_model.recommend(users=users_to_recommend, k=n_rec)
    recom.print_rows(n_display)


    json_output=[]
    temp_inc=0
    for i in recom:
        if(i['score']!=0):
            i['key']=temp_inc
            temp_inc=temp_inc+1
            json_output.append(i)


    final_output=json.dumps(json_output)
    return final_output






if __name__=='__main__':
    app.run(debug=True)