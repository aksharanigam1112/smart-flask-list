import datetime
import json
from src.DB_helper.DB_fetch import db_fecth
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import math
from scipy import stats


class prediction:
    kernel = 'rbf'
    C = 1e3
    gamma = 0.1
    n_estimators = 5
    random_state = 10

    def __init__(self,userid):
        self.db=db_fecth()
        self.userid=userid

    def prefetch(self, item_id_dict, item_info):
        for x in item_info:
            for y in x["Transaction"]:
                if item_id_dict.get(y['item_id']) is not None:
                    dates = []
                    quantity = []
                    item_trans = y['item_transactions']
                    for z in item_trans:
                        dates.append(z['date'])
                        quantity.append(z['quantity'])
                    item_id_dict[y['item_id']]["dates"] = dates
                    item_id_dict[y['item_id']]["quantity"] = quantity
        return item_id_dict

    def removeOutliers(self, frequency, threshold):
        modified_freq = []
        modified_quantity = []

        for freq, arr in frequency.items():
            if (len(arr) == 1):
                modified_freq.append(freq)
                modified_quantity.append(arr[0])
            else:
                z = stats.zscore(arr)
                for idx in range(0, len(z)):
                    if np.isnan(z[idx]) == True:
                        modified_freq.append(freq)
                        modified_quantity.append(arr[idx])
                    elif abs(z[idx]) < threshold:
                        modified_freq.append(freq)
                        modified_quantity.append(arr[idx])
        return modified_freq, modified_quantity

    def get_dates_quantity(self, dates, quantity, remove_outliers=0, outliers_threshold=float(0)):
        dates_arr = []
        frequency_distribution = {}
        for i in range(1, len(dates)):
            frequency = (dates[i] - dates[i - 1]).astype('int64')
            dates_arr.append(frequency)
            frequency_distribution[frequency] = []

        quantity = quantity[1:]

        if remove_outliers == 1:
            for idx in range(0, len(dates_arr)):
                frequency_distribution[dates_arr[idx]].append(quantity[idx])
            modified_dates, modified_quantity = self.removeOutliers(frequency_distribution, outliers_threshold)
            modified_dates = np.array(modified_dates).astype('int64')
            modified_dates = np.reshape(modified_dates, (len(modified_dates), 1))
            return modified_dates, modified_quantity
        else:
            dates_arr = np.array(dates_arr).astype('int64')
            dates_arr = np.reshape(dates_arr, (len(dates_arr), 1))
        return (dates_arr, quantity)

    def calc_error(self, predicted, actual):
        error = 0
        for i in range(0, len(actual)):
            error = error + ((actual[i] - predicted[i]) * (actual[i] - predicted[i]))
        error = error / len(actual)
        return math.sqrt(error)

    def prepare_svr_model(self):
        svr_rbf = SVR(kernel=self.kernel, C=1e3, gamma=0.1)
        return svr_rbf

    def prepare_random_forest_model(self):
        random_forest = RandomForestRegressor(n_estimators=5, random_state=10)
        return random_forest

    def algo(self, dates, quantity, gap):
        dates = np.array(dates).astype('datetime64[D]')
        # preparing frequncy array(dates_arr)
        (dates_arr, quantity) = self.get_dates_quantity(dates, quantity, 0, 1.5)

        # INITIALISING THE MODEL

        svr_rbf = self.prepare_svr_model()
        random_forest = self.prepare_random_forest_model()

        # FITTING THE MODEL
        svr_rbf.fit(dates_arr, quantity)
        random_forest.fit(dates_arr, quantity)

        # READING THE CURRENT TIMESTAMP TO FIND THE GAP
        predict_dates = gap
        predict_dates = np.reshape(predict_dates, (1, 1))

        # PREDICTING FROM THE FITTED MODEL
        if predict_dates > max(dates_arr):
            maximum = max(dates_arr)[0]
            k = 0
            max_quant = 0
            for i in dates_arr:
                if i[0] == maximum:
                    if quantity[k] > max_quant:
                        max_quant = quantity[k]
                k += 1
            return math.ceil(max_quant)

        rbf = svr_rbf.predict(dates_arr)
        rf = random_forest.predict(dates_arr)  # rf=Random Forest

        rounded_rbf = []
        rounded_rf = []

        for i in range(0, len(rbf)):
            rounded_rbf.append(math.ceil(rbf[i]))
            rounded_rf.append(math.ceil(rf[i]))

        error_rbf = self.calc_error(rounded_rbf, quantity)
        error_rf = self.calc_error(rounded_rf, quantity)
        # print(error_rbf,error_rf) -->> ERROR PRINTING
        if error_rbf <= error_rf:
            return svr_rbf.predict(predict_dates)[0]
        else:
            return random_forest.predict(predict_dates)[0]

    def predict(self):
        transaction = self.db.read_json( "transactions")
        recent_purchases = self.db.read_json("Recent_purchases")  # Getting the rta table

        # itemlist = db.itemlist
        user_dict = {}
        user_dict["cust_id"] = int(self.userid)
        item_info = transaction.find(user_dict, {"Transaction.item_transactions.date": 1,
                                                 "Transaction.item_transactions.quantity": 1, "Transaction.item_id": 1,
                                                 "_id": 0})
        itemDetails = recent_purchases.find(user_dict, {'_id': 0})  # Mongo query

        output = []
        item_id_dict = {}  # Stores the item and dates and quantity array
        item_info_dict = []  # stores the avg , last_date and item_id

        for item in itemDetails:
            for one_item in item['recents']:
                item_obj_dict = {}
                item_id_dict[one_item["item_id"]] = {}
                item_obj_dict["item_id"] = one_item["item_id"]
                item_obj_dict["avg"] = one_item["avg"]
                item_obj_dict["last_date"] = one_item["last_date"]
                item_info_dict.append(item_obj_dict)

        item_id_dict = self.prefetch(item_id_dict, item_info)
        for one_item in item_info_dict:
            avg = one_item['avg']  # Fetch the avg of an item for a particular user
            datetimeobj = datetime.datetime.now()
            date = datetimeobj.strftime("%Y") + "-" + datetimeobj.strftime("%m") + "-" + datetimeobj.strftime("%d")

            last_date_of_purchase = one_item['last_date']

            t = (datetime.datetime.strptime(date, "%Y-%m-%d") - datetime.datetime.strptime(last_date_of_purchase,
                                                                                           "%Y-%m-%d"))
            t = t.days
            avg = math.ceil(avg)
            if (avg != 0 and ((avg) - 2) <= t and t <= (avg + 3)):
                item_pred = {}
                itemid = one_item['item_id']
                item_dict = item_id_dict.get(itemid)

                if (len(item_dict["dates"]) > 2 and len(item_dict["quantity"]) > 2):
                    ans = self.algo(dates=item_dict["dates"], quantity=item_dict["quantity"], gap=t)
                    dictionary = dict({'item_id': itemid})

                    # itemName = itemlist.find( dictionary, {'item_name':1 ,'item_id':1, '_id':0})

                    item_pred['itemID'] = itemid
                    # for name in itemName['item_name']:
                    item_pred['itemName'] = "Test_items"
                    item_pred['Quantity'] = round(ans)
                    output.append(item_pred)

        json_output = json.dumps(output)
        return json_output
