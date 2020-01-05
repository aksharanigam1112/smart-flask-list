from src.DB_helper.DB_fetch import db_fecth
import turicreate as tc
import json

class recommendation:
    similarity_type = 'cosine'
    target = "frequency"
    def __init__(self,user_id):
        self.db = db_fecth()
        self.user_id=user_id

    def get_user(self):
        """
        To get the overall users list
        :return:
        """
        users_table = self.db.read_json("customers")
        res = users_table.find({}, {"_id": 0})
        users = []
        for i in res:
            users.append(str(i["cust_id"]))
        return users

    def get_data(self, users):
        """
        To get the the data for recommendation
        :param users:
        :return:
        """
        user_data = []  # output 1
        item_data = []  # output 2
        target_data = []  # output 3

        transactions_table = self.db.read_json("transactions")

        for user in users:
            # An object to find in the table
            query = {}
            query["cust_id"] = int(user)

            res = transactions_table.find(query, {"_id": 0, "cust_id": 0})  # ignoring the _id and cust_id fields
            for obj in res:
                for enteries in obj["Transaction"]:
                    user_data.append(str(user))
                    item_data.append(str(enteries["item_id"]))
                    target_data.append(len(enteries["item_transactions"]))
        max_target = max(target_data)
        min_target = min(target_data)
        if max_target != min_target:
            for i in range(0, len(target_data)):
                target_data[i] = (target_data[i] - min_target) / (max_target - min_target)

        return user_data, item_data, target_data

    def prepare_model(self,user_data,item_data,target_data):
        sf = tc.SFrame({'user_id': user_data, 'item_id': item_data, 'frequency': target_data})
        model = tc.item_similarity_recommender.create(sf, target=self.target, similarity_type=self.similarity_type)
        return model

    def recommend(self):
        users = self.get_user()
        # users=[25]
        user_data, item_data, target_data = self.get_data(users)
        user_arr = []
        user_arr.append(str(self.user_id))
        model=self.prepare_model(user_data=user_data,item_data=item_data,target_data=target_data)

        # recom=m.recommend(users,k=10) UNCOMMENT IF want to test for all users

        recom = model.recommend(user_arr, k=10)
        output = {}
        output["item_id"] = []

        for items in recom["item_id"]:
            output["item_id"].append(items)

        return json.dumps(output)
