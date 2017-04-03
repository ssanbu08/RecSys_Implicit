# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:55:31 2017

@author: anbarasan.selvarasu
"""


from testtrain_V3 import DataPreparation
from configurations import Configurations as cfg
from sparsematrix_V2 import UtilityMatrix

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import jaccard_similarity_score
from math import sqrt
import pickle
from collections import Counter

class UserSimilarity(object):
    
    def __init__(self,train_data):
        
        print("#####User Similarity#####")
        self.sparse_train_data = train_data
        #self.user_mapping = user_mapping
        #self.item_mapping = item_mapping
          
              
    def user_sim_sparse(self):
        
        similarities_sparse = cosine_similarity(self.sparse_train_data,dense_output=False)
        return similarities_sparse
        
    def get_top_n(self,sparse_cos_similarity,user_id,num_items):
        
        #sparse_cos_similarity = self.user_sim_sparse(self.sparse_train_data)
        
        #user_id = user_id - 1 # To match with the corresponding index
    
        # Get items already purchased by the given user
        past_items = self.sparse_train_data[user_id,:].indices
        
        # Get users similar to the user based on similarity measure    
        sim_users = sparse_cos_similarity[user_id,:].indices # index of similar users
        sim_score= sparse_cos_similarity[user_id,:].data # sim_score with the similar users
        user_score = dict(zip(sim_users,sim_score))
        user_score_sorted = sorted(user_score, key=user_score.get, reverse=True)
        user_score_sorted.remove(user_id)# Sort the users in decreasing order of similarity
        user_score_sorted_top5 = user_score_sorted[:5]
        # Get the items bought by similar users
        similar_items = []
        for user in user_score_sorted_top5:
            items = self.sparse_train_data[user,:].indices
            similar_items.append(items.tolist())
        
        # Filter for items already bought by the user and remove duplicates
        similar_items_all = [y for x in similar_items for y in x if y not in past_items]
        similar_items_counter = Counter(similar_items_all)
        simila_items_majvote = [i[0] for i in similar_items_counter.most_common()]
        #similar_items_unique = sorted(set(similar_items_all), key = similar_items_all.index)
        
        return simila_items_majvote
#==============================================================================
#         
#         
#         ## Get top n products
#         item_names = [self.item_mapping[w] for w in similar_items_unique]
#         print("Displaying Top %d Recommended items..." %num_items)
#         for i in range(len(item_names)):
#             if i < num_items:
#                 print(i,item_names[i])
#             else: 
#                 break
#==============================================================================
            
    def display_top_n(self,sparse_cos_similarity,user_id,num_items,item_mapping):
        
        similar_items_unique = self.get_top_n(sparse_cos_similarity,user_id,num_items)
        
         ## Get top n products
        item_names = []
        keys_list = list(item_mapping.keys())
        values_list = list(item_mapping.values())
        for w in similar_items_unique:
            item = keys_list[values_list.index(w)] 
            item_names.append(item)
        
        #item_names = [self.item_mapping[w] for w in similar_items_unique]
        print("Displaying Top %d Recommended items..." %num_items)
        print()
        for i in range(len(item_names)):
            if i < num_items:
                print(i,item_names[i])
            else: 
                break
    
    
    
def main():
    print("Neighbourhood Based Method")
 
########### USER SIMILARITY: SPARSE MATRIX STARTS HERE ##############   
    utility = UtilityMatrix()
    processed_file = pd.read_csv(cfg.PROCSD_FILE)
    item_mapping = pickle.load(open(cfg.ITEM_MAPPING,"rb"))
    user_mapping = pickle.load(open(cfg.USER_MAPPING,"rb"))
       
    # Get the data in sparse matrix format 
    sp_data, confidence_data = utility.get_sparse_matrix(processed_file, True)     
    
    usersim = UserSimilarity(sp_data)
    user_id = int(input("User ID: "))
    if user_id in user_mapping.keys():
        mapped_user_id = user_mapping[user_id]    
        sparse_cos_similarity = usersim.user_sim_sparse()
        usersim.display_top_n(sparse_cos_similarity,mapped_user_id,5,item_mapping)
    else:
        print(" Invalid User_id...! ")
        print("Enter User_id that has atleast five transactions..! ")
    
    
    
if __name__ == "__main__":
    main()