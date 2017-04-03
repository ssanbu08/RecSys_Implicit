# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:22:55 2017

@author: anbarasan.selvarasu
"""

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
from collections import defaultdict
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class ItemSimilarity(object):
    
    def __init__(self,train_data):
        
        print("##### Item Similarity#####")
        self.sparse_train_data = train_data
        #self.item_mapping = item_mapping
    
        
    def item_sim_sparse(self):
        similarities_sparse = cosine_similarity(self.sparse_train_data.transpose(),dense_output=False)
        return similarities_sparse
        
           
    def get_top_n(self,sparse_item_similarity,user_id,num_items):
        
        #sparse_item_similarity = self.item_sim_sparse(self.sparse_train_data)
        # Get items already purchased by the given user
        past_items = self.sparse_train_data[user_id,:].indices
        
       # Get items similar to items already purchased
        items_dict = defaultdict(list)
        for i in past_items:
            items = sparse_item_similarity[i,:].indices
            score= sparse_item_similarity[i,:].data
            items_score = zip(items.tolist(),score.tolist())
            items_score = sorted(items_score, key = lambda x: x[1], reverse = True)
            items_score_top5 = items_score[:5]
            for i,s in items_score_top5:
                items_dict[i].append(s)
        
        for k in items_dict.keys():
            items_dict[k] = sum(items_dict[k])
            
        item_score_sorted = sorted(items_dict, key=items_dict.get, reverse=True)
        item_score_sorted = [x for x in item_score_sorted if x not in past_items]  
        
        return item_score_sorted
#==============================================================================
#         ## Get top n products
#         item_names = [self.item_mapping[w] for w in item_score_sorted]
#         print("Displaying Top %d Recommended items..." %num_items)
#         for i in range(len(item_names)):
#             if i < num_items:
#                 print(i,item_names[i])
#             else: 
#                 break  
#==============================================================================
        
    def display_top_n(self,sparse_item_similarity,user_id,num_items,item_mapping):
        item_score_sorted = self.get_top_n(sparse_item_similarity,user_id,num_items)
        ## Get top n products
        item_names = []
        keys_list = list(item_mapping.keys())
        values_list = list(item_mapping.values())
        for w in item_score_sorted:
            item = keys_list[values_list.index(w)] 
            item_names.append(item)
        print("Displaying Top %d Recommended items..." %num_items)
        print()
        for i in range(len(item_names)):
            if i < num_items:
                print(i,item_names[i])
            else: 
                break  
        
   
    
def main():
    print("Neighborhood Based Method")
 
########### USER SIMILARITY: SPARSE MATRIX STARTS HERE ##############   
    utility = UtilityMatrix()
    processed_file = pd.read_csv(cfg.PROCSD_FILE)
    item_mapping = pickle.load(open(cfg.ITEM_MAPPING,"rb"))
    user_mapping = pickle.load(open(cfg.USER_MAPPING,"rb"))
       
    # Get the data in sparse matrix format 
    sp_data, confidence_data = utility.get_sparse_matrix(processed_file, True)     
    
    itemsim = ItemSimilarity(sp_data)
    user_id = int(input("User ID: "))
    if user_id in user_mapping.keys():
        mapped_user_id = user_mapping[user_id]  
        similarity_matrix = itemsim.item_sim_sparse()
        itemsim.display_top_n(similarity_matrix,mapped_user_id,5,item_mapping)
    else:
        print(" Invalid User_id...! ")
        print("Enter User_id that has atleast five transactions..! ")
    
if __name__ == "__main__":
    main()