# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:35:19 2017

@author: anbarasan.selvarasu
"""
from configurations import Configurations as cfg
from sparsematrix_V2 import UtilityMatrix
import numpy as np
import pandas as pd
import pickle

class ModelBased(object):
    
    def __init__(self):
        pass
    
    def model_based(self,user_id,train_data,user_params,item_params,item_mapping,num_items):
        past_items = train_data[user_id, :].indices

                   
        preference = np.dot(user_params[user_id,:], item_params.T)
        sorted_preference_ix = np.argsort(preference)[::-1]
        sorted_preference_fltrd = [x for x in sorted_preference_ix if x not in past_items] 
        sorted_preference_ix_top5 = sorted_preference_fltrd[:5]        
       
        item_names = []
        keys_list = list(item_mapping.keys())
        values_list = list(item_mapping.values())
        for w in sorted_preference_ix_top5:
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
    
    processed_file = pd.read_csv(cfg.PROCSD_FILE)
    item_mapping = pickle.load(open(cfg.ITEM_MAPPING,"rb"))
    user_mapping = pickle.load(open(cfg.USER_MAPPING,"rb"))
    user_params = pickle.load(open("model/user_params_all_50f.pickle","rb"))
    item_params = pickle.load(open("model/item_params_all_50f.pickle","rb"))
    
    
    utility = UtilityMatrix()
    sp_data, confidence_data = utility.get_sparse_matrix(processed_file, True) 
    
    model = ModelBased()
    
    user_id = int(input("User ID: "))
    if user_id in user_mapping.keys():
        mapped_user_id = user_mapping[user_id]  
        model.model_based(mapped_user_id,sp_data,user_params,item_params, item_mapping,5)
    else:
        print(" Invalid User_id...! ")
        print("Enter User_id that has atleast five transactions..! ")
    

if __name__ == "__main__":
    main()