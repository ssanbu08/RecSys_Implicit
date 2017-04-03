# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 13:17:46 2017

@author: anbarasan.selvarasu
"""
from time import time
from sparsematrix_V2 import UtilityMatrix
import pandas as pd

class EvaluationPB(object):
    
    def __init__(self):
        pass
        
    def popularity_based(self,sp_test):
        
        base_df = pd.read_csv("data/base_df_8020.csv")
        item_freq = base_df['item_id_2'].value_counts().index.tolist()
        
        # Top 5 Frequent Items
        top_1 = item_freq[0]       
        top_2 = item_freq[0:2]
        top_3 = item_freq[0:3]
        top_4 = item_freq[0:4]
        top_5 = item_freq[0:5]
        
        top_1_list = []
        top_2_list = []
        top_3_list = []
        top_4_list = []
        top_5_list = []
        
        top_prop = []
        for user in range(sp_test.shape[0]):
           
                                        
            look_up_t = time()
            test_items = sp_test[user,:].indices.tolist()
            top_1_flag = top_1 in test_items                    
            top_2_flag = any([x in top_2 for x in test_items])
            top_3_flag = any([x in top_3 for x in test_items])
            top_4_flag = any([x in top_4 for x in test_items])
            top_5_flag = any([x in top_5 for x in test_items])
            print("time taken for sirting preference",(time()-look_up_t))
              
            top_1_list.append(top_1_flag)
            top_2_list.append(top_2_flag)
            top_3_list.append(top_3_flag)
            top_4_list.append(top_4_flag)
            top_5_list.append(top_5_flag)
            
        top_prop.append(sum(top_1_list) / sp_test.shape[0])
        top_prop.append(sum(top_2_list) / sp_test.shape[0])
        top_prop.append(sum(top_3_list) / sp_test.shape[0])
        top_prop.append(sum(top_4_list) / sp_test.shape[0])
        top_prop.append(sum(top_5_list) / sp_test.shape[0])
        
        return top_prop
        
def main():
    utility = UtilityMatrix()
    train_data, test_data = utility.load_test_train("data/train_8020.csv", "data/test_8020.csv")
    
    # load test data   
    sp_train, confidence_train = utility.get_sparse_matrix(train_data, True)    
    sp_test, confidence_test = utility.get_sparse_matrix(test_data, True)    
    
    evalpb = EvaluationPB()
    user_prop = evalpb.popularity_based(sp_test)
    
    
if __name__ == "__main__":
    main()
