# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 12:20:12 2017

@author: anbarasan.selvarasu
"""
from itemsimilarity import ItemSimilarity
from usersimilarity import UserSimilarity
from sparsematrix_V2 import UtilityMatrix
from time import time

class EvaluationNB(object):
    
    def __init__(self):
        pass
    
    def neigh_item_based(self,sp_train,sp_test,_type):
        
        if _type == 'item':
            sim = ItemSimilarity(sp_train)
            sim_matrix = sim.item_sim_sparse()
        else:
            sim = UserSimilarity(sp_train)
            sim_matrix = sim.user_sim_sparse()
        
        top_1_list = []
        top_2_list = []
        top_3_list = []
        top_4_list = []
        top_5_list = []
        
        top_prop = []        
        for user in range(sp_test.shape[0]):
            look_up_t = time()
            top_5 = sim.get_top_n(sim_matrix,user,5)
            #print("top5 time",( time()-look_up_t))
            
            look_up_t = time()
            test_items = sp_test[user,:].indices.tolist()
            top_1_flag = top_5[:1] in test_items                    
            top_2_flag = any([x in top_5[:2] for x in test_items])
            top_3_flag = any([x in top_5[:3] for x in test_items])
            top_4_flag = any([x in top_5[:4] for x in test_items])
            top_5_flag = any([x in top_5[:5] for x in test_items])
            print("user:%d time : %0.3f"%(user, time()-look_up_t))
              
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
    
    evalnb = EvaluationNB()
    #user_based_prop = evalnb.neigh_item_based(sp_train, sp_test, 'user')
    item_based_prop = evalnb.neigh_item_based(sp_train, sp_test, 'item')
    
    
    
if __name__ == "__main__":
    main()
            