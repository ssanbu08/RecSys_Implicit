# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 13:48:24 2017

@author: anbarasan.selvarasu
"""
from time import time
from sparsematrix_V2 import UtilityMatrix
import numpy as np
import pickle

class EvaluationMB(object):
    
    def __init__(self):
        pass
    
    

    def model_based(self,user_params,item_params, sp_test,confidence_test):
        
        # Compute estimated ranking on the test set
        rank = []
        for user in range(user_params.shape[0]):
            # obtain the latent factors for corresponding user
            preference = np.dot(user_params[user,:], item_params.T)
            #preference_score = [(100-stats.percentileofscore(preference, a, 'rank')) for a in preference.tolist()]
            
            # prefernce score 0% indicates high preference
            # preference score 100% indicates low preference
            # Any score beyond 50% is not a good recommendation
            test_items = sp_test[user,:].indices # this gives the list of items for that user in test set
            confidence_test_items = confidence_test[user,:].data
            p_score_test_items = preference[test_items] # get the preference score for items in test data
            estimated_rank = np.dot(confidence_test_items,p_score_test_items) / sum(confidence_test_items)
            rank.append(estimated_rank)
            
        estimated_rank = sum(rank)##  lower values of estimated_rank is preferred
        
        ### Approach 2 - Cumulative Distribution Function
        ### get top 5% of recommended items, and  find the number of items that were actually watched 
        ### Split the top 5% into 5 deciles  and calculate the number of purchases in each decile
        
        ### Find the probability of that a show is watched from top1% for each user ,continue till top 5%
        ### repeat for every user and take average
        
        ### get the top 1% recommendation for every user
        top_1_list = []
        top_2_list = []
        top_3_list = []
        top_4_list = []
        top_5_list = []
        top_prop = []
        t0 = time()
        for user in range(user_params.shape[0]):
            # obtain the latent factors for corresponding user
            preference = np.dot(user_params[user,:], item_params.T)
            #preference_score = [(100-stats.percentileofscore(preference, a, 'rank')) for a in preference]
            
            # sort the preference score and get the item_id for that scores
            # top 1%,top 2%, top 3%....
    #==============================================================================
    #         sort_t = time()
    #         sorted_preference_ix = sorted(range(len(preference)), key=lambda k: preference[k]
    #                                     , reverse = True)
    #         print("time taken for sirting preference",(time()-sort_t))
    #         sort_t1 = time()
    #==============================================================================
            sort_t1 = time()
            sorted_preference_ix = np.argsort(preference)[::-1]
            print("time taken for sirting preference",(time()-sort_t1))
             # number of items purchased in top 1%
            top_1 = sorted_preference_ix[0]       
            top_2 = sorted_preference_ix[0:2]
            top_3 = sorted_preference_ix[0:3]
            top_4 = sorted_preference_ix[0:4]
            top_5 = sorted_preference_ix[0:5]
                                        
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
            
            print("Analyzed user :%d in time: %0.3f" %(user,time()-t0))
            
        top_prop.append(sum(top_1_list) / user_params.shape[0])
        top_prop.append(sum(top_2_list) / user_params.shape[0])
        top_prop.append(sum(top_3_list) / user_params.shape[0])
        top_prop.append(sum(top_4_list) / user_params.shape[0])
        top_prop.append(sum(top_5_list) / user_params.shape[0])
        
        return top_prop
        
def main():
    
    utility = UtilityMatrix()
    train_data, test_data = utility.load_test_train("data/train_8020.csv", "data/test_8020.csv")
    
    # load test data   
    sp_train, confidence_train = utility.get_sparse_matrix(train_data, True)    
    sp_test, confidence_test = utility.get_sparse_matrix(test_data, True)
    
    # Sample sp_test to match with that of user_params
    #sp_train = sp_train[:1000]
    #sp_test = sp_test[:1000]
    #confidence_test = sp_test[:1000]
    
    # load user_params and Item_params from pickle files
    user_params = pickle.load(open("model/user_params_all_50f.pickle","rb"))
    item_params = pickle.load(open("model/item_params_all_50f.pickle","rb"))
    
    ### Matrix Factorization Based ####
    evalmb = EvaluationMB()
    top_prop = evalmb.model_based(user_params, item_params, sp_test, confidence_test)
    
if __name__ =="__main__":
    main()