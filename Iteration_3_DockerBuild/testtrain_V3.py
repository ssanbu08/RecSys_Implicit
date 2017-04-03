# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:35:21 2017

@author: anbarasan.selvarasu
"""

from configurations import Configurations as cfg

import pandas as pd
import numpy as np
import scipy as sp
from sklearn import cross_validation as cv
from dateutil import relativedelta as rdelta
from time import time
import pickle

class DataPreparation(object):
    '''
    This class prepares the data for our models.
    '''
    
    def __init__(self):
        self.file_path = cfg.BASE_FILE
        self.testsize = cfg.TRAIN_TEST_SPLIT
        self.items_threshold = cfg.ITEM_THRESHOLD
        
        
    def get_data(self):
        ''' This function performs the following activities 
            1. reads the raw file 
            2. Create a new field item_2 using the field- item
            3. Create a mapping between the item_id and item_name
            4. Count the frequency of purchase of each product
            5. Add a binary flag ,to indicate whether a product is purchased
            
            @input : None
            @output : 1.for each user ,the products purchased and frequency of purchases
                      2. Dictionary of mapping between column index and product_id'''
               
        
        base_df = pd.read_table(self.file_path,sep = ";",header = None \
                                , names = ['timestamp','user_id','item_id'])
        
        base_df["timestamp"] = pd.to_datetime(base_df["timestamp"], unit='s')
        
        # Sort and get the firsts n users
        base_df.sort_values(by = ['user_id','timestamp'],inplace = True)  
        
        uniq_items = base_df['item_id'].unique()
        num_items = uniq_items.shape[0]
        items_ix = list(range(num_items))        
        item_mapping = dict(zip(uniq_items,items_ix))
        base_df.loc[:, 'item_id_2'] = base_df['item_id'].apply(lambda x: item_mapping[x])
                       
        utility_df = base_df.groupby(['user_id','item_id_2']).size().reset_index().rename(columns={0:'freq'})
        utility_df.loc[:, 'binary'] = np.ones(utility_df.shape[0])  
        
        
        print("get data completed....")
        return utility_df,item_mapping
        
        
    def test_train_split(self,data):
        '''
        @input : processed_data
        @output : training_data, test_data
        
        Split each user's activities into train and test, such that each user is available
        in both training and test set.
        
        thi script will take approximately 50 minutes to complete
        
        '''
        train = pd.DataFrame()
        test = pd.DataFrame()
        #Filter users with atleast 2 transaction      
        df =  data.groupby("user_id").filter(lambda x: len(x) > self.items_threshold)  # Have to decide on this               
        unique_users = df['user_id'].unique()
        num_users = unique_users.shape[0]
        users_ix = list(range(num_users))
        user_mapping = dict(zip(unique_users,users_ix))
        df.loc[:, 'user_id_2'] = df['user_id'].apply(lambda x: user_mapping[x])
        
        t0 = time()
        print("processing for each user started")
        counter = 0
        train_table = []
        test_table = []
        for user in unique_users :
            temp_df = df[ df['user_id']== user]                        
            temp_train, temp_test = cv.train_test_split(temp_df,test_size = self.testsize)
            train_table.append(temp_train)
            test_table.append(temp_test)       
            
            counter += 1
            if (counter % 1000) == 0:
                print("splitted %d users in %0.3f time" %(counter,time() - t0))
            #print("processed User : %d" % user)
        train = pd.concat(train_table)
        test = pd.concat(test_table)
        
        print("Time taken :test-train split %0.3f" %(time() - t0))        
        # Time taken :test-train split 3030.886
        
        return train, test, user_mapping, df
        
    

def main():
    print("Preparing Test Train Matrix....")
    
    preparefile = DataPreparation()
    
    ## Get the Base Utility DF ###
    utility_df,item_mapping = preparefile.get_data()  
        
    ## Test Train Split ###
    train_df, test_df,user_mapping, df_80_20 = preparefile.test_train_split(utility_df)    
    
    ## Filtered Utility DF and the Corresponding Test and Train data ##
    utility_df.to_csv("data/utility.csv", index = False)
    df_80_20.to_csv("data/base_df_8020.csv", index = False)
    train_df.to_csv("data/train_8020.csv", index = False)
    test_df.to_csv("data/test_8020.csv", index = False)      
    pickle.dump(user_mapping, open('data/user_mapping.pickle', 'wb'))
    pickle.dump(item_mapping, open('data/item_mapping.pickle', 'wb'))
    
    
    

if __name__ == "__main__":
    main()
