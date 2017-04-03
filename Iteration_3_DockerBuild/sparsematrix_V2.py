# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:53:41 2017

@author: anbarasan.selvarasu
"""

import pandas as pd
import scipy as sp
from scipy import sparse
import numpy as np

class UtilityMatrix(object):
    '''
    This class takes the dataset and returns in a matrix format,
    Rows => Users
    Columns => Items
    Values => Ratings/Frequency of Purchase
    '''
    
    def __init__(self):
        pass
        self.tot_users = 699157
        self.tot_items = 81903
    
    def load_test_train(self,train_file,test_file):
        '''
        Load the test and train data from csv files
        
        '''
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        return train_data, test_data        
        
    def get_dense_matrix(self,data):
        '''
        @ input : data
        @ output : data in dense matrix format
        get the input data in sparse matrix format
        
        '''    
        utility_df = self.get_data()
        
        n_users = utility_df['user_id'].unique().shape[0]
        n_business = utility_df['item_id'].unique().shape[0]
        
        print("Num of users: %d | Num of Items: %d " %(n_users,n_business))
        train_data =   utility_df      
        
        user_id_ix = utility_df['user_id'].unique()
        business_id_ix = utility_df['item_id'].unique()

        train_data_matrix = pd.DataFrame(np.zeros((n_users, n_business))
                                        ,index = user_id_ix
                                         ,columns = business_id_ix) 
                                        
        for i in range(train_data.shape[0]):
             train_data_matrix.ix[train_data.iloc[i]['user_id'],train_data.iloc[i]['item_id']] = train_data.iloc[i]['binary']
         
       
        return train_data_matrix
        
    def get_sparse_matrix(self,data, is_mf = False):  
        '''
        @ input : data , is_matrixFactorization flag
        @ output : returns data in sparse matrix format
        '''
        n_users = data['user_id_2'].unique().shape[0]
        n_items = data['item_id_2'].unique().shape[0]        
        print("Num of train users: %d | Num of train Items: %d " %(n_users,n_items)) 

        data_df = pd.DataFrame(data)               
        row_ix = data_df['user_id_2'].values 
        col_ix = data_df['item_id_2'].values 
        data = data_df['binary']
        sp_train = sparse.csr_matrix( (data,(row_ix,col_ix)),shape = (n_users, self.tot_items) )
        
        if is_mf:
            row_ix = data_df['user_id_2'].values 
            col_ix = data_df['item_id_2'].values 
            data = data_df['freq']
            confidence_train = sparse.csr_matrix( (data,(row_ix,col_ix)), shape = (n_users, self.tot_items)) 
            return sp_train,confidence_train
                    
        return sp_train


def main():
    print("Sparse matric generation.....")
    utility = UtilityMatrix()
    train_file = "data/train_50.csv"
    test_file = "data/test_50.csv"
    train_data, test_data = utility.load_test_train(train_file, test_file)
    sp_train, confidence_train = utility.get_sparse_matrix(train_data, True)
    sp_test, confidence_train = utility.get_sparse_matrix(test_data, True)
    #unique_users (699157,)
    #unique_items (81903,)
    
    
    
if __name__=="__main__":
    main()