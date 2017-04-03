# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:01:03 2017

@author: anbarasan.selvarasu
"""
from configurations import Configurations as cfg
from sparsematrix_V2 import UtilityMatrix
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from time import time
import pickle


class ALS(object):
    
    def __init__(self):
            self.n_epochs = 5
            self.lmbda = 0.01
            self.learning_rate = 0.01
            self.latent_features = 50
            
            
    def als_3(self,train_data,confidence):
#==============================================================================
#         print("Train Data#############")
#         print(train_data)
#         print(train_data.indices)
#         print(train_data.data)
#         print(train_data.indptr)
#         print(train_data.indices[train_data.indptr[12]:train_data.indptr[13]])
#         
#         print("Confidence Data#############")
#         print(confidence)
#         print(confidence.indices)
#         print(confidence.data)
#         print(confidence.indptr)
#==============================================================================
        
        num_users = train_data.shape[0]        
        num_items = train_data.shape[1]
        
        
        # Randomly initializae the trainable parameters        
        init_wt =  0.01 # Try uniform distribution initialization instead of random one
        
        user_params = 3 * np.random.rand(train_data.shape[0],self.latent_features) 
        item_params = 3 * np.random.rand(train_data.shape[1],self.latent_features)
        #user_params = 
        #item_params = 
        
        #sp_train = sp.sparse.csr_matrix( (data,(row_ix,col_ix)))
        
        alpha = 20
        alpha_confidence = confidence.multiply(alpha)
        alpha_confidence_coo = alpha_confidence.tocoo()
        diagonal_matrix_latent = np.eye(self.latent_features)
        ones_items = [1]*num_items
        ones_users = [1]*num_users
        diagonal_matrix_items = sp.sparse.csr_matrix((ones_items, (range(num_items),range(num_items))))
        diagonal_matrix_users = sp.sparse.csr_matrix((ones_users, (range(num_users),range(num_users))))
        
        train_data_coo = train_data.tocoo()
        # Repeat until convergence
        for epoch in range(self.n_epochs):
            print("Running Epoch :" ,epoch+1)
            t0 = time()
            #1.Fix item parameters  and estimate user parameters
            #2. Item parameters becomes constant and differentiate loss function w.r.t user parameters
            #3.Iterate for every user ,the ratings associated with that user . i.e ratings_of_user_u
            
            # STEP 1: Re-Compute all user factors
            # calculate [item_factors.T * item_factors] to get (f*n).(n*f) = (f*f) 
            yty = np.dot(item_params.T, item_params) # (f*f)
            
            for i in set(train_data_coo.row):
                '''
                For a given user, get  preferences towards all items.
                preference by m users towards the item j.
                '''
                #print(i)
                #print("###",train_data[i,:].indices)
                u_time = time()
                
                user_pref = train_data[i,:].todense() # Preference array: Boolean ,0 or 1
                              
                item_conf = np.add(1, alpha_confidence[i,:].A) # Smooth Confidence and Convert to dense array
                item_conf_diag = sp.sparse.csr_matrix((item_conf[0],(range(num_items),range(num_items)))) # Create a diagonal matrix of confidence level for all items and not for only rated items
                item_conf_fltr = (item_conf_diag - diagonal_matrix_items)
                inter_1 = item_conf_fltr.dot(item_params)
                inter_2= inter_1.transpose().dot(item_params)
                
                yty_1 = yty + inter_2
                # Regularization               
                reg = self.lmbda * diagonal_matrix_latent
                # Term 1
                Ai =  yty_1 + reg
                
                # Term 2
                # Sub 1
                term_2_1 = item_conf_diag.dot(item_params)
                Vi = term_2_1.transpose().dot(user_pref.T)
                # Update User Params                              
                user_params[i,:] = np.linalg.solve(Ai,Vi).T
                
                print("User %d is completed in %0.3f" %(i,(time()-u_time)))
                
            print("User params completed")
            
            train_data_t = train_data.transpose() # Do transpose to get indices of users prefrred that item
            train_data_coo_t = train_data_t.tocoo()
            # Transpose of sparse matrix of Confidence levels
            alpha_confidence_t = alpha_confidence.transpose()
            
            # Step 2: Recompute all item factors,keeping user_factors as constant
            # Calculate [user_factors.T * user_factors] to get (f*m).(m*f) = (f*f)
            xtx = np.dot(user_params.T, user_params)
            
            for j in set(train_data_coo_t.row):
                '''
                For a given item, get  all users' preference towards this item.
                preference by m users towards the item j.
                
                '''
                i_time = time()
                #print(j) # item
                #print("###",train_data_t[j,:].indices)# users preferrred this particular item
                # get the list of all users' preference for this item
                item_pref = train_data_t[j,:].todense() # Preference array: Boolean ,0 or 1
                              
                user_conf = np.add(1, alpha_confidence_t[j,:].A) # Smooth Confidence and Convert to dense array
                user_conf_diag = sp.sparse.csr_matrix((user_conf[0],(range(num_users),range(num_users)))) # Create a diagonal matrix of confidence level for all items and not for only rated items
                user_conf_fltr = (user_conf_diag - diagonal_matrix_users) # Pick only those users, who has preference
                user_inter_1 = user_conf_fltr.dot(user_params )
                user_inter_2= user_inter_1.transpose().dot(user_params)
                
                xtx_1 = xtx + user_inter_2 # [X.T*C*X] = [ (X.T*X) + (X.T*{C-I}*X) ]
                # Regularization               
                reg = self.lmbda * diagonal_matrix_latent # regularization
                # Term 1
                Aj =  xtx_1 + reg # Aj's
                
                # Term 2
                # Sub 1
                term_2_1 = user_conf_diag.dot(user_params)
                Vj = term_2_1.transpose().dot(item_pref.T)
                # Update User Params                              
                item_params[j,:] = np.linalg.solve(Aj,Vj).T
                print("Item %d is completed in %0.3f" %(j,(time()-i_time)))
            print("item params completed")
            print("Epoch  completed " ,epoch+1)
            print("completed in %0.3f"%(time()-t0))
                 
        #print("[Epoch %d/%d] train error: %f, test error: %f" \
        #%(epoch+1, self.n_epochs, train_rmse, test_rmse))"
        return user_params,item_params   
            
   
        
def main():
    print("ALS 2..")
    utility = UtilityMatrix()
    
    train_data, test_data = utility.load_test_train("data/train_8020.csv", "data/test_8020.csv")
    sp_train, confidence_train = utility.get_sparse_matrix(train_data, True)
    
    als = ALS()
    
    user_params,item_params = als.als_3(sp_train,confidence_train)
    
    pickle.dump(user_params, open('model/user_params_all_50f.pickle', 'wb'))
    pickle.dump(item_params, open('model/item_params_all_50f.pickle', 'wb'))
    
    
    #predicted_preference = np.dot(user_params,item_params.T)
    print("predicted Preference..... ")
    
if __name__ == "__main__":
    main()
        
    