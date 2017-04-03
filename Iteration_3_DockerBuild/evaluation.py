# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:56:05 2017

@author: anbarasan.selvarasu
"""

import pickle
from sparsematrix_V2 import UtilityMatrix
from evaluation_mb import EvaluationMB
from evaluation_pb import EvaluationPB
from evaluation_nb import EvaluationNB
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

# Load Test and Train data
utility = UtilityMatrix()
train_data, test_data = utility.load_test_train("data/train_8020.csv", "data/test_8020.csv")

# Sparse Matrix for test and train   
sp_train, confidence_train = utility.get_sparse_matrix(train_data, True)    
sp_test, confidence_test = utility.get_sparse_matrix(test_data, True)
       
# load user_params and Item_params from pickle files
user_params = pickle.load(open("model/user_params_all_50f.pickle","rb"))
item_params = pickle.load(open("model/item_params_all_50f.pickle","rb"))

### Matrix Factorization Based ####
evalmb = EvaluationMB()
top_prop_mb = evalmb.model_based(user_params, item_params, sp_test, confidence_test)


### Popularity ####
evalpb = EvaluationPB()
top_prop_pb = evalpb.popularity_based(sp_test)

### Neighbourhood Based ###
### Item Similarity ####
evalnb = EvaluationNB()
user_based_prop = evalnb.neigh_item_based(sp_train, sp_test, 'user')
item_based_prop = evalnb.neigh_item_based(sp_train, sp_test, 'item')

### Plot the Cumulative Distribution ####
### Pre-computedc values ####
#==============================================================================

# top_prop_mb = [0, 0.185, 0.2842, 0.343, 0.3833, 0.4133]
# top_prop_pb = [0, 0.0118, 0.0203, 0.0300,0.0387, 0.0457]
# user_based_prop = [0, 0.0,0.03, 0.036, 0.0416, 0.0466]
# item_based_prop = [0, 0.0, 0.0246, 0.029118, 0.0336, 0.04075]
#==============================================================================
x_label = [0,1,2,3,4,5]
label_names = ["0","Top 1","Top 2", "Top 3", "Top 4", "Top 5"]

# Figure-1
model, =plt.plot(x_label, top_prop_mb,label = 'Model Based')
item, =plt.plot(x_label, item_based_prop,label = 'Item Based')
user, =plt.plot(x_label, user_based_prop,label = 'User Based')
popularity, =plt.plot(x_label, top_prop_pb,label = 'Popularity Based')


plt.legend(handles=[model, item, user,popularity],loc='lower center', bbox_to_anchor=(0.5, 1.05)
           ,ncol = 4,fancybox=True, shadow=True)
plt.xticks(x_label, label_names)
plt.xlabel('Top Items Recommended')
plt.ylabel('Probability')
plt.title("Cumulative distribution function")    
plt.show()

# Figure-2
item, =plt.plot(x_label, item_based_prop,label = 'Item Based')
user, =plt.plot(x_label, user_based_prop,label = 'User Based')
popularity, =plt.plot(x_label, top_prop_pb,label = 'Popularity Based')


plt.legend(handles=[ item, user,popularity],loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.xticks(x_label, label_names)
plt.xlabel('Top Items Recommended')
plt.ylabel('Probability')
plt.title("Cumulative distribution function")  
plt.show()






        
    
    
    
    

