# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:45:45 2017

@author: anbarasan.selvarasu
"""

class Configurations(object):
    
    def __init__(self):
        pass
    
    BASE_FILE = "D:/Recommender Systems/purchases/purchases.csv"
    PROCSD_FILE = "data/base_df_8020.csv"
    ITEM_MAPPING = "data/item_mapping.pickle"
    USER_MAPPING = "data/user_mapping.pickle"
    TRAIN_TEST_SPLIT = 0.2
    ITEM_THRESHOLD = 5