#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:59:02 2018

@author: yoon
"""




import pandas as pd

file = "/Users/yoon/Downloads/data2.csv"
df = pd.read_csv(file)

id_group = df.groupby('pcid')['tq'].apply(list)
df_id_group = id_group.add_suffix('_list').reset_index()