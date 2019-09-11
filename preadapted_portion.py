# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 16:13:29 2019

@author: Leo Lin
"""
import os
import string
import re
import numpy as np
import pandas as pd
from scipy.stats import mode

def get_filename(path, data_type):
    files_list =[]
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name1 = file.split('.')[0]                                # 去除文件名后缀
            file_name2 = file_name1.strip(string.digits)        # 去除文件名两端额数字
            if file_name2==data_type:
                dispersal_group = root.split('\\')[-2]
                offsprings_pool_size = root.split('\\')[-1]
                replication_time = re.findall(r'\d+', file_name1)[0]
                file_path = root+'\\'+file
                files_list.append((dispersal_group, offsprings_pool_size, replication_time, file_path))
    return files_list

def get_theo_dominant_sp_list(df):
    theo_dom_sp_dis = {}
    for i in range(1,17):
        patch_id = 'patch%d'%i
        theo_dom_sp_dis_patch = {}
        for j in range(4):
            habitat_id = 'h%d'%j
            dominant_sp = dominant_sp = mode(df.loc[patch_id, habitat_id].values)[0][0]
            theo_dom_sp_dis_patch[habitat_id] = dominant_sp
        theo_dom_sp_dis[patch_id] = theo_dom_sp_dis_patch
    return theo_dom_sp_dis
            
def get_dominant_sp_list(df, time_step):
    dyn_dom_sp_dis = {}
    for i in range(1,17):
        patch_id = 'patch%d'%i
        dom_sp_dis_patch = {}
        for j in range(4):
            habitat_id = 'h%d'%j
            dom_sp = mode(df.loc[time_step][patch_id, habitat_id].values)[0][0]
            mode_count = mode(df.loc[time_step][patch_id, habitat_id].values)[1][0]
            if mode_count >= 12:
                dom_sp_dis_patch[habitat_id] = dom_sp
            else:
                dom_sp_dis_patch[habitat_id] = np.nan
        dyn_dom_sp_dis[patch_id] = dom_sp_dis_patch
    return dyn_dom_sp_dis

def similarity(theo_sp, dyn_sp):
    return 1-abs(theo_sp-dyn_sp)/3

def theo_adapted_portion(theo_dic, dyn_dic):
    match_h = 0
    for patch_id in theo_dic:
        sp_patch = theo_dic[patch_id]
        for habitat_id in sp_patch:
            theo_sp = theo_dic[patch_id][habitat_id]
            dyn_sp = dyn_dic[patch_id][habitat_id]
            if np.isnan(dyn_sp)==True:
                match_h += 0
            else:
                match_h+=similarity(theo_sp, dyn_sp)
    return match_h/(16*4)
    
def percent_dynamics(file):
    file_name = file[3]
    percent_results = []
    df_sp_dyn = pd.read_csv(file_name, index_col=[0], header=[0,1,2])
    df_sp_theo = pd.read_csv('therotical preadapted species.csv', index_col=[0,1], header=[0])
    for time in range(5000):
        theo_dom_sp_dis = get_theo_dominant_sp_list(df=df_sp_theo)
        dyn_dom_sp_dis = get_dominant_sp_list(df=df_sp_dyn, time_step='time_step'+str(time))
        percent = theo_adapted_portion(theo_dic=theo_dom_sp_dis, dyn_dic=dyn_dom_sp_dis)
        percent_results.append(percent)
        print(file[0], file[1], file[2], 'timestep%d'%time, percent)
    return percent_results

path = 'C:\\Users\\Leo Lin\\Desktop\\model\\Mymodel\\mymodel2.5(formal)'
data_type = 'all time species_distribution replication='
files_list = get_filename(path, data_type)
header = [np.array([]),
          np.array([]),
          np.array([])]

all_percent_results = []

for file in files_list:
    percent_results = np.array(percent_dynamics(file)).reshape(5000, 1)
    header = [np.append(header[0], file[0]), np.append(header[1], file[1]), np.append(header[2], file[2])]       
    # multiheader, header[0]=dispersal_group, header[1]=size of pool, header[2]=replication
    if all_percent_results == []:
        all_percent_results = percent_results
    else:
        all_percent_results = np.hstack((all_percent_results, percent_results))

all_percent_results_df = pd.DataFrame(all_percent_results, index=['time_step%d'%i for i in range(5000)], columns=header)
all_percent_results_df.to_csv('all_percent_results.csv')  

        
        
        
        
        
        
        
        