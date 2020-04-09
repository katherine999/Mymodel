# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:02:17 2020

@author: Leo Lin
"""
import pandas as pd
import numpy as np
import os
import string
import re


def inverse_Simpson_index(sp1_num, sp2_num, sp3_num, sp4_num):
    N = sp1_num + sp2_num + sp3_num + sp4_num
    inverse_simpson_index = 1/((sp1_num*(sp1_num-1))/(N*(N-1)) + (sp2_num*(sp2_num-1))/(N*(N-1)) + (sp3_num*(sp3_num-1))/(N*(N-1)) + (sp4_num*(sp4_num-1))/(N*(N-1)))
    return inverse_simpson_index

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

def patch_diversit_list(df, time_step):
    patch_diversit_list = []
    for patch_id in ['patch%d' % i for i in range(0,17)]:
        species_num_counts = df.loc['time_step4999'][patch_id].value_counts()
        try:
            sp1_num = species_num_counts[1.0]
        except:
            sp1_num = 0
        try:
            sp2_num = species_num_counts[2.0]
        except:
            sp2_num = 0
        try:
            sp3_num = species_num_counts[3.0]
        except:
            sp3_num = 0
        try:
            sp4_num = species_num_counts[4.0]
        except:
            sp4_num = 0
        patch_diversity = inverse_Simpson_index(sp1_num, sp2_num, sp3_num, sp4_num)
        patch_diversit_list.append(patch_diversity)
    return np.array(patch_diversit_list).reshape(17,1)

def elevation_diversity_list(df, time_step):
    elevation_diversit_list = []
    sp1_ele, sp2_ele, sp3_ele, sp4_ele = 0, 0, 0, 0
    for patch_id in ['patch%d' % i for i in range(1,17)]:
        patch_sp_num_counts = df.loc['time_step4999'][patch_id].value_counts()
        try:
            sp1_num = patch_sp_num_counts[1.0]
        except:
            sp1_num = 0
        try:
            sp2_num = patch_sp_num_counts[2.0]
        except:
            sp2_num = 0
        try:
            sp3_num = patch_sp_num_counts[3.0]
        except:
            sp3_num = 0
        try:
            sp4_num = patch_sp_num_counts[4.0]
        except:
            sp4_num = 0
        sp1_ele += sp1_num 
        sp2_ele += sp2_num
        sp3_ele += sp3_num
        sp4_ele += sp4_num
        if int(patch_id[-1]) % 4 == 0:
            elevation_diversity = inverse_Simpson_index(sp1_ele, sp2_ele, sp3_ele, sp4_ele)
            elevation_diversit_list.append(elevation_diversity)
            sp1_ele, sp2_ele, sp3_ele, sp4_ele = 0, 0, 0, 0
    return np.array(elevation_diversit_list).reshape(4,1)

def reginal_diversity(df, time_step):
    patch_all_counts = df.loc['time_step4999'].value_counts()
    patch0_counts = df.loc['time_step4999']['patch0'].value_counts()
    other_patch_counts = patch_all_counts - patch0_counts
    try:
        sp1_num = other_patch_counts[1.0]
    except:
        sp1_num = 0
    try:
        sp2_num = other_patch_counts[2.0]
    except:
        sp2_num = 0
    try:
        sp3_num = other_patch_counts[3.0]
    except:
        sp3_num = 0
    try:
        sp4_num = other_patch_counts[4.0]
    except:
        sp4_num = 0
    regional_diversity = inverse_Simpson_index(sp1_num, sp2_num, sp3_num, sp4_num)
    return np.array([regional_diversity])

path = 'C:\\Users\\Leo Lin\\Desktop\\model\\Mymodel\\mymodel2.5(formal)'
data_type = 'all time species_distribution replication='
files_list = get_filename(path, data_type)
header = [np.array([]),    # group
          np.array([]),    # szie
          np.array([])]    # replication
index = ['patch%d'%i for i in range(0,17)] + ['evelation%d'%i for i in range(1,5)] + ['regional']
counter = 0
for file in files_list:
    counter += 1
    print(counter, file[0], file[1], file[2])
    header = [np.append(header[0], file[0]),       # group
              np.append(header[1], file[1]),       # szie
              np.append(header[2], file[2])]       # replication
    file_name = file[3]
    df_sp_distr_n = pd.read_csv(file_name, index_col=[0], header=[0,1,2])
    
    patch_diversity_n = patch_diversit_list(df=df_sp_distr_n, time_step='time_step4999')
    elevation_diversity_n = elevation_diversity_list(df=df_sp_distr_n, time_step='time_step4999')
    regional_diversity_n = reginal_diversity(df=df_sp_distr_n, time_step='time_step4999')
    
    diversity_infor_n = np.vstack((patch_diversity_n, elevation_diversity_n, regional_diversity_n))
    
    if counter == 1:
        all_patch_diversity_results = diversity_infor_n
    else:
        all_patch_diversity_results = np.hstack((all_patch_diversity_results, diversity_infor_n))
        
all_patch_diversity_results_df = pd.DataFrame(all_patch_diversity_results, index=index, columns=header)

all_patch_diversity_results_df.to_csv('all_patch_diversity_results.csv')  
























