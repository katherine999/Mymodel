# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:44:10 2019

@author: Leo Lin
"""

import numpy as np
import random
import math
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#######################################  创建模型地貌  #######################################################
def generate_habitat(e, t_sigmoid, length, width):
    microsite_e_values = np.random.normal(loc=0, scale=0.025, size=(length, width)) + e
    microsite_t_values = np.random.normal(loc=0, scale=0.025, size=(length, width)) + t_sigmoid
    microsite_individuals = [[None for i in range(length)] for i in range(width)]
    return {'microsite_e_values':microsite_e_values, 'microsite_t_values':microsite_t_values, 'microsite_individuals':microsite_individuals}
    #生成一个hibitat包括environment_value,temperature_value和microsite_individual
def sigmoid(T):
    return 1/(1+np.exp(-T))
    # 设置s型函数，作为温度归一化处理函数

def generate_patch(length, width, t0, t1, t2, t3, e0, e1, e2, e3):
    h0 = generate_habitat(e = e0, t_sigmoid=sigmoid((t0-26)/2), length=length, width=width)
    h1 = generate_habitat(e = e1, t_sigmoid=sigmoid((t1-26)/2), length=length, width=width)
    h2 = generate_habitat(e = e2, t_sigmoid=sigmoid((t2-26)/2), length=length, width=width)
    h3 = generate_habitat(e = e3, t_sigmoid=sigmoid((t3-26)/2), length=length, width=width)
    patch = {'h0':h0, 'h1':h1, 'h2':h2, 'h3':h3}
    return patch
    # 生成一个patch
def generate_matacommunity(patch_number, initially_occupy_number, patch_number_each_altitude, length, width,
                           e0, e1, e2, e3, T_upstream, T_upmidstream, T_downmidstream, T_downstream, E_source, T_source):  # E_source=[e0, e1, e2, e3], T_source=[T0, T1, T2, T3]
    metacommunity = {}                                                                                                         
    for i in range(patch_number):
        if i < initially_occupy_number:                                                                                        # for initial occupied patches
            patch = generate_patch(length=length, width=width, 
                                   e0=E_source[0], e1=E_source[1], e2=E_source[2], e3=E_source[3],
                                   t0=T_source[0], t1=T_source[1], t2=T_source[2], t3=T_source[3])
        if (i-initially_occupy_number)//patch_number_each_altitude == 0:                                                       # for patch at 1st altitude
            patch = generate_patch(length=length, width=width, e0=e0, e1=e1, e2=e2, e3=e3,
                                   t0=T_upstream, t1=T_upstream, t2=T_upstream, t3=T_upstream)
        if (i-initially_occupy_number)//patch_number_each_altitude == 1:                                                       # for patch at 2nd altitude
            patch = generate_patch(length=length, width=width, e0=e0, e1=e1, e2=e2, e3=e3,
                                   t0=T_upmidstream, t1=T_upmidstream, t2=T_upmidstream, t3=T_upmidstream)
        if (i-initially_occupy_number)//patch_number_each_altitude == 2:                                                       # for patch at 3rd altitude
            patch = generate_patch(length=length, width=width, e0=e0, e1=e1, e2=e2, e3=e3,
                                   t0=T_downmidstream, t1=T_downmidstream, t2=T_downmidstream, t3=T_downmidstream)
        if (i-initially_occupy_number)//patch_number_each_altitude == 3:                                                       # for patch at 4th altitude
            patch = generate_patch(length=length, width=width, e0=e0, e1=e1, e2=e2, e3=e3,
                                   t0=T_downstream, t1=T_downstream, t2=T_downstream, t3=T_downstream)
            
        metacommunity['patch%s'%(str(i))] = patch
    return metacommunity

#######################################  初始化并加入物种个体  ############################################################
def init_individual(frequency_for_neutral_gene, habitat_id, e, T, sexual, L_e, L_t, L_n):               #sexual为male, female；只有有性繁殖时该参数才会被调用
    phenotype = e + random.gauss(0,0.025)
    termotype = sigmoid((T-26)/2) + random.gauss(0,0.025)
    
    pheno_random = random.sample(range(0,L_e),int(e*L_e))
    termo_random = random.sample(range(0,L_t),int(sigmoid((T-26)/2)*L_t+0.5))

    phenotye_gene = [1 if i in pheno_random else 0 for i in range(L_e)] 
    termotype_gene = [1 if i in termo_random else 0 for i in range(L_t)]
    neutral_gene = [0 if frequency_for_neutral_gene[i]<np.random.uniform(0,1,1)[0] else 1 for i in range(L_n)]
        
    genotype = phenotye_gene + termotype_gene + neutral_gene

    return {'identifier': (int(habitat_id[1:])+1), 'sexual': sexual, 'phenotype':phenotype, 'termotype':termotype,'genotype':genotype} 
    # 生成一个个体，个体包括'identifier'(物种的标识)；'sexual'(性别)；'phenotype'(表型，与环境值对应)；'termotype'(温度型，与环境温度对应)；'genotype'(基因型，计算出表型，温度型其中包括中性基因)

def initialization(metacommunity, initially_occupy_number, length, width, E_source, T_source, L_e, L_t, L_n): #E_source=[e0,e1,e2,e3], T_source=[T0,T1,T2,T3]
    frequency_for_neutral_gene = np.array([0.5]*L_n)                      #frequency_of_gene value = 0.5
    for patch_id in metacommunity:
        if int(re.findall(r"\d+",patch_id)[0]) < initially_occupy_number:
            patch = metacommunity[patch_id]
            for habitat_id in patch:
                e = E_source[int(habitat_id[1:])]
                t = T_source[int(habitat_id[1:])]
                for row in range(length):
                    for col in range(width):
                        if np.random.uniform(0,1,1)[0] > 0.5: new_individual = init_individual(frequency_for_neutral_gene=frequency_for_neutral_gene, habitat_id=habitat_id, e=e, T=t, L_e=L_e, L_t=L_t, L_n=L_n, sexual='male')
                        else: new_individual = init_individual(frequency_for_neutral_gene=frequency_for_neutral_gene, habitat_id=habitat_id, e=e, T=t, L_e=L_e, L_t=L_t, L_n=L_n, sexual='female')
                        metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = new_individual

    return metacommunity
    # 对集合群落进行初始化操作，即对initially_occupied_patch 加入pre-adapted的个体，这些个体适应与local habitat
#########################################################   死亡选择  ###################################################################################
def survival_rate(d, z, em, ti, tm, w = 0.5):
    survival_rate = (1-d) * math.exp((-1)*math.pow(((z-em)/w),2)) * math.exp((-1)*math.pow(((ti-tm)/w),2))
       # 存活的几率符合正态分布,d为基本死亡率，z为表型，em为环境因子，ti为温度型，tm环境温度归一化值
    return survival_rate
# d为基本的死亡量，z为某个个体的表型，e为该microsite的环境值，w为width of the fitness function


def dead_selection(metacommunity, length, width):           # 死亡选择，对集合群落中的所有个体
    for patch_id in metacommunity:                                         # metacommunity为一个字典，此语法得到的patch_id，为字典的key值
        patch = metacommunity[patch_id]
        for habitat_id in patch:                                           # patch为一个字典，此语法得到habitat_id, 为字典的一个key值
            habitat = patch[habitat_id]
            microsite_individuals_set = habitat['microsite_individuals']   # 一个5*5的list，包含25个e值
            microsite_e_values_set = habitat['microsite_e_values']         # 一个5*5的list，包含25个individual值
            microsite_t_values_set = habitat['microsite_t_values']
            for row in range(length):
                for col in range(width):
                    microsite_e_value = microsite_e_values_set[row][col]   # 一个microsite的e_value
                    microsite_t_value = microsite_t_values_set[row][col]
                    if microsite_individuals_set[row][col] != None:
                        microsite_individual = microsite_individuals_set[row][col] #一个microsite的individual值
                        phenotype = microsite_individual['phenotype']              # 该microsite个体的一个phenotype值
                        termotype = microsite_individual['termotype']
                        survival_p = survival_rate(d=0.1, z=phenotype, em=microsite_e_value, ti=termotype, tm=microsite_t_value, w = 0.5)

                        # 通过表型和环境的e_value计算该个体的存活率
                        if survival_p < np.random.uniform(0,1,1):
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = None

#####################################################  繁殖  #######################################################################
def species_division(metacommunity, length, width):
    species_category = {}
    for patch_id in metacommunity:
        patch = metacommunity[patch_id]
        species_category_patch ={}
        for habitat_id in patch:
            habitat = patch[habitat_id]
            species_category_habitat = {}
            microsite_individuals_set = habitat['microsite_individuals']
            species1, species2, species3, species4 = {'male':[], 'female':[]}, {'male':[], 'female':[]}, {'male':[], 'female':[]}, {'male':[], 'female':[]}
            for row in range(length):
                for col in range(width):
                    if microsite_individuals_set[row][col] != None:
                        microsite_individual = microsite_individuals_set[row][col]
                        identifier = microsite_individual['identifier']
                        sexual = microsite_individual['sexual']
                        
                        if identifier == 1: species1[sexual].append(microsite_individual)
                        if identifier == 2: species2[sexual].append(microsite_individual)
                        if identifier == 3: species3[sexual].append(microsite_individual)
                        if identifier == 4: species4[sexual].append(microsite_individual)
            species_category_habitat['species1'] = species1
            species_category_habitat['species2'] = species2
            species_category_habitat['species3'] = species3
            species_category_habitat['species4'] = species4
            species_category_patch[habitat_id] = species_category_habitat
        species_category[patch_id] = species_category_patch
    return species_category        
# species_category[patch_id][habitat_id]['speciesX'][sexual]
# 对集合群落中的所有个体根据species indentifier和sexual（性别）进行分类，用于后续进行有性生活的亲本的匹配、
# 在有性生殖的亲本中只有同一物种的不同性别才能进行交配
    
def species_parents_pair(species, length, width):
    matching_results_species = [] 
    for matching in range((length*width)//2): 
        if len(species['male']) >= 1 and len(species['female']) >= 1:
            male = random.sample(species['male'], 1)[0]
            species['male'].remove(male)
            female = random.sample(species['female'], 1)[0]
            species['female'].remove(female)
            matching_results_species.append((male, female))
        else:
            break
    return matching_results_species
   
# 对同一物种个体进行父本和母本随机两两配对    

def matching_parents(length, width, patch_id, habitat_id, species_category):
    species_category_habitat = species_category[patch_id][habitat_id]
    species1 = species_category_habitat['species1']
    species2 = species_category_habitat['species2']
    species3 = species_category_habitat['species3']
    species4 = species_category_habitat['species4']
    
    matching_results_species1 = species_parents_pair(species1, length, width)
    matching_results_species2 = species_parents_pair(species2, length, width)
    matching_results_species3 = species_parents_pair(species3, length, width)
    matching_results_species4 = species_parents_pair(species4, length, width)
    
    matching_results = matching_results_species1 + matching_results_species2 + matching_results_species3 + matching_results_species4
    return matching_results
# 对4个物种种进行两两配对的操作，返回父本母本的配对情况


def reproduce(parents, reproduction_type, L_e, L_t):                          #当无性繁殖时parents为一个亲本，当有性繁殖是parents为两个亲本
    if reproduction_type == 'asexual':                              # 无性繁殖
         
         n_identifier = parents['identifier']
         n_sexual = parents['sexual']
         n_genotype = parents['genotype']
         n_phenotype= np.mean(n_genotype[0:L_e]) + random.gauss(0,0.025)
         n_termotype = np.mean(n_genotype[L_e:L_e+L_t]) + random.gauss(0,0.025)

         new = {'identifier':n_identifier, 'sexual':n_sexual,'phenotype':n_phenotype, 'termotype':n_termotype ,'genotype':n_genotype}
    # 无性生殖时identifier，genotype与亲本相同，sexual与亲本相同（此参数在无性繁殖时不会被调用），phenotype和termotype根据genotype计算的出
    
    if reproduction_type == 'sexual':                               # 有性繁殖
        male = parents[0]                                           # 有性繁殖父本
        female = parents[1]                                         # 有性繁殖母本
        locus_num = len(female['genotype']) 
        female_gamete = random.sample([i for i in range(locus_num)],int(locus_num/2))   # 雌性配子所含的基因id
        n_identifier = female['identifier']      # 新个体的identifier与亲本相同
        n_genotype = []                       
            
        if 0.5 > np.random.uniform(0,1,1): n_sexual = 'male'        # 后代的性别，female与male均为50%
        else: n_sexual = 'female'
 
        for allelic in range(locus_num):                            # 15个基因来自母本，15个基因来自父本，随机抽样选取
            if allelic in female_gamete: n_genotype.append(female['genotype'][allelic])
            else: n_genotype.append(male['genotype'][allelic])
                
        n_phenotype = np.mean(n_genotype[0:L_e]) + random.gauss(0,0.025)
        n_termotype = np.mean(n_genotype[L_e:L_e+L_t]) + random.gauss(0,0.025)
        
        new = {'identifier':n_identifier, 'sexual':n_sexual,'phenotype':n_phenotype, 'termotype':n_termotype ,'genotype':n_genotype}
    return new

def mutate(individual, L_e, L_t, mutation_rate):                             # 突变函数，对个体进行突变处理
    genotype = individual['genotype']
    new_genotype = []
    for allelic in range(len(genotype)):
        random_uniform_variable = np.random.uniform(0,1,1)[0]
        if mutation_rate > random_uniform_variable:                # 以一定突变率进行突变
            if genotype[allelic] ==0: new_genotype.append(1)       # 0 突变成 1
            if genotype[allelic] ==1: new_genotype.append(0)       # 1 突变成 0
        else:
            new_genotype.append(genotype[allelic])                 # (1-μ)的概率不发生突变
    if new_genotype == genotype:
        new_individual = individual
        return new_individual

    else:                                                          # 若该个体发生了突变则表型和温度型重新计算得出
        new_phenotype = np.mean(new_genotype[0:L_e]) + random.gauss(0,0.025)
        new_termotype = np.mean(new_genotype[L_e:L_e+L_t]) + random.gauss(0,0.025)
        return {'identifier':individual['identifier'], 'sexual':individual['sexual'], 'genotype':new_genotype, 'termotype':new_termotype, 'phenotype':new_phenotype} 

def generate_offsprings_pool():
    return []
def eliminate_offsprings_pool():                                   # 如果offsprings_pool_memory=False，则每个time_step要清空offsprings_pool
    return []


def reproduction(metacommunity, length, width, reproduction_type, L_e, L_t, birth_rate, mutation_rate, time_step):
    offsprings_pool = []
    if reproduction_type == 'asexual':                                         # 对集合群落种的所有个体以一定的出生率进行无性繁殖
        for patch_id in metacommunity: 
            patch = metacommunity[patch_id]
            for habitat_id in patch:
                habitat = patch[habitat_id]
                microsite_individuals_set = habitat['microsite_individuals']   # 一个5*5的list，包含25个individual值
                for row in range(length):
                    for col in range(width):
                        random_uniform_variable = np.random.uniform(0,1,1)[0]
                        if birth_rate > random_uniform_variable and microsite_individuals_set[row][col] != None:
                        # 该microsite不为empty状态，并且以某一birth rate繁殖后代
                            parent = microsite_individuals_set[row][col]
                            new_individual = reproduce(reproduction_type = 'asexual', parents=parent, L_e=L_e, L_t=L_t)
                            new_mutated_individual = mutate(individual=new_individual, L_e=L_e, L_t=L_t, mutation_rate=mutation_rate)
                            offsprings_pool.append((patch_id, habitat_id, new_mutated_individual, time_step))
                                
    if reproduction_type == 'sexual':                                          # 集合群落中以habitat为范围进行两两配对
        species_category = species_division(metacommunity, length, width)
        for patch_id in metacommunity:
            patch = metacommunity[patch_id]
            for habitat_id in patch:
                habitat = patch[habitat_id]
                parents_list = matching_parents(length, width, patch_id, habitat_id, species_category)
                for parents in parents_list:
                    random_uniform_variable = np.random.uniform(0,1,1)[0]
                    if birth_rate > random_uniform_variable:
                        new_individual = reproduce(reproduction_type = 'sexual', parents=parents, L_e=L_e, L_t=L_t)
                        new_mutated_individual = mutate(individual=new_individual, L_e=L_e, L_t=L_t, mutation_rate=mutation_rate)
                        offsprings_pool.append((patch_id, habitat_id, new_mutated_individual,time_step))
                        
    return offsprings_pool

def merge_pool(past_offsprings_pool, current_offsprings_pool, offsprings_pool_memory, offsprings_pool_maxsize):
    
    if offsprings_pool_memory == False or past_offsprings_pool==[]:
        offsprings_pool = current_offsprings_pool

    elif offsprings_pool_memory == True:
        if len(current_offsprings_pool) >= offsprings_pool_maxsize: 
            offsprings_pool = current_offsprings_pool
        else:
            if len(past_offsprings_pool) <= (offsprings_pool_maxsize-len(current_offsprings_pool)):
                offsprings_pool = past_offsprings_pool + current_offsprings_pool
            else:
                past_offsprings_pool=random.sample(past_offsprings_pool,(offsprings_pool_maxsize-len(current_offsprings_pool)))
                offsprings_pool = past_offsprings_pool + current_offsprings_pool

    return offsprings_pool


###########################################  个体迁移   ################################################################################################
def immigrant_category(offsprings_pool, patch_id, habitat_id, initially_occupy_number, patch_number_each_altitude):
    # 以待迁入的empty_microsite的 patch_id和habitat_id为依据，将offsprings_pool中的个体进行分类
    within_pool, among_pool, across_pool, original_pool, local_pool= [],[],[],[],[]
    for offsprings_information in offsprings_pool:
        offspring_patch_id = offsprings_information[0]
        offspring_habitat_id = offsprings_information[1]

        if (int(re.findall(r"\d+",patch_id)[0])-initially_occupy_number) >= 0:
            if offspring_patch_id in ['patch%d'%i for i in range(initially_occupy_number)]: 
                original_pool.append(offsprings_information)
            if offspring_patch_id == patch_id and offspring_habitat_id == habitat_id:
                local_pool.append(offsprings_information)
            if offspring_patch_id == patch_id and offspring_habitat_id != habitat_id: 
                within_pool.append(offsprings_information)
            if offspring_patch_id != patch_id and ((int(re.findall(r"\d+",patch_id)[0])-initially_occupy_number)//patch_number_each_altitude - (int(re.findall(r"\d+",offspring_patch_id)[0])-initially_occupy_number)//patch_number_each_altitude)==0:
                among_pool.append(offsprings_information)
            if offspring_patch_id != patch_id and ((int(re.findall(r"\d+",patch_id)[0])-initially_occupy_number)//patch_number_each_altitude - (int(re.findall(r"\d+",offspring_patch_id)[0])-initially_occupy_number)//patch_number_each_altitude)==1 and (int(re.findall(r"\d+",offspring_patch_id)[0])-initially_occupy_number)>0:
                across_pool.append(offsprings_information)

        if (int(re.findall(r"\d+",patch_id)[0])-initially_occupy_number) < 0:
            if offspring_patch_id == patch_id and offspring_habitat_id == habitat_id:
                local_pool.append(offsprings_information)
            if offspring_patch_id == patch_id and offspring_habitat_id != habitat_id:
                within_pool.append(offsprings_information)
            if offspring_patch_id != patch_id and (int(re.findall(r"\d+",offspring_patch_id)[0])-initially_occupy_number) < 0:
                among_pool.append(offsprings_information)

    if original_pool==[]: original_pool=[None]             #对于source patch内的dispersal成立
    if across_pool == []: across_pool=[None]
    if among_pool == []: among_pool =[None]
    if within_pool ==[]: within_pool=[None]
    if local_pool == []: local_pool=[None]

    return within_pool, among_pool, across_pool, original_pool, local_pool

def dispersal(metacommunity, length, width, offsprings_pool, source_rate, across_rate, among_rate, within_rate, initially_occupy_number, patch_number_each_altitude, adapted_dispersal_storage):
    a = source_rate
    b = source_rate + across_rate
    c = source_rate + across_rate + among_rate
    d = source_rate + across_rate + among_rate + within_rate
    for patch_id in metacommunity:
        patch = metacommunity[patch_id]
        for habitat_id in patch:
            within_pool, among_pool, across_pool, original_pool, local_pool= immigrant_category(offsprings_pool=offsprings_pool, patch_id=patch_id, habitat_id=habitat_id, initially_occupy_number=initially_occupy_number, patch_number_each_altitude=patch_number_each_altitude)
            habitat = patch[habitat_id]
            microsite_individuals_set = habitat['microsite_individuals']  # 一个5*5的list，包含25个individual值
            for row in range(length):
                for col in range(width):
                    if microsite_individuals_set[row][col] == None: # 找出当前处于empty状态的patch
                        random_uniform_variable = np.random.uniform(0,1,1)[0]
                        if 0 < random_uniform_variable < a and original_pool != [None]:
                            dispersal_info = random.sample(original_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = dispersal_info[2]
                            adapted_dispersal_storage = add_dispersal(metacommunity, patch_id, habitat_id, row, col, adapted_dispersal_storage, dispersal_info)
                            
                        elif a < random_uniform_variable < b and across_pool != [None]:
                            dispersal_info = random.sample(across_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = dispersal_info[2]
                            adapted_dispersal_storage = add_dispersal(metacommunity, patch_id, habitat_id, row, col, adapted_dispersal_storage, dispersal_info)
                            
                        elif b < random_uniform_variable < c and among_pool !=[None]:
                            dispersal_info = random.sample(among_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = dispersal_info[2]
                            adapted_dispersal_storage = add_dispersal(metacommunity, patch_id, habitat_id, row, col, adapted_dispersal_storage, dispersal_info)
                            
                        elif c < random_uniform_variable < d and within_pool !=[None]:
                            dispersal_info = random.sample(within_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = dispersal_info[2]
                            adapted_dispersal_storage = add_dispersal(metacommunity, patch_id, habitat_id, row, col, adapted_dispersal_storage, dispersal_info)
    return adapted_dispersal_storage
                        
def dispersal_only_within_patch(metacommunity, length, width, initially_occupy_number, patch_number_each_altitude, offsprings_pool, within_rate, adapted_dispersal_storage):
    dispersal_within_rate = within_rate
    for patch_id in metacommunity:
        patch = metacommunity[patch_id]
        for habitat_id in patch:
            within_pool = immigrant_category(offsprings_pool, patch_id, habitat_id, initially_occupy_number, patch_number_each_altitude)[0]
            habitat = patch[habitat_id]
            microsite_individuals_set = habitat['microsite_individuals']
            for row in range(length):
                for col in range(width):
                    if microsite_individuals_set[row][col] == None:
                        random_uniform_variable = np.random.uniform(0,1,1)[0]
                        if 0 < random_uniform_variable < dispersal_within_rate and within_pool != [None]:
                            dispersal_info = random.sample(within_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = dispersal_info[2]
                            adapted_dispersal_storage = add_dispersal(metacommunity, patch_id, habitat_id, row, col, adapted_dispersal_storage, dispersal_info)
    return adapted_dispersal_storage

def local_compensation(metacommunity, length, width, offsprings_pool, initially_occupy_number, patch_number_each_altitude):
    for patch_id in metacommunity:
        patch = metacommunity[patch_id]
        for habitat_id in patch:
            habitat = patch[habitat_id]
            microsite_individuals = habitat['microsite_individuals']
            local_pool = immigrant_category(offsprings_pool=offsprings_pool, patch_id=patch_id, habitat_id=habitat_id, 
                                            initially_occupy_number=initially_occupy_number, patch_number_each_altitude=patch_number_each_altitude)[4]
            for row in range(length):
                for col in range(width):
                    if microsite_individuals[row][col] == None and local_pool != [None] and local_pool != []:
                        compensation_individual_info = random.sample(local_pool, 1)[0]
                        local_pool.remove(compensation_individual_info)
                        metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = compensation_individual_info[2]         
                        
#######################################    lineage   ########################################################
def generate_adapted_dispersal_storage(patch_number, length, width):                                         # 用于保存dispersal个体并且其存活率大于0.7
    lineage_manager = {}
    for i in range(patch_number):
        h0 = [[[] for j in range(length)] for k in range(width)]
        h1 = [[[] for j in range(length)] for k in range(width)]
        h2 = [[[] for j in range(length)] for k in range(width)]
        h3 = [[[] for j in range(length)] for k in range(width)]
        patch_lineage={'h0':h0, 'h1':h1, 'h2':h2, 'h3':h3}
        lineage_manager['patch%d'%i] = patch_lineage
    return lineage_manager


def add_dispersal(metacommunity, patch_id, habitat_id, row, col, adapted_dispersal_storage, dispersal_info):
    em = metacommunity[patch_id][habitat_id]['microsite_e_values'][row][col]
    tm = metacommunity[patch_id][habitat_id]['microsite_t_values'][row][col]
    ei = dispersal_info[2]['phenotype']
    ti = dispersal_info[2]['termotype']
    survival_p = survival_rate(d=0.1, z=ei, em=em, ti=ti, tm=tm, w = 0.5)
    if survival_p >= 0.7:
        adapted_dispersal_info = list(dispersal_info)
        adapted_dispersal_info[2] = dispersal_info[2]['identifier']
        adapted_dispersal_info.append(survival_p)
        adapted_dispersal_info = tuple(adapted_dispersal_info)
        adapted_dispersal_storage[patch_id][habitat_id][row][col].append(adapted_dispersal_info)
    return adapted_dispersal_storage


#######################################     可视化    ########################################################
def location(patch_number, initially_occupy_number, patch_number_each_altitude):
    initial_occupied_location = []
    upstream_location = []
    upmidestream_location = []
    downmidstream_location = []
    downstream_location = []
    if patch_number_each_altitude >= initially_occupy_number:
        fig_size_length = patch_number_each_altitude
        fig_size_width = int((patch_number-initially_occupy_number)/patch_number_each_altitude+1)
    if patch_number_each_altitude < initially_occupy_number:
        fig_size_length = initially_occupy_number
        fig_size_width = int((patch_number-initially_occupy_number)/patch_number_each_altitude+1)
    for i in range(initially_occupy_number):
        initial_occupied_location.append(i+1)
    for i in range(fig_size_length, fig_size_length+patch_number_each_altitude):
        upstream_location.append(i+1)
    for i in range(fig_size_length*2, fig_size_length*2+patch_number_each_altitude):
        upmidestream_location.append(i+1)
    for i in range(fig_size_length*3, fig_size_length*3+patch_number_each_altitude):
        downmidstream_location.append(i+1)
    for i in range(fig_size_length*4, fig_size_length*4+patch_number_each_altitude):
        downstream_location.append(i+1)
    return initial_occupied_location+upstream_location+upmidestream_location+downmidstream_location+downstream_location, (fig_size_length, fig_size_width)
    
def get_values_set(metacommunity, patch_id, length, width, values_type):
    patch = metacommunity[patch_id]
    values_set = []
    for habitat_id in patch:
        habitat = patch[habitat_id]
        for row in range(length):
            for col in range(width):
                if habitat['microsite_individuals'][row][col]!=None:
                    values_set.append(habitat['microsite_individuals'][row][col][values_type])
                else:
                    values_set.append(np.nan) # 表示该晶格为空的状态
                    
    values_set = np.array(values_set).reshape(4,length*width)
    return values_set

def phenotype_to_fig(metacommunity, patch_number, initially_occupy_number, patch_number_each_altitude, fig, length, width, file_name, values_type='phenotype'):

    plt.title('individuals_phenotype')
    locate = location(patch_number, initially_occupy_number, patch_number_each_altitude)[0]
    fig_size = location(patch_number, initially_occupy_number, patch_number_each_altitude)[1] # fig_size=(fig_size_length, fig_size_width)
    
    for patch_id in metacommunity:
        l = locate[int(re.findall(r"\d+",patch_id)[0])]
        fig.add_subplot(fig_size[1],fig_size[0],l)
        plt.tight_layout()
        plt.title(patch_id)
        df = pd.DataFrame(get_values_set(metacommunity, patch_id, length, width, values_type))
        sns.heatmap(data=df, vmax=1.0, vmin=0)
    
    plt.savefig(file_name)
    plt.clf()

def termotype_to_fig(metacommunity, patch_number, initially_occupy_number, patch_number_each_altitude, fig, length, width, file_name, values_type='termotype'):

    plt.title('individuals_termotype')
    locate = location(patch_number, initially_occupy_number, patch_number_each_altitude)[0]
    fig_size = location(patch_number, initially_occupy_number, patch_number_each_altitude)[1] # fig_size=(fig_size_length, fig_size_width)

    for patch_id in metacommunity:
        l = locate[int(re.findall(r"\d+",patch_id)[0])]
        fig.add_subplot(fig_size[1],fig_size[0],l)
        plt.tight_layout()
        plt.title(patch_id)
        df = pd.DataFrame(get_values_set(metacommunity, patch_id, length, width, values_type))
        sns.heatmap(data=df, vmax=1.0, vmin=0)
    
    plt.savefig(file_name)
    plt.clf()

def species_to_fig(metacommunity, patch_number, initially_occupy_number, patch_number_each_altitude, fig, length, width, file_name, values_type='identifier'):

    plt.title('species_distribution')
    locate = location(patch_number, initially_occupy_number, patch_number_each_altitude)[0]
    fig_size = location(patch_number, initially_occupy_number, patch_number_each_altitude)[1] # fig_size=(fig_size_length, fig_size_width)

    for patch_id in metacommunity:
        l = locate[int(re.findall(r"\d+",patch_id)[0])]
        fig.add_subplot(fig_size[1],fig_size[0],l)
        plt.tight_layout()
        plt.title(patch_id)
        df = pd.DataFrame(get_values_set(metacommunity, patch_id, length, width, values_type))
        sns.heatmap(data=df, vmax=5.0, vmin=1.0)
    
    plt.savefig(file_name)
    plt.clf()
#######################################    数据保存    ######################################################
def coordinate(length, width):
    XY = []
    for x in range(length):
        for y in range(width):
            X_Y_ = 'X%sY%s'%(str(x), str(y))
            XY.append(X_Y_)
    return XY

def generate_index_of_patch(metacommunity):
    patch_index = []
    for patch_id in metacommunity:
        patch_index += [patch_id]*4
    return patch_index

def to_csv(metacommunity, patch_number, length, width, file_name, values_type):
    XY = coordinate(length, width)
    patch_index = generate_index_of_patch(metacommunity)
    habitat_index = ['h0', 'h1', 'h2', 'h3']*patch_number
    for patch_id in metacommunity:
        if patch_id == 'patch0':
            species_distribution_data = get_values_set(metacommunity, patch_id, length, width, values_type)
        else:
            values_set = get_values_set(metacommunity, patch_id, length, width, values_type)
            species_distribution_data = np.append(species_distribution_data, values_set, axis=0)
            
    index=pd.MultiIndex.from_arrays([patch_index,habitat_index], names=['patch_id','habitat_id'])
    columns = XY
    df = pd.DataFrame(species_distribution_data,index=index, columns=columns)
    df.to_csv(file_name)
    
def termotype_to_csv(metacommunity, patch_number, length, width, file_name, values_type='termotype'):
    return to_csv(metacommunity, patch_number, length, width, file_name, values_type)

def phenotype_to_csv(metacommunity, patch_number, length, width, file_name, values_type='phenotype'):
    return to_csv(metacommunity, patch_number, length, width, file_name, values_type)

def species_to_csv(metacommunity, patch_number, length, width, file_name, values_type='identifier'):
    return to_csv(metacommunity, patch_number, length, width, file_name, values_type)    
#####################   dispersal data  #########################################    
def adapted_dispersal_to_csv(adapted_dispersal_storage, length, width, file_name):
    adapted_disprsal_list = []
    for patch_id in adapted_dispersal_storage:
        adapted_dispersal_patch = adapted_dispersal_storage[patch_id]
        for habitat_id in adapted_dispersal_patch:
            adapted_dispersal_habitat = adapted_dispersal_patch[habitat_id]
            for row in range(length):
                for col in range(width):
                    if adapted_dispersal_habitat[row][col] != None:
                        for adapted_dispersal_info in adapted_dispersal_patch[habitat_id][row][col]:
                            birth_patch   = adapted_dispersal_info[0]
                            birth_habitat = adapted_dispersal_info[1]
                            identifier    = adapted_dispersal_info[2]
                            time_step     = adapted_dispersal_info[3]
                            survival_rate = adapted_dispersal_info[4]
                            adapted_disprsal_list.append([patch_id, habitat_id, 'X%d'%row, 'Y%d'%col, birth_patch,birth_habitat,identifier,time_step,survival_rate])
    df_adapted_dispersal = pd.DataFrame(adapted_disprsal_list, columns=('patch_id', 'habitat_id', 'row', 'col', 'birth_patch', 'birth_habitat', 'identifier', 'time_step', 'survival_rate'))
    df_adapted_dispersal.set_index('patch_id', 'habitat_id')
    df_adapted_dispersal.to_csv(file_name)
#######################################  gene frequency  ###################################################3
def read_gene_frequency(metacommunity, length, width, L_e, L_t, L_n):
    sum_genotype = np.array([0]*(L_e+L_t+L_n))
    individual_counter = 0
    for patch_id in metacommunity: 
        patch = metacommunity[patch_id]
        for habitat_id in patch: 
            habitat = patch[habitat_id]
            habitat['microsite_individuals']
            microsite_individuals_set = habitat['microsite_individuals']
            for row in range(length):
                for col in range(width):
                    if microsite_individuals_set[row][col] != None:
                        microsite_individual = microsite_individuals_set[row][col]
                        genotype = microsite_individual['genotype']
                        individual_counter += 1
                        sum_genotype += np.array(genotype)
    return (sum_genotype/individual_counter)

def gene_frequency_to_cvs(all_time_step_gene_frequency, gene_id, file_name):

    columns = ['gene%d'%i for i in range(gene_id[0],gene_id[1])]
    index = ['initial_gene_frequency']+['time_step%d'%i for i in range(5000)]
    
    df_gene_frequency = pd.DataFrame(all_time_step_gene_frequency, index=index, columns=columns)
    ''' 根据gene_id 切片，待优化'''
    df_gene_frequency.to_csv(file_name)
    return df_gene_frequency
################################  species distribution data  ##########################################    
    
def read_species_distribution(metacommunity, length, width):
	species_distribution = np.array([], dtype=int)
	for patch_id in metacommunity:
		species_distribution_patch = get_values_set(metacommunity=metacommunity, patch_id=patch_id, length=length, width=width, values_type='identifier')
		species_distribution = np.append(species_distribution, species_distribution_patch.reshape(-1))
	return species_distribution


def species_distribution_to_csv(species_distribution_all_time_step, length, width, hab_num_patch, patch_number, file_name):
    indi_num_patch = length*width*hab_num_patch
    columns_patch_id = np.array([['patch%d'%i]*indi_num_patch for i in range(patch_number)]).reshape(-1)
    columns_habitat_id = np.array([[['h%d'%i]*25 for i in range(4)] for j in range(17)]).reshape(-1)
    columns_mocrosite_id = np.array([['X%dY%d'%(i,j) for i in range(5) for j in range(5)] for k in range(4*17)]).reshape(-1)

    columns = [columns_patch_id, columns_habitat_id, columns_mocrosite_id]
    index = ['initial_gene_frequency']+['time_step%d'%i for i in range(5000)]
	
    df_species_distribution = pd.DataFrame(species_distribution_all_time_step, index=index, columns=columns)
    df_species_distribution.to_csv(file_name)
    return df_species_distribution
    
############################################   def mian()  #############################################################################################

#######################################  参数  ############################################################
def main(replication_time, offsprings_pool_maxsize):
    patch_number = 17                                                          #patch总数量，包括initial_occupied 和 initial_empty
    initially_occupy_number = 1                                                #initial_occupied_patch 的数量
    patch_number_each_altitude = 4                                             #每个海拔高度的patch的数量
    length = 5                                                                 #habitat的长
    width = 5                                                                  #habitat的宽
    e0 = 0.2                                                                   #patch中 habitat='h0'时的环境值
    e1 = 0.4                                                                   #patch中 habitat='h1'时的环境值
    e2 = 0.6                                                                   #patch中 habitat='h2'时的环境值
    e3 = 0.8                                                                   #patch中 habitat='h3'时的环境值
    T_upstream = 23                                                            #上游河流的温度
    T_upmidstream = 25                                                         #中上游河流温度
    T_downmidstream = 27                                                       #中下游河流温度
    T_downstream = 29                                                          #下游河流的温度
    E_source = [0.2, 0.4, 0.6, 0.8]                                            #物种源地每个habitat分别的环境值
    T_source=[23,25,27,29]                                                     #物种源地每个habitat分别的温度
    L_e=20                                                                     #决定物种表型（与环境值匹配）的基因数量
    L_t=20                                                                     #决定物种温度型（与温度值匹配）的基因数量
    L_n=20                                                                     #物种中性基因的数量
    reproduction_type='asexual'                                                 #物种的繁殖方式
    birth_rate=0.5                                                             #出生率
    mutation_rate = 0.0001                                                     #突变率
    offsprings_pool_memory=True                                               #布尔值，储存后代的库是否具有记忆性，及是否保存上一个time_step所产生的后代的数据
    offsprings_pool_maxsize=offsprings_pool_maxsize                             #当offsprings_pool_memory = True时起作用，决定储存后代的库的大小
    source_rate=0.01                                                            #物种个体从源地迁出，迁入到empty_microsite的迁移率
    across_rate=0.01                                                           #物种个体跨越海拔高度迁移时的迁移率
    among_rate=0.05                                                            #物种个体在同一海拔高度，不同patch之间迁移时的迁移率
    within_rate=0.1                                                            #物种个体在同一个patch不同habitat之间迁移时的迁移率
    ##################################################################################################################
    metacommunity = generate_matacommunity(patch_number=patch_number, initially_occupy_number=initially_occupy_number, 
                                       patch_number_each_altitude = patch_number_each_altitude,
                                       length = length, width = width, e0 = e0, e1 = e1, e2 = e2, e3 = e3,
                                       T_upstream=T_upstream, T_upmidstream=T_upmidstream, 
                                       T_downmidstream=T_downmidstream, T_downstream=T_downstream, 
                                       E_source = E_source, T_source=T_source)
    initialization(metacommunity, initially_occupy_number=initially_occupy_number, 
                   length=length, width=width, E_source=E_source, T_source=T_source, L_e=L_e, L_t=L_t, L_n=L_n)
    offsprings_pool = generate_offsprings_pool()
    adapted_dispersal_storage = generate_adapted_dispersal_storage(patch_number = patch_number, length=length, width=width)
    
                
    
    
    all_time_step_gene_frequency = read_gene_frequency(metacommunity, length=length, width=width, L_e=L_e, L_t=L_t, L_n=L_n)
    all_time_step_species_distribution = read_species_distribution(metacommunity=metacommunity, length=length, width=width)
    
    for time_step in range(0,25):
        print('Replication=', replication_time, ', time_step=', time_step, ', len_of_pool=', len(offsprings_pool))
        dead_selection(metacommunity, length=length, width=width)
        current_offsprings_pool = reproduction(metacommunity, length=length, width=width, reproduction_type='sexual', L_e=L_e, L_t=L_t, birth_rate=1.0, mutation_rate=mutation_rate, time_step=time_step)
        offsprings_pool = merge_pool(offsprings_pool, current_offsprings_pool, offsprings_pool_memory, offsprings_pool_maxsize)
        local_compensation(metacommunity, length=length, width=width, offsprings_pool=offsprings_pool, initially_occupy_number=initially_occupy_number, patch_number_each_altitude=patch_number_each_altitude)
        all_time_step_gene_frequency=np.vstack((all_time_step_gene_frequency, read_gene_frequency(metacommunity, length=length, width=width, L_e=L_e, L_t=L_t, L_n=L_n)))
        all_time_step_species_distribution=np.vstack((all_time_step_species_distribution, read_species_distribution(metacommunity=metacommunity, length=length, width=width)))
    
    for time_step in range(25,50):
        print('Replication=', replication_time, ', time_step=', time_step, ', len_of_pool=', len(offsprings_pool))
        dead_selection(metacommunity, length=length, width=width)
        current_offsprings_pool = reproduction(metacommunity, length=length, width=width, reproduction_type='sexual', L_e=L_e, L_t=L_t, birth_rate=1.0, mutation_rate=mutation_rate, time_step=time_step)
        offsprings_pool = merge_pool(offsprings_pool, current_offsprings_pool, offsprings_pool_memory, offsprings_pool_maxsize)
        local_compensation(metacommunity, length=length, width=width, offsprings_pool=offsprings_pool, initially_occupy_number=initially_occupy_number, patch_number_each_altitude=patch_number_each_altitude)
        adapted_dispersal_storage = dispersal_only_within_patch(metacommunity, length=length, width=width, initially_occupy_number=initially_occupy_number, patch_number_each_altitude=patch_number_each_altitude, 
                                offsprings_pool=offsprings_pool, within_rate=within_rate, adapted_dispersal_storage=adapted_dispersal_storage)
        all_time_step_gene_frequency=np.vstack((all_time_step_gene_frequency, read_gene_frequency(metacommunity, length=length, width=width, L_e=L_e, L_t=L_t, L_n=L_n)))
        all_time_step_species_distribution=np.vstack((all_time_step_species_distribution, read_species_distribution(metacommunity=metacommunity, length=length, width=width)))
        
    for time_step in range(50, 5000):
        print('Replication=', replication_time, ', time_step=', time_step, ', len_of_pool=', len(offsprings_pool))
        dead_selection(metacommunity, length=length, width=width)
        current_offsprings_pool = reproduction(metacommunity, length=length, width=width, reproduction_type=reproduction_type, L_e=L_e, L_t=L_t, birth_rate=birth_rate, mutation_rate=mutation_rate, time_step=time_step)
        offsprings_pool = merge_pool(offsprings_pool, current_offsprings_pool, offsprings_pool_memory, offsprings_pool_maxsize)
        local_compensation(metacommunity, length=length, width=width, offsprings_pool=offsprings_pool, initially_occupy_number=initially_occupy_number, patch_number_each_altitude=patch_number_each_altitude)
        adapted_dispersal_storage = dispersal(metacommunity, length=length, width=width, offsprings_pool=offsprings_pool, source_rate=source_rate, across_rate=across_rate, among_rate=among_rate, within_rate=within_rate, 
                                          initially_occupy_number=initially_occupy_number, patch_number_each_altitude=patch_number_each_altitude, adapted_dispersal_storage=adapted_dispersal_storage)
        all_time_step_gene_frequency=np.vstack((all_time_step_gene_frequency, read_gene_frequency(metacommunity, length=length, width=width, L_e=L_e, L_t=L_t, L_n=L_n)))
        all_time_step_species_distribution=np.vstack((all_time_step_species_distribution, read_species_distribution(metacommunity=metacommunity, length=length, width=width)))
        
    fig = plt.figure(figsize=(10, 10))
    species_to_fig(metacommunity, patch_number=patch_number, initially_occupy_number=initially_occupy_number, patch_number_each_altitude=patch_number_each_altitude, fig=fig, length=length, width=width, 
                   file_name = 'indentifier %s%d.jpg'%(reproduction_type, replication_time)) 
    phenotype_to_fig(metacommunity, patch_number=patch_number, initially_occupy_number=initially_occupy_number, patch_number_each_altitude=patch_number_each_altitude, fig=fig, length=length, width=width,
                     file_name = 'phenotype %s%d.jpg'%(reproduction_type, replication_time))                  
    termotype_to_fig(metacommunity, patch_number=patch_number, initially_occupy_number=initially_occupy_number, patch_number_each_altitude=patch_number_each_altitude, fig=fig, length=length, width=width,
                     file_name = 'termotype %s%d.jpg'%(reproduction_type, replication_time))           
    
    
    species_to_csv(metacommunity, patch_number=patch_number, length=length, width=width,
                   file_name = 'identifier %s%d.csv'%(reproduction_type, replication_time))
    phenotype_to_csv(metacommunity, patch_number=patch_number, length=length, width=width,
                     file_name = 'phenotype %s%d.csv'%(reproduction_type, replication_time))
    termotype_to_csv(metacommunity, patch_number=patch_number, length=length, width=width,
                     file_name = 'termotype %s%d.csv'%(reproduction_type, replication_time))
    gene_frequency_to_cvs(all_time_step_gene_frequency=all_time_step_gene_frequency, gene_id=[0,60], file_name='all time gene frequency replication=%d.csv'%replication_time)
    species_distribution_to_csv(species_distribution_all_time_step=all_time_step_species_distribution, length=length, width=width, hab_num_patch=4, patch_number=patch_number, 
                                file_name='all time species_distribution replication=%d.csv'%replication_time)
    
    adapted_dispersal_to_csv(adapted_dispersal_storage, length=length, width=width, file_name='adapted_dispersal_record%s%d.csv'%(reproduction_type, replication_time))
    
    return metacommunity
#############################3333333333333333333333333333333333333333333333#############################################
if __name__ == '__main__':
    for replication_time in range(10):
        metacommunity = main(replication_time, offsprings_pool_maxsize=8000)
    
    
    
    
    
    
    
    
    
    
    
        