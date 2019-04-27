# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:22:21 2019

@author: Leo Lin
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#######################################  参数  ############################################################

#######################################  创建模型地貌  #####################################################
def generate_habitat(e, t_sigmoid, length, width):
    microsite_e_values = np.random.normal(loc=0, scale=0.025, size=(length, width)) + e
    microsite_t_values = np.random.normal(loc=0, scale=0.025, size=(length, width)) + t_sigmoid
    microsite_individuals = [[None for i in range(length)] for i in range(width)]
    return {'microsite_e_values':microsite_e_values, 'microsite_t_values':microsite_t_values, 
            'microsite_individuals':microsite_individuals}

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

def generate_matacommunity(patch_number, initially_occupy_number, patch_number_each_altitude, length, width,
                               e0, e1, e2, e3, T_upstream, T_midstream, T_downstream, E_source, T_source):  # E_source=[e0, e1, e2, e3], T_source=[T0, T1, T2, T3]
    metacommunity = {}                                                                                      # for initial occupied patches
    for i in range(patch_number):
        if i < initially_occupy_number: 
            patch = generate_patch(length=length, width=width, 
                                   e0=E_source[0], e1=E_source[1], e2=E_source[2], e3=E_source[3],
                                   t0=T_source[0], t1=T_source[1], t2=T_source[2], t3=T_source[3])
        if (i-initially_occupy_number)//patch_number_each_altitude == 0: 
            patch = generate_patch(length=length, width=width, e0=e0, e1=e1, e2=e2, e3=e3,
                                   t0=T_upstream, t1=T_upstream, t2=T_upstream, t3=T_upstream)
        if (i-initially_occupy_number)//patch_number_each_altitude == 1: 
            patch = generate_patch(length=length, width=width, e0=e0, e1=e1, e2=e2, e3=e3,
                                   t0=T_midstream, t1=T_midstream, t2=T_midstream, t3=T_midstream)
        if (i-initially_occupy_number)//patch_number_each_altitude == 2: 
            patch = generate_patch(length=length, width=width, e0=e0, e1=e1, e2=e2, e3=e3,
                                   t0=T_midstream, t1=T_midstream, t2=T_midstream, t3=T_midstream)
        metacommunity['patch%s'%(str(i))] = patch
    return metacommunity


#######################################  初始化并加入物种个体  ############################################################

def init_individual(habitat_id, e, T, sexual, L_e, L_t, L_n):               #sexual为None, male, female
    phenotype = e + random.gauss(0,0.025)
    termotype = sigmoid((T-26)/2) + random.gauss(0,0.025)
    frequency_for_neutral_gene = np.random.rand(10)                      #frequency_of_gene value = 1
    
    phenotye_gene = [0 if e<np.random.uniform(0,1,1)[0] else 1 for i in range(L_e)] 
    termotype_gene = [0 if sigmoid((T-26)/2)<np.random.uniform(0,1,1)[0] else 1 for i in range(L_t)]
    neutral_gene = [0 if frequency_for_neutral_gene[i]<np.random.uniform(0,1,1)[0] else 1 for i in range(L_n)]
    
    genotype = phenotye_gene + termotype_gene + neutral_gene
    return {'identifier': (int(habitat_id[1:])+1), 'sexual': sexual, 'phenotype':phenotype, 'termotype':termotype,'genotype':genotype} 


def initialization(metacommunity, initially_occupy_number, length, width, E_source, T_source, L_e, L_t, L_n): #E_source=[e0,e1,e2,e3], T_source=[T0,T1,T2,T3]
    for patch_id in metacommunity:
        if int(patch_id[-1]) < initially_occupy_number:
            patch = metacommunity[patch_id]
            for habitat_id in patch:
                e = E_source[int(habitat_id[1:])]
                t = T_source[int(habitat_id[1:])]
                for row in range(length):
                    for col in range(width):
                        if np.random.uniform(0,1,1)[0] > 0.5: new_individual = init_individual(habitat_id=habitat_id, e=e, T=t, L_e=L_e, L_t=L_t, L_n=L_n, sexual='male')
                        else: new_individual = init_individual(habitat_id=habitat_id, e=e, T=t, L_e=L_e, L_t=L_t, L_n=L_n, sexual='female')
                        metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = new_individual
    return metacommunity
#####################################################  可视化  #####################################################################  
def location(patch_number, initially_occupy_number, patch_number_each_altitude):
    initial_occupied_location = []
    upstream_location = []
    midestream_location = []
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
        midestream_location.append(i+1)
    for i in range(fig_size_length*3, fig_size_length*3+patch_number_each_altitude):
        downstream_location.append(i+1)
    return initial_occupied_location+upstream_location+midestream_location+downstream_location, (fig_size_length, fig_size_width)
    
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
                    values_set.append(np.nan) # 表示空物种
                    
    values_set = np.array(values_set).reshape(4,length*width)
    return values_set

def phenotype_to_fig(metacommunity, patch_number, initially_occupy_number, patch_number_each_altitude, fig, length, width, file_name, values_type='phenotype'):

    plt.title('individuals_phenotype')
    locate = location(patch_number, initially_occupy_number, patch_number_each_altitude)[0]
    fig_size = location(patch_number, initially_occupy_number, patch_number_each_altitude)[1] # fig_size=(fig_size_length, fig_size_width)
    
    for patch_id in metacommunity:
        l = locate[int(patch_id[-1])]
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
        l = locate[int(patch_id[-1])]
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
        l = locate[int(patch_id[-1])]
        fig.add_subplot(fig_size[1],fig_size[0],l)
        plt.tight_layout()
        plt.title(patch_id)
        df = pd.DataFrame(get_values_set(metacommunity, patch_id, length, width, values_type))
        sns.heatmap(data=df, vmax=5.0, vmin=1.0)
    
    plt.savefig(file_name)
    plt.clf()
#################################### 数据保存 ##################################################################
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

def to_csv(metacommunity, length, width, file_name, values_type):
    XY = coordinate(length, width)
    patch_index = generate_index_of_patch(metacommunity)
    habitat_index = ['h0', 'h1', 'h2', 'h3']*10
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
    
def termotype_to_csv(metacommunity, length, width, file_name, values_type='termotype'):
    return to_csv(metacommunity, length, width, file_name, values_type)

def phenotype_to_csv(metacommunity, length, width, file_name, values_type='phenotype'):
    return to_csv(metacommunity, length, width, file_name, values_type)

def species_to_csv(metacommunity, length, width, file_name, values_type='identifier'):
    return to_csv(metacommunity, length, width, file_name, values_type)

def read_gene_frequency(metacommunity, length, width, gene_id):
    sum_genotype = np.array([0]*30)
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
    return (sum_genotype/individual_counter)[gene_id[0]:gene_id[1]]

def gene_frequency_all_time_step_to_cvs(all_time_step_gene_frequency, gene_id, file_name):

    columns = ['gene%d'%i for i in range(gene_id[0],gene_id[1])]
    index = ['initial_gene_frequency']+['time_step%d'%i for i in range(5000)]
    
    df_all_time_step_frequency = pd.DataFrame(all_time_step_gene_frequency, index=index, columns=columns)
    df_all_time_step_frequency.to_csv(file_name)
     
    
def gene_frequency_to_csv(gene_frequency_set_start, gene_frequency_set_end, gene_id, file_name):
    
    gene_frequency_data = np.vstack(([gene_frequency_set_start], [gene_frequency_set_end]))
    columns = ['gene%d'%i for i in range(gene_id[0],gene_id[1])]
    index = ['inital_gene_frequency', 'finished_gene_frequency']

    frequency_Dataframe = pd.DataFrame(gene_frequency_data, index=index, columns=columns)
    frequency_Dataframe.to_csv(file_name)
###############################################  死亡选择  ###########################################################
def survival_rate(d, z, em, ti, tm, w = 0.5):
    survival_rate = (1-d) * math.exp((-1)*math.pow(((z-em)/w),2)) * math.exp((-1)*math.pow(((ti-tm)/w),2))
       # 存活的几率符合正态分布,d为基本死亡率，z为表型，em为环境因子，ti为温度型，tm环境温度归一化值
    return survival_rate
# d为基本的死亡量，z为某个个体的表型，e为该microsite的环境值，w为width of the fitness function

def dead_selection(metacommunity, length, width):
    for patch_id in metacommunity:      # metacommunity为一个字典，此语法得到的patch_id，为字典的key值
        patch = metacommunity[patch_id]
        for habitat_id in patch:        # patch为一个字典，此语法得到habitat_id, 为字典的一个key值
            habitat = patch[habitat_id]
            microsite_individuals_set = habitat['microsite_individuals']   # 一个5*5的list，包含25个e值
            microsite_e_values_set = habitat['microsite_e_values']         # 一个5*5的list，包含25个individual值
            microsite_t_values_set = habitat['microsite_t_values']
            for row in range(length):
                for col in range(width):
                    microsite_e_value = microsite_e_values_set[row][col]  # 一个microsite的e_value
                    microsite_t_value = microsite_t_values_set[row][col]
                    if microsite_individuals_set[row][col] != None:
                        microsite_individual = microsite_individuals_set[row][col] #一个microsite的individual值
                        phenotype = microsite_individual['phenotype']                  # 该microsite个体的一个phenotype值
                        termotype = microsite_individual['termotype']
                        survival_p = survival_rate(d=0.1, z=phenotype, em=microsite_e_value, ti=termotype, tm=microsite_t_value, w = 0.5)
                        # 通过表型和环境的e_value计算该个体的存活率
                        if survival_p < np.random.uniform(0,1,1):
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = None
                            # 表示该个体已经死亡，用None表示
####################################################  繁殖   ######################################################################
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

def reproduce(parents, reproduction_type):       #当无性繁殖时parents为一个亲本，当有性繁殖是parents为两个亲本
    if reproduction_type == 'asexual':           # 无性繁殖
         
         n_identifier = parents['identifier']
         n_sexual = parents['sexual']
         n_genotype = parents['genotype']
         n_termotype = np.mean(n_genotype[0:10]) + random.gauss(0,0.025)
         n_phenotype = np.mean(n_genotype[10:20]) + random.gauss(0,0.025)
         
         new = {'identifier':n_identifier, 'sexual':n_sexual,'phenotype':n_phenotype, 'termotype':n_termotype ,'genotype':n_genotype}
         
    if reproduction_type == 'sexual':            # 有性繁殖
        male = parents[0]
        female = parents[1]
        locus_num = len(female['genotype'])
        female_gamete = random.sample([i for i in range(locus_num)],int(locus_num/2))   # 雌性配子所含的基因id
        n_identifier = female['identifier']
        n_genotype = []
            
        if 0.5 > np.random.uniform(0,1,1): n_sexual = 'male' # 后代的性别，female与male均为50%
        else: n_sexual = 'female'
 
        for allelic in range(locus_num): # 5个基因来自母本，5个基因来自父本
            if allelic in female_gamete: n_genotype.append(female['genotype'][allelic])
            else: n_genotype.append(male['genotype'][allelic])
                
        n_phenotype = np.mean(n_genotype[0:10]) + random.gauss(0,0.025)
        n_termotype = np.mean(n_genotype[10:20]) + random.gauss(0,0.025)
        new = {'identifier':n_identifier, 'sexual':n_sexual,'phenotype':n_phenotype, 'termotype':n_termotype ,'genotype':n_genotype}
    return new
#############################################################################################################################################
def generate_offsprings_pool():
    return []
def eliminate_offsprings_pool():
    return []

def fix_offsprings_pool_size(offsprings_pool, offsprings_pool_maxsize):
    if len(offsprings_pool) <= offsprings_pool_maxsize:
        return offsprings_pool
    if len(offsprings_pool) > offsprings_pool_maxsize:
        offsprings_pool.remove(offsprings_pool[0])
        return fix_offsprings_pool_size(offsprings_pool, offsprings_pool_maxsize)

def reproduction(metacommunity, length, width, reproduction_type, birth_rate, offsprings_pool, offsprings_pool_memory):
    if offsprings_pool_memory == False: offsprings_pool = eliminate_offsprings_pool()
    if offsprings_pool_memory == True: offsprings_pool = offsprings_pool
    if reproduction_type == 'asexual':
        for patch_id in metacommunity: 
            patch = metacommunity[patch_id]
            for habitat_id in patch:
                habitat = patch[habitat_id]
                microsite_individuals_set = habitat['microsite_individuals'] # 一个5*5的list，包含25个individual值
                for row in range(length):
                    for col in range(width):
                        random_uniform_variable = np.random.uniform(0,1,1)[0]
                        if birth_rate > random_uniform_variable and microsite_individuals_set[row][col] != None:
                        # 该microsite不为empty状态，并且以某一birth rate繁殖后代
                            parent = microsite_individuals_set[row][col]
                            new_individual = reproduce(reproduction_type = 'asexual', parents=parent)
                            offsprings_pool.append((patch_id, habitat_id, new_individual))          # 将一个patch里的所有后代储存起来
                                
    if reproduction_type == 'sexual':
        species_category = species_division(metacommunity, length, width)
        for patch_id in metacommunity:
            patch = metacommunity[patch_id]
            for habitat_id in patch:
                habitat = patch[habitat_id]
                parents_list = matching_parents(length, width, patch_id, habitat_id, species_category)
                for parents in parents_list:
                    random_uniform_variable = np.random.uniform(0,1,1)[0]
                    if birth_rate > random_uniform_variable:
                        new_individual = reproduce(reproduction_type = 'sexual', parents=parents)
                        offsprings_pool.append((patch_id, habitat_id, new_individual))
                        
    return offsprings_pool

def mutate(individual, mutation_rate):
    genotype = individual['genotype']
    new_genotype = []
    for allelic in range(len(genotype)):
        random_uniform_variable = np.random.uniform(0,1,1)[0]
        if mutation_rate > random_uniform_variable:
            if genotype[allelic] ==0: new_genotype.append(1) 
            if genotype[allelic] ==1: new_genotype.append(0) 
        else:
            new_genotype.append(genotype[allelic])
    if new_genotype == genotype:
        new_individual = individual
        return new_individual

    else: 
        new_phenotype = np.mean(new_genotype[0:10]) + random.gauss(0,0.025)
        new_termotype = np.mean(new_genotype[10:20]) + random.gauss(0,0.025)
        return {'identifier':individual['identifier'], 'sexual':individual['sexual'], 'genotype':new_genotype, 'termotype':new_termotype, 'phenotype':new_phenotype}          
            
            
def mutation(offsprings_pool, mutation_rate):
    mutated_offsprings_pool = []
    for offsprings_information in offsprings_pool:
        offsprings_individual_patch_id = offsprings_information[0]
        offsprings_individual_habitat_id = offsprings_information[1]
        offsprings_individual = offsprings_information[2]
        mutated_individual = mutate(offsprings_individual, mutation_rate)
        mutated_offsprings_pool.append((offsprings_individual_patch_id, offsprings_individual_habitat_id, mutated_individual))
    return mutated_offsprings_pool
    
############################################  迁移  ##################################################################
def generate_immigrant_pool(metacommunity): 
    immigrant_pool={}
    for patch_id in metacommunity:
        immigrant_patch_pool = {}
        for habitat_id in metacommunity[patch_id]:
            immigrant_patch_pool[habitat_id]=[]
        immigrant_pool[patch_id] = immigrant_patch_pool
    return immigrant_pool

def offsprings_pool_to_immigrant_pool(offsprings_pool, immigrant_pool):
    for offspring_information in offsprings_pool:
        patch_id = offspring_information[0]
        habitat_id = offspring_information[1]
        offspring_individual = offspring_information[2]
        immigrant_pool[patch_id][habitat_id].append(offspring_individual)
    return immigrant_pool

def find_pools(patch_id, habitat_id, immigrant_pool):         # patch_id和habitat_id作为分类指标,将immigrant_pool的子代分为4类
    within_pool, among_pool, across_pool, original_pool = [],[],[],[]
    for p_id in immigrant_pool:
        immigrant_pool_patch = immigrant_pool[p_id]
        for h_id in immigrant_pool_patch:
            immigrant_pool_habitat=immigrant_pool_patch[h_id]
            if immigrant_pool_habitat != [None]:
                for offspring in immigrant_pool_habitat:
                    if p_id == 'patch0': original_pool.append(offspring)
                    elif p_id == patch_id and h_id != habitat_id: within_pool.append(offspring)
                    elif p_id != patch_id and abs(((int(patch_id[-1])-1)//3 - (int(p_id[-1])-1)//3)) == 0: among_pool.append(offspring)
                    elif p_id != patch_id and ((int(patch_id[-1])-1)//3 - (int(p_id[-1])-1)//3) == 1: across_pool.append(offspring)

    if across_pool ==[]: across_pool = [None]
    if among_pool == []: among_pool = [None]
    if within_pool ==[]: within_pool = [None]
    return within_pool, among_pool, across_pool, original_pool


def dispersal(metacommunity, length, width, immigrant_pool, source_rate, across_rate, among_rate, within_rate):
    a = source_rate
    b = source_rate + across_rate
    c = source_rate + across_rate + among_rate
    d = source_rate + across_rate + among_rate + within_rate
    for patch_id in metacommunity:
        patch = metacommunity[patch_id]
        for habitat_id in patch:
            within_pool, among_pool, across_pool, original_pool = find_pools(patch_id, habitat_id, immigrant_pool)
            habitat = patch[habitat_id]
            microsite_individuals_set = habitat['microsite_individuals']  # 一个5*5的list，包含25个individual值
            for row in range(length):
                for col in range(width):
                    if microsite_individuals_set[row][col] == None: # 找出当前处于empty状态的patch
                        random_uniform_variable = np.random.uniform(0,1,1)[0]
                        if 0 < random_uniform_variable < a and original_pool != [None]:
                            dispersal = random.sample(original_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = dispersal
                        elif a < random_uniform_variable < b and across_pool != [None]:
                            dispersal = random.sample(across_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = dispersal
                        elif b < random_uniform_variable < c and among_pool !=[None]:
                            dispersal = random.sample(among_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = dispersal
                        elif c < random_uniform_variable < d and within_pool !=[None]:
                            dispersal = random.sample(within_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = dispersal
                            
def dispersal_only_within_patch(metacommunity, immigrant_pool, within_rate):
    dispersal_within_rate = within_rate
    for patch_id in metacommunity:
        patch = metacommunity[patch_id]
        for habitat_id in patch:
            within_pool = find_pools(patch_id, habitat_id, immigrant_pool)[0]
            habitat = patch[habitat_id]
            microsite_individuals_set = habitat['microsite_individuals']
            for length in range(5):
                for width in range(5):
                    if microsite_individuals_set[length][width] == None:
                        random_uniform_variable = np.random.uniform(0,1,1)[0]
                        if 0 < random_uniform_variable < dispersal_within_rate and within_pool != [None]:
                            dispersal = random.sample(within_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][length][width] = dispersal

def local_compensation(metacommunity, length, width, immigrant_pool):
    for patch_id in metacommunity:
        patch = metacommunity[patch_id]
        for habitat_id in patch:
            habitat = patch[habitat_id]
            microsite_individuals = habitat['microsite_individuals']
            local_pool = immigrant_pool[patch_id][habitat_id]
            for row in range(length):
                for col in range(width):
                    if microsite_individuals[row][col] == None and local_pool != [None] and local_pool != []:
                        compensation_individual = random.sample(local_pool, 1)[0]
                        local_pool.remove(compensation_individual)
                        metacommunity[patch_id][habitat_id]['microsite_individuals'][row][col] = compensation_individual         
                                
############################################  main program  ####################################################################  
def main(replication):
    patch_number = 10                                                          #patch总数量，包括initial_occupied 和 initial_empty
    initially_occupy_number = 1                                                #initial_occupied_patch 的数量
    patch_number_each_altitude = 3                                             #每个海拔高度的patch的数量
    length = 6                                                                 #habitat的长
    width = 6                                                                  #habitat的宽
    e0 = 0.2                                                                   #patch中 habitat='h0'时的环境值
    e1 = 0.4                                                                   #patch中 habitat='h1'时的环境值
    e2 = 0.6                                                                   #patch中 habitat='h2'时的环境值
    e3 = 0.8                                                                   #上游河流的温度
    T_midstream = 26                                                           #patch中 habitat='h3'时的环境值
    T_upstream = 24                                                            #中游河流的温度
    T_downstream = 28                                                          #下游河流的温度
    E_source = [0.2, 0.4, 0.6, 0.8]                                            #物种源地每个habitat分别的环境值
    T_source=[23,25,27,29]                                                     #物种源地每个habitat分别的温度
    L_e=10                                                                     #决定物种表型（与环境值匹配）的基因数量
    L_t=10                                                                     #决定物种温度型（与温度值匹配）的基因数量
    L_n=10                                                                     #物种中性基因的数量
    reproduction_type='asexual'                                                #物种的繁殖方式
    birth_rate=0.5                                                             #出生率
    mutation_rate = 0.00001                                                    #突变率
    offsprings_pool_memory=False                                               #储存后代的库是否具有记忆性，及是否保存上一个time_step所产生的后代的数据
    offsprings_pool_maxsize=1000                                               #当offsprings_pool_memory = True时起作用，决定储存后代的库的大小
    source_rate=0.01                                                           #物种个体从源地迁出，迁入到empty_microsite的迁移率
    across_rate=0.01                                                           #物种个体跨越海拔高度迁移时的迁移率
    among_rate=0.05                                                            #物种个体在同一海拔高度，不同patch之间迁移时的迁移率
    within_rate=0.1                                                            #物种个体在同一个patch不同habitat之间迁移时的迁移率
    
    metacommunity = generate_matacommunity(patch_number=patch_number, initially_occupy_number=initially_occupy_number, 
                                           patch_number_each_altitude = patch_number_each_altitude,
                                           length = length, width = width, e0 = e0, e1 = e1, e2 = e2, e3 = e3,
                                           T_upstream=T_upstream, T_midstream=T_midstream, T_downstream=T_downstream, 
                                           E_source = E_source, T_source=T_source)          
    fig = plt.figure(figsize=(8, 8))
    initialization(metacommunity, initially_occupy_number=initially_occupy_number, length=length, width=width, E_source=E_source, T_source=T_source, L_e=L_e, L_t=L_t, L_n=L_n)            
    initial_gene_frequency = read_gene_frequency(metacommunity, length=length, width=width, gene_id=[20,30])
    all_time_step_gene_frequency = initial_gene_frequency
    offsprings_pool = generate_offsprings_pool()
    
    for time_step in range(0,25):
        print(replication, time_step)
        dead_selection(metacommunity, length=length, width=width)
        offsprings_pool = reproduction(metacommunity, length=length, width=width, reproduction_type='sexual', birth_rate=1, offsprings_pool=offsprings_pool, offsprings_pool_memory=offsprings_pool_memory)
        mutation_offsprings_pool = mutation(offsprings_pool, mutation_rate)
        immigrant_pool = generate_immigrant_pool(metacommunity)
        immigrant_pool = offsprings_pool_to_immigrant_pool(offsprings_pool=mutation_offsprings_pool, immigrant_pool=immigrant_pool)
        local_compensation(metacommunity, length=length, width=width, immigrant_pool=immigrant_pool)
        all_time_step_gene_frequency=np.vstack((all_time_step_gene_frequency, read_gene_frequency(metacommunity, length=length, width=width, gene_id=[20,30])))
        #print(time_step, read_gene_frequency(metacommunity, length=length, width=width))
    
    for time_step in range(25,50):
        print(replication, time_step)
        dead_selection(metacommunity, length=length, width=width)
        offsprings_pool = reproduction(metacommunity, length=length, width=width, reproduction_type='sexual', birth_rate=1, offsprings_pool=offsprings_pool, offsprings_pool_memory=offsprings_pool_memory)
        mutation_offsprings_pool = mutation(offsprings_pool, mutation_rate)
        immigrant_pool = generate_immigrant_pool(metacommunity)
        immigrant_pool = offsprings_pool_to_immigrant_pool(offsprings_pool=mutation_offsprings_pool, immigrant_pool=immigrant_pool)
        dispersal_only_within_patch(metacommunity, immigrant_pool, within_rate=within_rate)
        local_compensation(metacommunity, length=length, width=width, immigrant_pool=immigrant_pool)
        all_time_step_gene_frequency=np.vstack((all_time_step_gene_frequency, read_gene_frequency(metacommunity, length=length, width=width, gene_id=[20,30])))
        #print(time_step, read_gene_frequency(metacommunity, length=length, width=width))
    
    for time_step in range(50,5000):
        print(replication, time_step)
        dead_selection(metacommunity, length=length, width=width)
        offsprings_pool = reproduction(metacommunity, length=length, width=width, reproduction_type=reproduction_type, birth_rate=birth_rate, offsprings_pool=offsprings_pool, offsprings_pool_memory=offsprings_pool_memory)
        mutation_offsprings_pool = mutation(offsprings_pool, mutation_rate)
        immigrant_pool = generate_immigrant_pool(metacommunity)
        immigrant_pool = offsprings_pool_to_immigrant_pool(offsprings_pool=mutation_offsprings_pool, immigrant_pool=immigrant_pool)
        dispersal(metacommunity, length=length, width=width, immigrant_pool=immigrant_pool, source_rate=source_rate, across_rate=across_rate, among_rate=among_rate, within_rate=within_rate)
        local_compensation(metacommunity, length=length, width=width, immigrant_pool=immigrant_pool)
        all_time_step_gene_frequency=np.vstack((all_time_step_gene_frequency, read_gene_frequency(metacommunity, length=length, width=width, gene_id=[20,30])))
        #print(time_step, read_gene_frequency(metacommunity, length=length, width=width))
  
    species_to_fig(metacommunity, patch_number=patch_number, initially_occupy_number=initially_occupy_number, patch_number_each_altitude=patch_number_each_altitude, fig=fig, length=length, width=width, 
                   file_name = 'indentifier replication=%d.jpg'%replication)            
    termotype_to_fig(metacommunity, patch_number=patch_number, initially_occupy_number=initially_occupy_number, patch_number_each_altitude=patch_number_each_altitude, fig=fig, length=length, width=width,
                   file_name = 'termotype replication=%d.jpg'%replication)           
    phenotype_to_fig(metacommunity, patch_number=patch_number, initially_occupy_number=initially_occupy_number, patch_number_each_altitude=patch_number_each_altitude, fig=fig, length=length, width=width,
                   file_name = 'phenotype replication=%d.jpg'%replication)            
    phenotype_to_csv(metacommunity, length=length, width=width,
                   file_name = 'phenotype replication=%d.csv'%replication)
    termotype_to_csv(metacommunity, length=length, width=width,
                   file_name = 'termotype replication=%d.csv'%replication)
    species_to_csv(metacommunity, length=length, width=width,
                   file_name = 'identifier replication=%d.csv'%replication)
    
    finished_gene_frequency = read_gene_frequency(metacommunity, length=length, width=width, gene_id=[20,30])            
    gene_frequency_to_csv(gene_frequency_set_start=initial_gene_frequency, gene_frequency_set_end=finished_gene_frequency, gene_id=[20,30], file_name='gene frequency replication=%d.csv'%replication)
    gene_frequency_all_time_step_to_cvs(all_time_step_gene_frequency=all_time_step_gene_frequency, gene_id=[20,30], file_name='all time gene frequency replication=%d.csv'%replication)



if __name__ == '__main__':
    for replication in range(100):
        main(replication)            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            