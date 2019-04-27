# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 19:48:21 2019

@author: Lin
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#############################################  初始化   ###################################################
def generate_habitat(e, t_sigmoid, length, width):
    microsite_e_values = np.random.normal(loc=0, scale=0.025, size=(length, width)) + e
    microsite_t_values = np.random.normal(loc=0, scale=0.025, size=(length, width)) + t_sigmoid
    microsite_individuals = [[None for i in range(length)] for i in range(width)]
    return {'microsite_e_values':microsite_e_values, 'microsite_t_values':microsite_t_values, 'microsite_individuals':microsite_individuals}
# habitat生成函数，habitat里包括了microsite_e_values 和 microsite_individuals 两个值, 分别是5*5的list

def sigmoid(x):
    return 1/(1+np.exp(-x))
# 设置s型函数，作为温度归一化函数
    
def generate_patch(e0=0.2, e1=0.4, e2=0.6, e3=0.8,
                   t0=0.0, t1=10, t2=20, t3=30):
    h0 = generate_habitat(e = e0, t_sigmoid=sigmoid((t0-15)/10), length=5, width=5)
    h1 = generate_habitat(e = e1, t_sigmoid=sigmoid((t1-15)/10), length=5, width=5)
    h2 = generate_habitat(e = e2, t_sigmoid=sigmoid((t2-15)/10), length=5, width=5)
    h3 = generate_habitat(e = e3, t_sigmoid=sigmoid((t3-15)/10), length=5, width=5)
    patch = {'h0':h0, 'h1':h1, 'h2':h2, 'h3':h3}
    return patch
# patch生成函数，一个patch包括4个habitat：h0、h1、h2和h3分别以e和t作为环境因子和温度的参数
    
def generate_matacommunity():
    metacommunity = {}
    for i in range(10):
        if i==0: patch = generate_patch(t0=0.0, t1=10, t2=20, t3=30)
        if i in [1,2,3]: patch = generate_patch(t0=5, t1=5, t2=5, t3=5)
        if i in [4,5,6]: patch = generate_patch(t0=15, t1=15, t2=15, t3=15)
        if i in [7,8,9]: patch = generate_patch(t0=25, t1=25, t2=25, t3=25)
        metacommunity['patch%s'%(str(i))] = patch
    return metacommunity
# metacommunity生成函数，其中p0为initially_occupied_patch,p1~p9为initially_empty_patch

def init_individual(e, t, sexual, L_e=10, L_t=10):               # sexual为None, male, female
    phenotype = e + random.gauss(0,0.025)
    termotype = sigmoid((t-15)/10) + random.gauss(0,0.025)
    genotype = [0 if e<np.random.uniform(0,1,1)[0] else 1 for i in range(L_e)] + [0 if sigmoid((t-15)/10)<np.random.uniform(0,1,1)[0] else 1 for i in range(L_t)]
    return {'identifier': int(e/0.2), 'sexual': sexual, 'phenotype':phenotype, 'termotype':termotype,'genotype':genotype}
# individual初始化函数，其中phenotype（表型）、termotype（温度型）、genotype（基因型）适应于initially_occupied_patch

def initialization(metacommunity, reproduction_type, occupied_patch=['patch0']):
    if reproduction_type == 'asexual':
        for patch_id in metacommunity:
            if patch_id in occupied_patch:
                patch = metacommunity[patch_id]
                for habitat_id in patch:
                    e = (int(habitat_id[1:]) + 1) * 0.2
                    t = int(habitat_id[1:]) * 10
                    for length in range(5):
                        for width in range(5):
                            new_individual = init_individual(e=e, t=t, L_e=10, L_t=10,sexual=None)
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][length][width] = new_individual
                            
    if reproduction_type == 'sexual':
        for patch_id in metacommunity:
            if patch_id in occupied_patch:
                patch = metacommunity[patch_id]
                for habitat_id in patch:
                    e = (int(habitat_id[1:]) + 1) * 0.2
                    t = int(habitat_id[1:]) * 10
                    for length in range(5):
                        for width in range(5):
                            if np.random.uniform(0,1,1)[0] > 0.5: new_individual = init_individual(e=e, t=t, L_e=10, L_t=10,sexual='male')
                            else: new_individual = init_individual(e=e, t=t, L_e=10, L_t=10,sexual='female')
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][length][width] = new_individual
    return 0
# 初始化函数，对initially_occupied_patch进行操作，e【0.2,0.4,0.6,0.8】、t【0,10,20,30】
#############################################  可视化  ######################################################
def get_values_set(metacommunity, patch_id, values_type):
    patch = metacommunity[patch_id]
    values_set = []
    for habitat_id in patch:
        habitat = patch[habitat_id]
        for length in range(5):
            for width in range(5):
                if habitat['microsite_individuals'][length][width]!=None:
                    values_set.append(habitat['microsite_individuals'][length][width][values_type])
                else:
                    values_set.append(np.nan) # 表示空物种
                    
    values_set = np.array(values_set).reshape(4,25)
    return values_set

def phenotype_to_fig(metacommunity, fig, file_name, values_type='phenotype'):

    plt.title('individuals_phenotype')
    location=[2,4,5,6,7,8,9,10,11,12]
    for patch_id in metacommunity:
        l = location[int(patch_id[-1])]
        fig.add_subplot(4,3,l)
        plt.tight_layout()
        plt.title(patch_id)
        df = pd.DataFrame(get_values_set(metacommunity, patch_id, values_type))
        sns.heatmap(data=df, vmax=1.0, vmin=0)
    
    plt.savefig(file_name)
    plt.clf()

def termotype_to_fig(metacommunity, fig, file_name, values_type='termotype'):

    plt.title('individuals_termotype')
    location=[2,4,5,6,7,8,9,10,11,12]
    for patch_id in metacommunity:
        l = location[int(patch_id[-1])]
        fig.add_subplot(4,3,l)
        plt.tight_layout()
        plt.title(patch_id)
        df = pd.DataFrame(get_values_set(metacommunity, patch_id, values_type))
        sns.heatmap(data=df, vmax=1.0, vmin=0)
    
    plt.savefig(file_name)
    plt.clf()
    
def species_to_fig(metacommunity, fig, file_name, values_type='identifier'):

    plt.title('species')
    location=[2,4,5,6,7,8,9,10,11,12]
    for patch_id in metacommunity:
        l = location[int(patch_id[-1])]
        fig.add_subplot(4,3,l)
        plt.tight_layout()
        plt.title(patch_id)
        df = pd.DataFrame(get_values_set(metacommunity, patch_id, values_type))
        sns.heatmap(data=df, vmax=5.0, vmin=1.0)
    
    plt.savefig(file_name)
    plt.clf()
    
##############################################  数据保存  ######################################################
def to_csv(metacommunity, file_name, values_type):
    location = ['X0Y0','X0Y1','X0Y2','X0Y3','X0Y4',
                'X1Y0','X1Y1','X1Y2','X1Y3','X1Y4',
                'X2Y0','X2Y1','X2Y2','X2Y3','X2Y4',
                'X3Y0','X3Y1','X3Y2','X3Y3','X3Y4',
                'X4Y0','X4Y1','X4Y2','X4Y3','X4Y4']
    patch_list = ['patch0']*4+['patch1']*4+['patch2']*4+['patch3']*4+['patch4']*4+['patch5']*4+['patch6']*4+['patch7']*4+['patch8']*4+['patch9']*4
    habitat_list = ['h0', 'h1', 'h2', 'h3']*10
    for patch_id in metacommunity:
        if patch_id == 'patch0':
            species_distribution_data = get_values_set(metacommunity, patch_id, values_type)
        else:
            values_set = get_values_set(metacommunity, patch_id, values_type)
            species_distribution_data = np.append(species_distribution_data, values_set, axis=0)
            
    index=pd.MultiIndex.from_arrays([patch_list,habitat_list], names=['patch_id','habitat_id'])
    columns = location
    df = pd.DataFrame(species_distribution_data,index=index, columns=columns)
    df.to_csv(file_name)
    
def termotype_to_csv(metacommunity, file_name, values_type='termotype'):
    return to_csv(metacommunity, file_name, values_type)

def phenotype_to_csv(metacommunity, file_name, values_type='phenotype'):
    return to_csv(metacommunity, file_name, values_type)

def species_to_csv(metacommunity, file_name, values_type='identifier'):
    return to_csv(metacommunity, file_name, values_type)
    
##############################################  死亡选择  ###################################################
def survival_rate(d, z, em, ti, tm, w = 0.5):
    survival_rate = (1-d) * math.exp((-1)*math.pow(((z-em)/w),2)) * math.exp((-1)*math.pow(((ti-tm)/w),2))
       # 存活的几率符合正态分布,d为基本死亡率，z为表型，em为环境因子，ti为温度型，tm环境温度归一化值
    return survival_rate
# d为基本的死亡量，z为某个个体的表型，e为该microsite的环境值，w为width of the fitness function


def dead_selection(metacommunity):
    for patch_id in metacommunity:      # metacommunity为一个字典，此语法得到的patch_id，为字典的key值
        patch = metacommunity[patch_id]
        for habitat_id in patch:        # patch为一个字典，此语法得到habitat_id, 为字典的一个key值
            habitat = patch[habitat_id]
            microsite_individuals_set = habitat['microsite_individuals']   # 一个5*5的list，包含25个e值
            microsite_e_values_set = habitat['microsite_e_values']         # 一个5*5的list，包含25个individual值
            microsite_t_values_set = habitat['microsite_t_values']
            for length in range(5):
                for width in range(5):
                    microsite_e_value = microsite_e_values_set[length][width]  # 一个microsite的e_value
                    microsite_t_value = microsite_t_values_set[length][width]
                    if microsite_individuals_set[length][width] != None:
                        microsite_individual = microsite_individuals_set[length][width] #一个microsite的individual值
                        phenotype = microsite_individual['phenotype']                  # 该microsite个体的一个phenotype值
                        termotype = microsite_individual['termotype']
                        survival_p = survival_rate(d=0.1, z=phenotype, em=microsite_e_value, ti=termotype, tm=microsite_t_value, w = 0.5)
                        # 通过表型和环境的e_value计算该个体的存活率
                        if survival_p < np.random.uniform(0,1,1):
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][length][width] = None
                            # 表示该个体已经死亡，用None表示
        
################################################################################################################
                            
############################################   reproduction   ##################################################
def species_division(metacommunity):
    species_category = {}
    for patch_id in metacommunity:
        patch = metacommunity[patch_id]
        species_category_patch ={}
        for habitat_id in patch:
            habitat = patch[habitat_id]
            species_category_habitat = {}
            microsite_individuals_set = habitat['microsite_individuals']
            species1, species2, species3, species4 = {'male':[], 'female':[]}, {'male':[], 'female':[]}, {'male':[], 'female':[]}, {'male':[], 'female':[]}
            for length in range(5):
                for width in range(5):
                    if microsite_individuals_set[length][width] != None:
                        microsite_individual = microsite_individuals_set[length][width]
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

def matching_parents(patch_id, habitat_id, species_category):
    matching_results = []
    species_category_habitat = species_category[patch_id][habitat_id]
    species1 = species_category_habitat['species1']
    species2 = species_category_habitat['species2']
    species3 = species_category_habitat['species3']
    species4 = species_category_habitat['species4']
    
    for matching in range(25//2): 
        if len(species1['male']) >= 1 and len(species1['female']) >= 1:
            male = random.sample(species1['male'], 1)[0]
            species1['male'].remove(male)
            female = random.sample(species1['female'], 1)[0]
            species1['female'].remove(female)
            matching_results.append((male, female))
        else:
            break
    for matching in range(25//2): 
        if len(species2['male']) >= 1 and len(species2['female']) >= 1:
            male = random.sample(species2['male'], 1)[0]
            species2['male'].remove(male)
            female = random.sample(species2['female'], 1)[0]
            species2['female'].remove(female)
            matching_results.append((male, female))
        else:
            break
    
    for matching in range(25//2): 
        if len(species3['male']) >= 1 and len(species3['female']) >= 1:
            male = random.sample(species3['male'], 1)[0]
            species3['male'].remove(male)
            female = random.sample(species3['female'], 1)[0]
            species3['female'].remove(female)
            matching_results.append((male, female))
        else:
            break
    
    for matching in range(25//2): 
        if len(species4['male']) >= 1 and len(species4['female']) >= 1:
            male = random.sample(species4['male'], 1)[0]
            species4['male'].remove(male)
            female = random.sample(species4['female'], 1)[0]
            species4['female'].remove(female)
            matching_results.append((male, female))
        else:
            break
    return matching_results


def reproduce(parents, reproduction_type):       #当无性繁殖时parents为一个亲本，当有性繁殖是parents为两个亲本
    if reproduction_type == 'asexual':           # 无性繁殖
         new = parents

    if reproduction_type == 'sexual':            # 有性繁殖
        male = parents[0]
        female = parents[1]
        female_gamete = random.sample([i for i in range(10)],5)   # 雌性配子所含的基因id
        n_identifier = female['identifier']
        n_genotype = []
            
        if 0.5 > np.random.uniform(0,1,1):       # 后代的性别，female与male均为50%
            n_sexual = 'male'
        else:
            n_sexual = 'female'
 
        for allelic in range(len(female['genotype'])): # 5个基因来自母本，5个基因来自父本
            if allelic in female_gamete:
                n_genotype.append(female['genotype'][allelic])
            else:
                n_genotype.append(male['genotype'][allelic])
                
        n_phenotype = np.mean(n_genotype[0:10]) + random.gauss(0,0.025)
        n_termotype = np.mean(n_genotype[10:20]) + random.gauss(0,0.025)
        new = {'identifier':n_identifier, 'sexual':n_sexual,'phenotype':n_phenotype, 'termotype':n_termotype ,'genotype':n_genotype}
    return new

def reproduction(metacommunity, reproduction_type, birth_rate = 0.5, offsprings_pool={}):
    if reproduction_type == 'asexual':
        for patch_id in metacommunity: 
            patch = metacommunity[patch_id]
            habitat_pool={}                           # 每个patch里分别有一个offspring的pool
            for habitat_id in patch:
                habitat = patch[habitat_id]
                microsite_individuals_set = habitat['microsite_individuals'] # 一个5*5的list，包含25个individual值
                pool = [] 
                for length in range(5):
                    for width in range(5):
                        random_normal_variable = np.random.uniform(0,1,1)[0]
                        if birth_rate < random_normal_variable and microsite_individuals_set[length][width] != None:
                        # 该microsite不为empty状态，并且以某一birth rate繁殖后代
                            parent = microsite_individuals_set[length][width]
                            new_individual = reproduce(reproduction_type = 'asexual', parents=parent)
                            pool.append(new_individual)          # 将一个patch里的所有后代储存起来
                if pool==[]:
                    habitat_pool[habitat_id]=[None]
                else:
                    habitat_pool[habitat_id] = pool                  # 一个habitat的子代集合
        
            offsprings_pool[patch_id] = habitat_pool     # 每个patch的子代个体集合,并以patch_id作为该集合的一个key值
            
    if reproduction_type == 'sexual':
        species_category = species_division(metacommunity)
        for patch_id in metacommunity:
            patch = metacommunity[patch_id]
            habitat_pool={}
            for habitat_id in patch:
                habitat = patch[habitat_id]
                pool = []
                parents_list = matching_parents(patch_id, habitat_id, species_category)
                for parents in parents_list:
                    new_individual = reproduce(reproduction_type = 'sexual', parents=parents)
                    pool.append(new_individual)
                if pool==[]:
                    habitat_pool[habitat_id]=[None]
                else:
                    habitat_pool[habitat_id] = pool
            offsprings_pool[patch_id] = habitat_pool
                    
    return offsprings_pool

################################################   mutation   ######################################################
def mutation(offsprings_pool, μ=0.0001):
    for patch_id in offsprings_pool:
        habitat_pool = offsprings_pool[patch_id]
        for habitat_id in habitat_pool:
            for i in range(len(habitat_pool[habitat_id])):
                individual = habitat_pool[habitat_id][i]
                if individual != None:
                    new_individual = {}
                    new_genotype = []
                    genotype = individual['genotype']
                    for allelic in range(len(genotype)):
                        random_normal_variable = np.random.uniform(0,1,1)[0]
                        if μ > random_normal_variable:
                            if genotype[allelic] ==0: new_genotype.append(1) 
                            if genotype[allelic] ==1: new_genotype.append(0) 
                            #print('yes',patch_id,habitat_id,i,allelic,genotype[allelic],new_genotype[allelic])
                        else:
                            new_genotype.append(genotype[allelic])
                    if new_genotype == genotype:
                        new_individual = individual
                    else:
                        new_phenotype = np.mean(new_genotype[0:10]) + random.gauss(0,0.025)
                        new_termotype = np.mean(new_genotype[10:20])
                        new_individual['identifier'] = individual['identifier']
                        new_individual['sexual'] = individual['sexual']
                        new_individual['phenotype'] = new_phenotype
                        new_individual['termotype'] = new_termotype
                        new_individual['genotype'] = new_genotype
                        offsprings_pool[patch_id][habitat_id][i] = new_individual
    return offsprings_pool
################################################## local compensation #########################################################        
def local_compensation(metacommunity, offsprings_pool):
    for patch_id in metacommunity:
        patch = metacommunity[patch_id]
        for habitat_id in patch:
            habitat = patch[habitat_id]
            microsite_individuals = habitat['microsite_individuals']
            local_pool = offsprings_pool[patch_id][habitat_id]
            for length in range(5):
                for width in range(5):
                    if microsite_individuals[length][width] == None and local_pool != [None] and local_pool != []:
                        compensation_individual = random.sample(local_pool, 1)[0]
                        local_pool.remove(compensation_individual)
                        metacommunity[patch_id][habitat_id]['microsite_individuals'][length][width] = compensation_individual

##################################################      dispersal     ########################################################
def dispersal_rate(dispersal_type):
    if dispersal_type == 'original':
        dispersal_rate = 0.01
    if dispersal_type == 'within':
        dispersal_rate = 0.1
    if dispersal_type == 'among':
        dispersal_rate = 0.05
    if dispersal_type == 'across': 
        dispersal_rate = 0.01
    return dispersal_rate

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

# across_pool, 不同海拔高度下迁移的子代个体
# among_pool, 同一海拔高度下不同的patch迁移的个体
# within_pool, 同一个patch之中不同habitat之间迁移的子代个体
# same_h, 同一个habitat里补充的新个体
    
def dispersal(metacommunity, immigrant_pool):
    a = dispersal_rate('original')
    b = dispersal_rate('original') + dispersal_rate('across')
    c = dispersal_rate('original') + dispersal_rate('across') + dispersal_rate('among')
    d = dispersal_rate('original') + dispersal_rate('across') + dispersal_rate('among') + dispersal_rate('within')
    for patch_id in metacommunity:
        patch = metacommunity[patch_id]
        for habitat_id in patch:
            within_pool, among_pool, across_pool, original_pool = find_pools(patch_id, habitat_id, immigrant_pool)
            habitat = patch[habitat_id]
            microsite_individuals_set = habitat['microsite_individuals']  # 一个5*5的list，包含25个individual值
            for length in range(5):
                for width in range(5):
                    if microsite_individuals_set[length][width] == None: # 找出当前处于empty状态的patch
                        random_uniform_variable = np.random.uniform(0,1,1)[0]
                        if 0 < random_uniform_variable < a and original_pool != [None]:
                            dispersal = random.sample(original_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][length][width] = dispersal
                        elif a < random_uniform_variable < b and across_pool != [None]:
                            dispersal = random.sample(across_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][length][width] = dispersal
                        elif b < random_uniform_variable < c and among_pool !=[None]:
                            dispersal = random.sample(among_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][length][width] = dispersal
                        elif c < random_uniform_variable < d and within_pool !=[None]:
                            dispersal = random.sample(within_pool, 1)[0]
                            metacommunity[patch_id][habitat_id]['microsite_individuals'][length][width] = dispersal
                        
def dispersal_only_within_patch(metacommunity, immigrant_pool):
    dispersal_within_rate = dispersal_rate('within')
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
    
    
    
    


#########################################################################################################################

def main(replication):
    metacommunity = generate_matacommunity()
    initialization(metacommunity=metacommunity, reproduction_type='sexual', occupied_patch=['patch0'])
    #由于前50个time_step要进行有性生殖，故初始化时考虑性别，但在50time_step以后性别属性不会被考虑
    fig = plt.figure(figsize=(8, 8))

    for time_step in range(0, 25):
        dead_selection(metacommunity)
        offsprings_pool = reproduction(metacommunity, reproduction_type='sexual', birth_rate = 1)
        offsprings_pool_mutation = mutation(offsprings_pool)
        local_compensation(metacommunity, offsprings_pool_mutation)

    
    for time_step in range(25, 50):
        dead_selection(metacommunity)
        offsprings_pool = reproduction(metacommunity, reproduction_type='sexual', birth_rate = 1)
        offsprings_pool_mutation = mutation(offsprings_pool)
        dispersal_only_within_patch(metacommunity, offsprings_pool_mutation)
        local_compensation(metacommunity, offsprings_pool_mutation)


    for time_step in range(50, 5000):
        dead_selection(metacommunity)
        offsprings_pool = reproduction(metacommunity, reproduction_type='asexual', birth_rate = 0.5)
        offsprings_pool_mutation = mutation(offsprings_pool)
        dispersal(metacommunity, immigrant_pool=offsprings_pool_mutation)
        local_compensation(metacommunity, offsprings_pool_mutation)

    phenotype_to_fig(metacommunity, fig, 
                   file_name = 'phenotype '+ 'asexual' +'time_step='+str(time_step)+'replication='+str(replication)+'.jpg')
    termotype_to_fig(metacommunity, fig, 
                   file_name = 'termotype '+ 'asexual' +'time_step='+str(time_step)+'replication='+str(replication)+'.jpg')
    species_to_fig(metacommunity, fig, 
                   file_name = 'species_identifier '+ 'asexual' +'time_step='+str(time_step)+'replication='+str(replication)+'.jpg')
    phenotype_to_csv(metacommunity, 
                   file_name = 'phenotype '+ 'asexual' +'time_step='+str(time_step)+'replication='+str(replication)+'.csv')
    termotype_to_csv(metacommunity, 
                   file_name = 'termotype '+ 'asexual' +'time_step='+str(time_step)+'replication='+str(replication)+'.csv')
    species_to_csv(metacommunity, 
                   file_name = 'species_identifier '+ 'asexual' +'time_step='+str(time_step)+'replication='+str(replication)+'.csv')
    

if __name__ == '__main__':
    for replication in range(100):
        main(replication)

'''
metacommunity = generate_matacommunity()
initialization(metacommunity=metacommunity, reproduction_type='sexual', occupied_patch=['patch0'])
fig = plt.figure(figsize=(8, 8))
#show_species(metacommunity, fig, file_name = 'sexual '+'init.jpg')
for time_step in range(0, 25):
    dead_selection(metacommunity)
    offsprings_pool = reproduction(metacommunity, reproduction_type='sexual', birth_rate = 1)
    offsprings_pool_mutation = mutation(offsprings_pool)
    local_compensation(metacommunity, offsprings_pool_mutation)
    #show_species(metacommunity, fig, file_name = 'species_identifier'+ 'sexual' +str(time_step)+'.jpg')
    
for time_step in range(25, 50):
    dead_selection(metacommunity)
    offsprings_pool = reproduction(metacommunity, reproduction_type='sexual', birth_rate = 1)
    offsprings_pool_mutation = mutation(offsprings_pool)
    dispersal_only_within_patch(metacommunity, offsprings_pool_mutation)
    local_compensation(metacommunity, offsprings_pool_mutation)
    #show_species(metacommunity, fig, file_name = 'species_identifier'+ 'sexual' +str(time_step)+'.jpg')

for time_step in range(50, 5000):
    dead_selection(metacommunity)
    offsprings_pool = reproduction(metacommunity, reproduction_type='sexual', birth_rate = 0.5)
    offsprings_pool_mutation = mutation(offsprings_pool)
    dispersal(metacommunity, immigrant_pool=offsprings_pool_mutation)
    local_compensation(metacommunity, offsprings_pool_mutation)
    #show_species(metacommunity, fig, file_name = 'species_identifier'+ 'sexual' +str(time_step)+'.jpg')
species_to_csv(metacommunity, 
               file_name = 'species_identifier'+ 'sexual' +'time_step='+str(time_step)+'replication='+str(1)+'.csv')
show_species(metacommunity, fig, 
               file_name = 'species_identifier'+ 'sexual' +'time_step='+str(time_step)+'replication='+str(1)+'.jpg')

'''


















