from __future__ import division
from concurrent.futures import ProcessPoolExecutor
from pyomo.environ import *
import csv
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations
import pickle
import pdb
import pandas as pd
import pickle
from typing import Any
import QlearnExp_v14_func_SimOpt as Qlearn
import QlearnExp_v14_func_SimOptRedS as QlearnRedS
import Rollout_v4_MC_func_OptRoll_par_rest as RolloutMCrest   #
import Rollout_v4_MC_func_OptRoll_par as RolloutMC   #
import kitchenheuristic as Heur
import Analyze as plots
import QlearnExp_v14_func_Greed as Qlearn_base

#Inside 
# sim-opt based Q-learn (*2)
# Qlearn Greedy (1)
# rollout parallel Sim  (*2)
# heuristic decision  (1)

#some initial inputs are read and sent to inside

debug_mode=False
Recipes={}
with open('Recipes.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        else:
            Recipes[line_count]=row[1:len(row)]  #int(row[selected_menu][ingredient_no])
            line_count += 1
num_menuoptions=len(Recipes)
Prices={} #for the whole package
with open('Pricesv1.csv', mode='r') as Prices_file:
    Prices_reader = csv.reader(Prices_file)
    line_count = 0
    for row in Prices_reader:
        if line_count == 0:
            line_count += 1
            continue
        else:
            Prices[line_count]=int(row[1][0])
            line_count += 1
num_ingredients=len(Prices)
NutritionalValue={} #for the whole package

with open('Nutrition.csv', mode='r') as Nutrition_file:
    Nutrition_reader = csv.reader(Nutrition_file)
    line_count = 0
    for row in Nutrition_reader:
        if line_count == 0:
            line_count += 1
            continue
        else:
            NutritionalValue[line_count]=int(row[1])
            line_count += 1

#################NEXT ACTION########################################################################################
num_MCepisodes=7 #100 generate num_MCepisodes many random values for price(i.e. stochastic parameter)
lenOfEpisode=15
total_nutrition=0
total_waste=0
total_cost=0
initial_budget=3000
cheapest_menu=8
most_exp_menu=41 #prices are given for pacekage size
max_pack_size=4
usagetime=3
scarcitypricecoeff=3
numberOfIntervals=initial_budget//(cheapest_menu)
menu_with_max_ingredients=13
max_waste=(menu_with_max_ingredients*4)  #in units of quarter package
last3days=[]
scarce_ingredients=[]   #6,17
monthly_selection=[]
monthly_selection_details=[]
failed=0
reward_details={}
unused={}
dailywasted=0
monthid=1
xfixes={}

# dictionary that holds the stock age of ingredients
for ingre in range(1,num_ingredients+1):
    for age in range(1, usagetime+1):
        unused[ingre,age]=0 # quantity and age of unused inredient

####################################################################################################################################################################
# Utils to pickle output files

def save_object(obj: Any, filename: str) -> None:
    """Save Python object to file (binary)."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(filename: str) -> Any:
    """Load Python object from file (binary)."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


################################################################ MAIN  ####################################################################################################
########################################################################################################################################################################################################
if __name__ == "__main__":
       
    trainingDuration=2000
    rng2 = np.random.default_rng(seed=399)
    warmstartoff=True
    distribution="lognormal"
    numOfEpisodes=50   
    
    pdb.set_trace() 
    ################################################## test RolloutMC #############################################
    
    menus_to_choose=[] #when empty no restricted list of menus to overwrite the menu optins inside the function
    summary_RollMC, time_to_RollMC=RolloutMC.test_rollout_MC(numOfEpisodes, lenOfEpisode, num_MCepisodes, num_ingredients, usagetime, rng2, distribution, 
        num_menuoptions, Recipes, Prices, NutritionalValue, scarce_ingredients, max_pack_size, max_waste, most_exp_menu, initial_budget, menus_to_choose, scarcitypricecoeff)#-----------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Save
    save_object(summary_RollMC, "summary_RollMC_log.pkl")
    # Load
    #summary_RollMC = load_object("summary_RollMC_log.pkl")
    #summary_RollMC = load_object("summary_RollMC_log_cri.pkl") 
    pdb.set_trace()     
    ################################################## test RolloutMC Reduced Control Options#############################################
    
    menus_to_choose=[]
    summary_RollMCReds, time_to_RollMC=RolloutMCrest.test_rollout_MC(numOfEpisodes, lenOfEpisode, num_MCepisodes, num_ingredients, usagetime, rng2, distribution, 
        num_menuoptions, Recipes, Prices, NutritionalValue, scarce_ingredients, max_pack_size, max_waste, most_exp_menu, initial_budget, menus_to_choose, scarcitypricecoeff)#-----------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Save
    save_object(summary_RollMCReds, "summary_RollMC_log_rest.pkl")
    # Load
    #summary_RollMCReds = load_object("summary_RollMC_log_rest.pkl")    
    #summary_RollMCReds = load_object("summary_RollMC_log_rest_cri.pkl")    
    pdb.set_trace()
    ################################################## test Heuristic #############################################
    
    summary_Heuristic, time_to_Heuristic=Heur.test_Heuristic(numOfEpisodes, lenOfEpisode, rng2, scarcitypricecoeff, scarce_ingredients, distribution, rng2 , initial_budget)  #-----------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Save
    save_object(summary_Heuristic, "summary_Heuristic_log.pkl")
    # Load
    #summary_Heuristic = load_object("summary_Heuristic_log.pkl")
    #summary_Heuristic = load_object("summary_Heuristic_log_cris.pkl")  
    pdb.set_trace()
    
    ################################################## test Q-learn Advanced #############################################
    QFactorGrid, QFactorGridPrev, time_to_train=Qlearn.prep_QMatrix(trainingDuration, lenOfEpisode, warmstartoff, distribution, initial_budget)#-----------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    save_object(QFactorGrid, "QFactorGrid_adv2000_log.pkl")

    '''
    #Training duration testing loop
    for i in range(1,21):
        filename="QFactorGrid_adv_log"+str(i*100)+".pkl"
        QFactorGrid= load_object(filename) 
        summary_Qlearn=QlearnRedS.simulateQLearn(numOfEpisodes, QFactorGrid, lenOfEpisode, scarcitypricecoeff, scarce_ingredients, distribution, rng2, initial_budget)

        # Save
        filename="summary_QFactorGrid_adv_log"+str(i*100)+".pkl"
        save_object(summary_Qlearn, filename)  
    df_means=plots.compare_trainDuration()
    pdb.set_trace()    
    '''
    pdb.set_trace()
    QFactorGrid = load_object("QFactorGrid_adv2000_log.pkl")
    summary_Qlearn=Qlearn.simulateQLearn(numOfEpisodes, QFactorGrid, lenOfEpisode, scarcitypricecoeff, scarce_ingredients, distribution, rng2, initial_budget)
    save_object(summary_Qlearn, "QFactorGrid_adv2000_log.pkl")
    
    # Load
    #summary_Qlearn=load_object("summary_QFactorGrid_adv_log200.pkl")
    #summary_Qlearn=load_object("summary_QFactorGrid_adv_log200_cris.pkl")  
     
    
    ################################################## test Q-learn Advanced Reduced States #############################################
    
    QFactorGrid_reds, QFactorGridPrev_reds, time_to_train_reds=QlearnRedS.prep_QMatrix(trainingDuration, lenOfEpisode, warmstartoff, distribution, initial_budget)#-----------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    save_object(QFactorGrid_reds, "QFactorGrid_reds_log.pkl")
    #pdb.set_trace()
       
    #******************************************
    '''
    #Training duration testing loop   
    for i in range(1,21):
        filename="QFactorGrid_adv_reds_log"+str(i*100)+".pkl"
        QFactorGrid= load_object(filename) 
        summary_Qlearn=QlearnRedS.simulateQLearn(numOfEpisodes, QFactorGrid, lenOfEpisode, scarcitypricecoeff, scarce_ingredients, distribution, rng2, initial_budget)

        # Save
        filename="summary_Qlearn_adv_reds_log"+str(i*100)+".pkl"
        save_object(summary_Qlearn, filename)    
    #df_means=plots.compare_trainDuration()
    pdb.set_trace()
    '''
    QFactorGrid_reds=load_object("QFactorGrid_reds_log140.pkl")
    summary_Qlearn_reds=QlearnRedS.simulateQLearn(numOfEpisodes, QFactorGrid_reds, lenOfEpisode, scarcitypricecoeff, scarce_ingredients, distribution, rng2, initial_budget)
    # Save
    save_object(summary_Qlearn_reds, "summary_Qlearn_reds_log.pkl")
    # Load    
    #summary_Qlearn_reds=load_object("summary_Qlearn_adv_reds_log140.pkl")
    #summary_Qlearn_reds=load_object("summary_Qlearn_reds_log140_cris.pkl")
    ################################################## test Q-learn base (Greedy) #############################################
    #trainingDuration= 20000
    QFactorGrid_base, QFactorGridPrev_base, time_to_train_Qbase=Qlearn_base.prep_QMatrix(trainingDuration, lenOfEpisode, warmstartoff, distribution, initial_budget)#-----------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    save_object(QFactorGrid_base, "QFactorGrid_base2000_log.pkl")
    
    #QFactorGrid_base=load_object("QFactorGrid_base2000_log.pkl")    
    #******************************************    
    summary_Qlearn_base=Qlearn_base.simulateQLearn(numOfEpisodes, QFactorGrid_base, lenOfEpisode, scarcitypricecoeff, scarce_ingredients, distribution, rng2, initial_budget)
    # Save
    save_object(summary_Qlearn_base, "summary_Qlearn_base2000_log.pkl")
    # Load
    #summary_Qlearn_base=load_object("summary_Qlearn_base2000_log.pkl")
    #summary_Qlearn_base10x=load_object("summary_Qlearn_base20000_log.pkl")
    #summary_Qlearn_base=load_object("summary_Qlearn_base2000_log_cris.pkl")
    #summary_Qlearn_base10x=load_object("summary_Qlearn_base20000_log_cris.pkl")

    ################# RESULTS Baseline ########################################################################################
    
    pdb.set_trace()   
    
    plots.merge_summaries(summary_Qlearn, summary_RollMC, summary_RollMCReds, summary_Heuristic, summary_Qlearn_base, summary_Qlearn_reds, summary_Qlearn_base10x)
    plots.radar_preprint(summary_Qlearn, summary_RollMC, summary_RollMCReds, summary_Heuristic, summary_Qlearn_base, summary_Qlearn_reds, summary_Qlearn_base10x)    
    plots.plot_bubble(summary_Qlearn, summary_RollMC, summary_RollMCReds, summary_Heuristic, summary_Qlearn_base, summary_Qlearn_reds, summary_Qlearn_base10x)
    plots.plot_box(summary_Qlearn, summary_RollMC, summary_RollMCReds, summary_Heuristic, summary_Qlearn_base, summary_Qlearn_reds, summary_Qlearn_base10x)
    #print(time_to_train, time_to_RollMC, time_to_RollOpt)
    #save_object([time_to_RollMC, time_to_Heuristic], "time_base.pkl") 
    plots.plot_box(summary_Qlearn, summary_RollMC, summary_RollMCReds, summary_Heuristic, summary_Qlearn_base, summary_Qlearn_reds, summary_Qlearn_base10x)



 



