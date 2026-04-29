from __future__ import division
from concurrent.futures import ProcessPoolExecutor
import os
import psutil
import csv
import numpy as np
import seaborn as sns
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import random
from itertools import permutations
import pickle
import pdb
import time
import copy
import optimise_func_v3_prev_v6 as opt  #_crisis
import RewardCalculate as rc
from typing import Any

def save_object(obj: Any, filename: str) -> None:
    """Save Python object to file (binary)."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def parallel_control_check(kCtG, i, unused, last3days, remaining_budget, num_MCepisodes, Recipes, Prices, NutritionalValue, xfixes,
    rng1, scarce_ingredients, num_ingredients, usagetime, max_pack_size, scarcitypricecoeff, lenOfEpisode, menus_to_choose,  distribution):
    #for k in menus_to_choose: #possible parallelization   - range(1,num_menuoptions+1)
    #check for any other constraints and eliminate any violating option here
    debug_mode=False         
    skipmenu=False
    for kk in range(len(last3days)):
        if last3days[kk]==kCtG :
            skipmenu=True
            break  
    if(skipmenu):
        #print("menu skipped:", kCtG)
        return kCtG, 10e10, 10e10, 0, unused, 10e10, 10e10
    
    last3daysMC=[]
    for g in range(len(last3days)): last3daysMC.append(last3days[g])
         
    immediate_control_unused={}
    average_cost=0
    average_nutrition=0
    average_wasted=0
    shopping_costk=0   
    scen_unused={}
    immediate_control_unused=copy.deepcopy(unused) #unused[ingre,age] is a global variable and holds the stockages of all ingredients at the start of i-th day
    #scen_unused[k] is a dictionary of dictionaries, which holds the unused info for each menu option, i.e. control
    #################### IMMEDIATE VALUE (cost,waste,nutrition) of the Recipes[k] ########################################################################################

    term_volatility=1.0
    [shopping_costk, immediate_control_unused]=rc.rewardcalculate(kCtG, Recipes, Prices, immediate_control_unused, term_volatility, scarce_ingredients, num_ingredients, usagetime, max_pack_size, scarcitypricecoeff)
    remaining_budgetRoll=remaining_budget-shopping_costk
    if remaining_budgetRoll<=0:
        #print("ran out of budget:", kCtG)
        return kCtG, 10e10, 10e10, 0, unused, 10e10, 10e10
    scen_unused=copy.deepcopy(immediate_control_unused) #scen_unused[period] is a dict and holds the stockages of all ingredients for candidate control k
    
    dailywastedkCost=0
    dailywastedkQuan=0
    for ingre in range(1,num_ingredients+1):
        for age in range(usagetime,0,-1):
            if age==usagetime:
                dailywastedkCost=dailywastedkCost+(immediate_control_unused[ingre,age]*Prices[ingre])
                dailywastedkQuan=dailywastedkQuan+immediate_control_unused[ingre,age]
                immediate_control_unused[ingre,age]=immediate_control_unused[ingre,age-1]
            elif age>1:
                immediate_control_unused[ingre,age]=immediate_control_unused[ingre,age-1]
            else:
                immediate_control_unused[ingre,age]=0


    valid_simulation_count=0
    ################# COST TO GO ########################################################################################
    # since "price" is stochastic, Bellman equation is wrapped in an expectation term, and MONTECARLO is used to approximate it below!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Recipe_dict={}
    num_menuoptions=len(Recipes)
    for menu in range(1, num_menuoptions + 1):
        ing_count=0
        for ing in range(1,num_ingredients+1):
            Recipe_dict[menu,ing]=Recipes[menu][ing_count]
            ing_count=ing_count+1
    for j in range(num_MCepisodes): #further parallelization is possible
        last3daysMC.append(kCtG)
        if len(last3daysMC) > 3:   del last3daysMC[:-3]                

        remaining_budgetRoll=remaining_budget-shopping_costk        
        valid_simulation_count=valid_simulation_count+1      
        Prices_dict={}

        for ing in range(1,num_ingredients+1):
            num_packsizeoptions=1
            for psize in range(1,num_packsizeoptions+1):
                if distribution=="uniform": currentprice=rng1.uniform(low=0.9, high=1.3)*Prices[ing] #uniform
                elif distribution=="normal":currentprice=rng1.normal(loc=Prices[ing], scale=0.1*Prices[ing]) #normal
                else:
                    mu = np.log(Prices[ing]**2 / np.sqrt(Prices[ing]**2 + (0.1*Prices[ing])**2))
                    sigma = np.sqrt(np.log(1 + ((0.1*Prices[ing])**2 / Prices[ing]**2)))
                    currentprice = rng1.lognormal(mean=mu, sigma=sigma, size=1)                   
                Prices_dict[ing,psize]=currentprice
        xfixes[kCtG,i]=1
        #check the cost-to-go starting from i+1 according to the base policy*************************************************************************************
        if i<lenOfEpisode-1:
            [temp_costTOgo,temp_nutriTOgo,temp_wasteTOgo,MMpolicy_action,num_infeas]=opt.deterministic_equival(i+1, xfixes, Prices_dict , Recipe_dict, NutritionalValue, num_packsizeoptions, lenOfEpisode, unused, scarcitypricecoeff, scarce_ingredients) 
            
            average_cost=average_cost+temp_costTOgo
            average_nutrition=average_nutrition+temp_nutriTOgo 
            average_wasted=average_wasted+temp_wasteTOgo 
            #print("**************************************************************************************")
            #print("day:",i," control:",kCtG," with shoppingcost:",temp_costTOgo)
            #print("nutrition:",temp_nutriTOgo)  
            #print("waste amounth:",temp_wasteTOgo)  
            #print("**************************************************************************************")   

    if valid_simulation_count>0:
        average_cost=average_cost/(valid_simulation_count)
        average_nutrition=average_nutrition/(valid_simulation_count) 
        average_wasted=average_wasted/(valid_simulation_count)
    else:
        average_cost=1e10
        average_nutrition=0
        average_wasted=1e10
    if debug_mode:
        print("day:",i,"control:",kCtG," avecosts:",average_cost)
        print("day:",i,"control:",kCtG," avenutrition intake:",average_nutrition)  
        print("day:",i,"control:",kCtG," avewasted:",average_wasted) 
    average_cost=average_cost+shopping_costk
    average_nutrition=average_nutrition+NutritionalValue[kCtG]
    average_wasted=average_wasted+dailywastedkCost           
    del last3daysMC
    del immediate_control_unused
    
    return kCtG, average_cost, average_wasted, average_nutrition, scen_unused, shopping_costk, dailywastedkQuan                             
     
def test_rollout_MC(numOfEpisodes, lenOfEpisode, num_MCepisodes, num_ingredients, usagetime, rng2, distribution, 
num_menuoptions, Recipes, Prices, NutritionalValue, scarce_ingredients, max_pack_size, max_waste, most_exp_menu, initial_budget, menus_to_choose, scarcitypricecoeff):
  
    episode_summary={}
    episode_piece_summary={}    
    scarcitypricecoefforiginal=scarcitypricecoeff
    random.seed(42)
    debug_mode=False
    last3days=[]
    start_time=time.time()
    unused={}
    rng1 = np.random.default_rng(seed=39)
    failed=0
    completion_time=0
    
    
    for episode in range(0, numOfEpisodes):
        for ingre in range(1,num_ingredients+1):
            for age in range(1, usagetime+1):
                unused[ingre,age]=0 # quantity and age of unused inredient
        
        monthly_selection=[]
        monthly_selection_details=[]
        reward_details={}
        remaining_budget=initial_budget
        total_cost=0
        total_nutrition=0
        total_waste=0
        episode_summary[episode]=[]
        episode_piece_summary[episode]=[]
        xfixes={}
        ave_decision_time=0
        dailywastedk=0
        ################# Beginning of new day ########################################################################################
        for i in range(lenOfEpisode): 
            #update stockages  at the beginning of the day
            if i>0:
                for ingre in range(1,num_ingredients+1):
                    for age in range(usagetime,0,-1):
                        if age==usagetime:
                            unused[ingre,age]=unused[ingre,age-1]
                        elif age>1:
                            unused[ingre,age]=unused[ingre,age-1]
                        else:
                            unused[ingre,age]=0
            
            if i <4: scarcitypricecoeff=1
            else: scarcitypricecoeff=scarcitypricecoefforiginal   
            
            control_averages={}
            control_cost={}
            control_wasted={}
            shopping_cost={}
            
            
            #if debug_mode:
                #print("term volatility:",term_volatility)
                #pdb.set_trace()
            #identify all posiible controls, i.e. menu options !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            recent_Prices={}
            for ingre in range(1,num_ingredients+1):
                #if (ingre not in scarce_ingredients) or (step<4):
                if distribution=="uniform": currentprice=rng2.uniform(low=0.9, high=1.3)*Prices[ingre] #uniform
                elif distribution=="normal":currentprice=rng2.normal(loc=Prices[ingre], scale=0.1*Prices[ingre]) #normal
                else:
                    mu = np.log(Prices[ingre]**2 / np.sqrt(Prices[ingre]**2 + (0.1*Prices[ingre])**2))
                    sigma = np.sqrt(np.log(1 + ((0.1*Prices[ingre])**2 / Prices[ingre]**2)))
                    currentprice = rng2.lognormal(mean=mu, sigma=sigma, size=1)                    
                recent_price=currentprice
                if (ingre in scarce_ingredients) and (i>=4):
                    recent_price=recent_price*scarcitypricecoeff   
                recent_Prices[ingre]=recent_price        
            reward_details[i]={}
            ####################### Parallel 
            ################################################################################ 
            results = []
            scen_unused_main={}
            restrictedoptions=[3,4,6,7,13,16,23,24,25,26,28,29,30,42,43]
            # Detect CPU cores automatically for parallel workers
            max_workers = os.cpu_count()  # use all available cores
            print(f"Detected {max_workers} CPU cores")
            print(f"Initial CPU Utilization: {psutil.cpu_percent(interval=1)}%")
            start_time_decision=time.time()
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(parallel_control_check, k, i, unused, last3days, remaining_budget, num_MCepisodes, Recipes, recent_Prices, NutritionalValue, xfixes,
                                    rng1, scarce_ingredients, num_ingredients, usagetime, max_pack_size, scarcitypricecoeff, lenOfEpisode, menus_to_choose, distribution) for k in restrictedoptions]
                #pdb.set_trace()
                for f in futures:
                    results.append(f.result())
                    #print(f"Current CPU Utilization: {psutil.cpu_percent(interval=0.5)}%")
            
            # Merge results back
            #pdb.set_trace()
            for k, cost, wasted, nutri, sunused, shopping_costk, dailywastedk in results:
                reward=(wasted-(nutri/10)+cost)*1000
                reward_details[i][k]=(cost,wasted,nutri)
                control_cost[k]=cost
                control_wasted[k]=dailywastedk
                control_averages[k]=reward
                scen_unused_main[k]=sunused
                shopping_cost[k]=shopping_costk
            
            #find the minimum of control_averages for day i *********************************************************************************************
            selected_menu = min(control_averages, key=control_averages.get)
            ave_decision_time=ave_decision_time+(time.time()-start_time_decision)
            #selected_menu=selected_menu+1
            monthly_selection.append(selected_menu)
            monthly_selection_details.append(reward_details[i][selected_menu])
            unused=copy.deepcopy(scen_unused_main[selected_menu]) 
            total_nutrition=total_nutrition+NutritionalValue[selected_menu]
            total_waste=total_waste+control_wasted[selected_menu]
            remaining_budget=remaining_budget-shopping_cost[selected_menu]
            total_cost=total_cost+shopping_cost[selected_menu]
            xfixes[selected_menu,i]=1
            print("**************************************************************************************")
            print("episode:",episode,"day:",i," selected menu:",selected_menu," with shoppingcost:",shopping_cost[selected_menu], " waste amounth:",control_wasted[selected_menu]," nutrition:",NutritionalValue[selected_menu])
            if debug_mode:
                print("**************************************************************************************")
                print("episode:",episode,"day:",i," selected menu:",selected_menu," with shoppingcost:",control_cost[selected_menu])
                print("nutrition:",NutritionalValue[selected_menu]," remaining budget:",remaining_budget)  
                print("waste amounth:",control_wasted[selected_menu])
                pdb.set_trace()
            
            #Prep next day environment (remaining budget, stockages)*********************************************************************************************
            
            last3days.append(selected_menu)
            if len(last3days) > 3:   del last3days[:-3]

            #################TERMINATION check########################################################################################
            if (remaining_budget<0): 
                failed=failed+1
                print("generated control sequence failed. Total:", failed)
                break
            ave_decision_time=ave_decision_time/lenOfEpisode
            completion_time=completion_time+ave_decision_time
                       
        print("*************************** Episode ****************************: ", episode)
        print(" with averagecost:",str(total_cost/lenOfEpisode), " waste:",str(total_waste/lenOfEpisode)," nutrition:",str(total_nutrition/lenOfEpisode))
        #pdb.set_trace()
        #episode_summary[episode].append(monthly_selection)
        #episode_summary[episode].append(reward_details)
        episode_summary[episode].append(total_cost/lenOfEpisode)
        episode_summary[episode].append(total_waste/lenOfEpisode)
        episode_summary[episode].append(total_nutrition/lenOfEpisode) 
        save_object(episode_summary, "summary_RollMC_log_rest.pkl")
        '''    
        if episode % 5 == 0:
            count=episode // 5
            file_name="summary_RollMC_base_p"+str(count)+".pkl"
            save_object(episode_piece_summary, file_name)
            episode_piece_summary={}
        '''
    completion_time=completion_time/numOfEpisodes
    return episode_summary, completion_time


