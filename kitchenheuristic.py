from __future__ import division
from pyomo.environ import *
import csv
import numpy as np
import RewardCalculate as rc
import random
import time
import pdb

debug_mode=True

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


QFactorGrid={}  #Budget grid no,Day,Action(menu selected):Value, actual remaining budget
############################################# Read remaining budget as grids #############################################
#lenOfEpisode=30
ϵ_init=0.9
ϵ_stable=0.05
cheapest_menu=8
most_exp_menu=41 #prices are given for pacekage size
max_pack_size=4
usagetime=3
scarcitypricecoeff=3
num_packsizeoptions=1


############################################################################################################################
############################################# Auxilary-Performance Assessment #############################################
############################################################################################################################
def test_Heuristic(num_episodes, lenOfEpisode, term_volatility_list, scarcitypricecoeff, scarce_ingredients, distribution, rng, initial_budget):
    global num_MCepisodes
    global total_nutrition, total_waste, total_cost
    global cheapest_menu, most_exp_menu
    global max_pack_size, usagetime
    global numberOfIntervals, menu_with_max_ingredients, max_waste
    global debug_mode, Recipes, num_menuoptions, Prices, num_ingredients, NutritionalValue

    remaining_budget=0
    #pdb.set_trace()
    episode_summary={}
    start_time=time.time()
    for episode in range(0,num_episodes):
        episode_summary[episode]=[]
        proposed_menu=[]
        random_menu=[]        
        gridno=initial_budget//(cheapest_menu)
        current_state={}
        #budget grid, day, stockage grid of ingredients (with the same index order of input)
        
        remaining_budget=initial_budget
        gridno=int(initial_budget//(cheapest_menu))
        current_state=(gridno,)
        #budget grid, day, stockage grid of ingredients (with the same index order of input)
        for i in range(1,num_ingredients+1):
            current_state=current_state+(0,)

        Total_wasted=0
        Total_spent=0
        Total_nutrition=0
        unused={}
        shopping_cost=0
        for ingre in range(1,num_ingredients+1):
            for age in range(1, usagetime+1):
                unused[ingre,age]=0 # quantity and age of unused inredient

        remaining_budget=remaining_budget-Total_spent
          
        last3days=[]
        for step in range(1,lenOfEpisode+1):

            for ingre in range(1,num_ingredients+1):
                for age in range(usagetime,0,-1):
                    if age>1:
                        unused[ingre,age]=unused[ingre,age-1]
                    else:
                        unused[ingre,age]=0
                     
            #################NEXT ACTION########################################################################################
            #find the most expensive ingredient in the stock; find the menus that use it and not same as last 2 days (satisficing)
            #find the least cost among them, break ties according to nutrition (lexicographic)            
            #################NEXT ACTION########################################################################################
            selected_menu=0
            stock_price={}
            menu_shopping_cost={}
            menu_nutrition={}
            ranked_keys_ing=[]
            ranked_menus_cost=[] 
            biggest_ingre_index=0       
            for ingre in range(1,num_ingredients+1):
                stock_value=0
                for age in range(usagetime,0,-1):
                    stock_value=stock_value+unused[ingre,age]*Prices[ingre]                
                stock_price[ingre]=stock_value
            # Extract only the keys in ranked order
            ranked_keys_ing = [k for k, v in sorted(stock_price.items(), key=lambda x: x[1])]
            biggest_ingre_index=len(ranked_keys_ing)
            # find the menus that use it and not same as last 2 days, find the cost and nutrition for them
            if stock_price[ranked_keys_ing[biggest_ingre_index-1]]>0:
                for stock in range(len(ranked_keys_ing)-1, -1, -1):
                    valuable_ingredient=ranked_keys_ing[stock]
                    if stock_price[valuable_ingredient]==0: continue   
                    for menu in range(1, num_menuoptions + 1):      
                        shopping_cost=10e10
                        for ing in range(1,num_ingredients+1):
                            if ing==valuable_ingredient:
                                if (int(Recipes[menu][ing-1])>0) and (menu not in last3days):
                                    term_volatility=1.0
                                    [shopping_cost, immediate_control_unused]=rc.rewardcalculate(menu, Recipes, Prices, unused, term_volatility, scarce_ingredients, num_ingredients, usagetime, max_pack_size, scarcitypricecoeff)
                                    menu_shopping_cost[menu]=shopping_cost
                                    menu_nutrition[menu]=NutritionalValue[menu]                                    
                                    break
                    if len(menu_shopping_cost)>0:
                        break
            if len(menu_shopping_cost)>0:
                # Extract only the menu(key) in ascending shopping cost
                ranked_menus_cost = [k for k, v in sorted(menu_shopping_cost.items(), key=lambda x: x[1])]
                if ranked_menus_cost[0]>0: 
                    if len(ranked_menus_cost)>1: 
                        if ranked_menus_cost[0]==ranked_menus_cost[1]:  #check if there is a tie
                            if menu_nutrition[ranked_menus_cost[0]]>menu_nutrition[ranked_menus_cost[0]]: selected_menu=ranked_menus_cost[0]
                            else:selected_menu=ranked_menus_cost[1]
                        else: selected_menu=ranked_menus_cost[0]

            # if could not select a menu using heuristic, select randomly
            if selected_menu==0:  
                selected_menu=random.randint(1,num_menuoptions)
                random_menu.append((episode,step))
            last3days.append(selected_menu)
            if len(last3days) > 3:   del last3days[:-3]
            
            if debug_mode:
                print("selected menu:",selected_menu)
                print("Day:",step)
                print("Random selections so far: ", random_menu)
                #pdb.set_trace()

            ##########collect information from ENVIRONMENT, price and scarcity of inredients#############################################            
            shop_list=Recipes[selected_menu]
            shopping_cost=0

            for ingre in range(1,num_ingredients+1):
                requirement=int(shop_list[ingre-1])
                while requirement>0:
                    useabletotal=0
                    for age in range(1,usagetime+1):
                        useabletotal=useabletotal+unused[ingre,age]

                    if requirement-useabletotal>0:
                        qntty_ordered=max_pack_size  #a whole tradeble package consists of 4 portions
                        unused[ingre,1]=qntty_ordered
                        #if (ingre not in scarce_ingredients) or (step<4):
                        if distribution=="uniform": currentprice=rng.uniform(low=0.9, high=1.3)*Prices[ingre] #uniform
                        elif distribution=="normal":currentprice=rng.normal(loc=Prices[ingre], scale=0.1*Prices[ingre]) #normal
                        else:
                            mu = np.log(Prices[ingre]**2 / np.sqrt(Prices[ingre]**2 + (0.1*Prices[ingre])**2))
                            sigma = np.sqrt(np.log(1 + ((0.1*Prices[ingre])**2 / Prices[ingre]**2)))
                            currentprice = rng.lognormal(mean=mu, sigma=sigma, size=1)                            
                        recent_price=currentprice
                        if (ingre in scarce_ingredients) and (step>=4):
                            recent_price=recent_price*scarcitypricecoeff
                        shopping_cost=shopping_cost+recent_price*qntty_ordered

                    for age in range(usagetime,0,-1):
                        if unused[ingre,age]-requirement>0:
                            unused[ingre,age]=unused[ingre,age]-requirement
                            requirement=0
                        elif unused[ingre,age]>0 and requirement>0:
                            requirement=requirement-unused[ingre,age]
                            unused[ingre,age]=0
                        if requirement==0:
                            break
                    if requirement<=0:
                        break



            #################ACTUAL VALUE########################################################################################
            Total_spent=Total_spent+shopping_cost
            Total_nutrition=Total_nutrition+NutritionalValue[selected_menu]        
        

        episode_summary[episode].append(Total_spent/lenOfEpisode)
        episode_summary[episode].append(Total_wasted/lenOfEpisode)
        episode_summary[episode].append(Total_nutrition/lenOfEpisode)
        time_to_RollOpt=time.time()-start_time
    return episode_summary, time_to_RollOpt
            

    

