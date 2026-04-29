from __future__ import division
from pyomo.environ import *
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
import optimise_func_v3_prev_v6  as opt

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


QFactorGrid={}  #Budget grid no,Day,Action(menu selected):Value, actual remaining budget
QFactorGridVisits={} #will follow QFactorGrid to count the number of visits
QFactorGridPrev={} #will follow QFactorGrid to count the number of visits
############################################# Read remaining budget as grids #############################################
def save_object(obj: Any, filename: str) -> None:
    """Save Python object to file (binary)."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

cheapest_menu=8
most_exp_menu=41 #prices are given for pacekage size
max_pack_size=4
usagetime=3
num_packsizeoptions=1
#prep model_master.Price = Param(model_master.INGREDIENTS, model_master.PACKSIZE, mutable=True) 
#prep model_master.Recipe = Param(model_master.MENU,model_master.INGREDIENTS, mutable=True)
Recipe_dict={}
for menu in range(1, num_menuoptions + 1):
    ing_count=0
    for ing in range(1,num_ingredients+1):
        Recipe_dict[menu,ing]=Recipes[menu][ing_count]
        ing_count=ing_count+1

def prep_QMatrix(num_episodes, lenOfEpisode, warmstartoff, distribution, initial_budget):
    global num_MCepisodes, QFactorGrid, QFactorGridVisits, Recipe_dict
    global total_nutrition, total_waste, total_cost, num_packsizeoptions
    global cheapest_menu, most_exp_menu
    global max_pack_size, usagetime, scarcitypricecoeff
    global numberOfIntervals, menu_with_max_ingredients, max_waste
    global rng
    global debug_mode, Recipes, num_menuoptions, Prices, num_ingredients, NutritionalValue
    start_time=time.time()
    menu_with_max_ingredients=13
    max_waste=(menu_with_max_ingredients*4)  #in units of quarter package

    if debug_mode:
        pdb.set_trace()
    actual_remaining_budget=initial_budget
    warm_up_content=[]
    with open('QFactorwarmstart.csv', mode='r') as QFactorfile:
        csv_reader = csv.reader(QFactorfile)
        line_count = 0
        for row in csv_reader:
            warm_up_content.append(row)

        for line_count in range(len(warm_up_content)):
            stock=0
            temp_list=[]
            if line_count == 0:
                continue
            elif line_count == 1:
                actual_remaining_budget=actual_remaining_budget
                temp_list.append(int(actual_remaining_budget)//(cheapest_menu))
                #temp_list.append(line_count)
                for i in range(3,len(row)):
                    temp_list.append(int(0))
                keyquad=tuple(temp_list)
                QFactorGrid[keyquad]=(float(warm_up_content[line_count][2]),int(warm_up_content[line_count][1]),actual_remaining_budget-int(warm_up_content[line_count][0]))
                #if keyquad not in QFactorGridPrev.keys(): QFactorGridPrev[keyquad]=[-1]
                if keyquad not in QFactorGridVisits.keys(): QFactorGridVisits[keyquad]=0
                QFactorGridVisits[keyquad]=QFactorGridVisits[keyquad]+1
            else:
                actual_remaining_budget=actual_remaining_budget-int(warm_up_content[line_count-1][0])
                temp_list.append(int(actual_remaining_budget)//(cheapest_menu))
                temp_list.append(line_count)
                for i in range(3,len(warm_up_content[line_count-1])):
                    stock=int(warm_up_content[line_count-1][i])
                    temp_list.append(int(stock))
                keyquad=tuple(temp_list)
                #keys:budget grid, day, stockage grid of ingredients (with the same index order of input)
                #Content : value,  action(menu), actual budget remaining                
                QFactorGrid[keyquad]=(float(warm_up_content[line_count][2]),int(warm_up_content[line_count][1]),actual_remaining_budget-int(warm_up_content[line_count][0]))
                #if keyquad not in QFactorGridPrev.keys() : QFactorGridPrev[keyquad]=[-1]
                if keyquad not in QFactorGridVisits.keys(): QFactorGridVisits[keyquad]=0
                QFactorGridVisits[keyquad]=QFactorGridVisits[keyquad]+1


    if warmstartoff: QFactorGrid={} #active if without warm start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ############################################# Algorithm loop #############################################
    def load_object(filename: str) -> Any:
        """Load Python object from file (binary)."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
    rng3=np.random.default_rng(seed=93)       
    '''
    QFactorGridPrev = load_object("QFactorGrid_adv_1500.pkl")
    pdb.set_trace()

    episode_summary = QFactorGridPrev.copy()
    
    for ingre in range(0,len(QFactorGridPrev)*lenOfEpisode):
        #if (ingre not in scarce_ingredients) or (step<4):
        if distribution=="uniform": currentprice=rng3.uniform(low=0.9, high=1.3) #uniform
        elif distribution=="normal":currentprice=rng3.normal(loc=0, scale=0.1) #normal
        else: currentprice=rng3.weibull(0.5)#weibull
    '''


    Total_days=0
    Total_wasted=0
    Total_spent=0
    Total_nutrition=0
    
    num_random_selection=0
    total_infeas=0
    for episode in range(0,num_episodes+1):
        remaining_budget=initial_budget
        gridno=int(initial_budget//(cheapest_menu))
        current_state=(gridno,)
        #budget grid, day, stockage grid of ingredients (with the same index order of input)
        for i in range(1,num_ingredients+1):
            current_state=current_state+(0,)

        MTotal_wasted=0
        MTotal_spent=0
        MTotal_nutrition=0
        #same order as Qfactor matrix-gridno,day,action,actual budget status
        unused={}
        xfixes={}

        for ingre in range(1,num_ingredients+1):
            for age in range(1, usagetime+1):
                unused[ingre,age]=0 # quantity and age of unused inredient

        for step in range(1,lenOfEpisode+1):
            
            Prices_dict={}
            Prices_current={}
            for ing in range(1,num_ingredients+1):
                for psize in range(1,num_packsizeoptions+1):
                    if distribution=="uniform": currentprice=rng3.uniform(low=0.9, high=1.3)*Prices[ing] #uniform
                    elif distribution=="normal":currentprice=rng3.normal(loc=Prices[ing], scale=0.1*Prices[ing]) #normal
                    else: currentprice=rng3.weibull(0.05)*Prices[ing] #weibull
                    Prices_dict[ing,psize]=currentprice
                    Prices_current[ing]=currentprice
            for ingre in range(1,num_ingredients+1):
                for age in range(usagetime,0,-1):
                    if age>1:
                        unused[ingre,age]=unused[ingre,age-1]
                    else:
                        unused[ingre,age]=0
            if debug_mode:
                print("at the beginning of day ",step, " current_state:",current_state)
                print("----------------------------------------------------")
                #print("at the beginning of day ",step, " QFactorGrid:",QFactorGrid)
                #pdb.set_trace()
            #################NEXT ACTION########################################################################################
            used_Qfactors=False
            selected_menu=0
            #rnd_for_selection=exploration_rate[(episode-1)*lenOfEpisode+step]  Exploitation is removed            
            scarcitypricecoeff=1
            scarce_ingredients=[]
            [temp_costTOgo,temp_nutriTOgo,temp_wasteTOgo,MMpolicy_action,num_infeas]=opt.deterministic_equival(step, xfixes, Prices_dict , Recipe_dict, NutritionalValue, num_packsizeoptions, lenOfEpisode, unused, scarcitypricecoeff, scarce_ingredients)                    
            #pdb.set_trace()
            total_infeas=total_infeas+num_infeas
            selected_menu=MMpolicy_action

            #if opt did not return solution then allow it to enter to the random selection/exploration part below
            if (selected_menu==0):
                selected_menu=random.randint(1, num_menuoptions)
                num_random_selection=num_random_selection+1
                print("************************************************************")
                print("at the beginning of day ",step, " episode",episode,"RANDOM SELECTED")
                print("***************************************************************")                
            if debug_mode:
                print("episode:", episode, " Day:",step, " selected menu:",selected_menu)

            ##########collect information from ENVIRONMENT, price and scarcity of inredients#############################################      
            
            shop_list=Recipes[selected_menu]
            shopping_cost=0
            dailywasted=0
            #pdb.set_trace()
            for ingre in range(1,num_ingredients+1):
                requirement=int(shop_list[ingre-1])
                recent_price=Prices_current[ingre]
                while requirement>0:
                    useabletotal=0
                    for age in range(1,usagetime+1):
                        useabletotal=useabletotal+unused[ingre,age]

                    if requirement-useabletotal>0:
                        qntty_ordered=max_pack_size  #a whole tradeble package consists of 4 portions
                        unused[ingre,1]=qntty_ordered                        
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
                
                for age in range(1,usagetime+1):
                    if (age==usagetime):
                        dailywasted=dailywasted+unused[i,age]*recent_price

            #################ACTUAL VALUE########################################################################################
            reward=((dailywasted)+1-((NutritionalValue[selected_menu])/10)+((shopping_cost)))*1000
            #reward=(((shopping_cost)/(most_exp_menu)))*1000
            #################NEXT STATE using transition########################################################################################
            #if debug_mode:
            #    pdb.set_trace()
            temp_list=[]
            temp_list.append(int((remaining_budget-shopping_cost)//cheapest_menu))
            #temp_list.append(current_state[1]+1)
            remaining_budget=remaining_budget-shopping_cost
            for i in range(1,num_ingredients+1):
                stock=0
                for age in range(1,usagetime):
                    stock=stock+unused[i,age]
                stock=int(stock)
                temp_list.append(stock)
            new_state=tuple(temp_list)
            
            ## find the min(best) possible value for the new state-QFactorGrid[(new_state[0],new_state[1],*)]
            min_newstate=1000000
            best_action=-10
            
            for keys in QFactorGrid.keys():
                if keys==new_state:
                    if float(QFactorGrid[keys][0])<min_newstate:
                        min_newstate=float(QFactorGrid[keys][0])
                        best_action=int(QFactorGrid[keys][1])
                        
            
            #################TERMINATION check########################################################################################
            if remaining_budget<=0:
                QFactorGrid[current_state][0]=[1000000,selected_menu,0]
                #if (current_state not in QFactorGridPrev.keys()) and episode<11: QFactorGridPrev[current_state]=[-1*episode]
                if current_state not in QFactorGridVisits.keys(): QFactorGridVisits[current_state]=0
                QFactorGridVisits[current_state]=QFactorGridVisits[current_state]+1
                break
                #################UPDATE Qfactor table of previous state-action##########################################
            else:
                #alfa=0.05
                alfa_a=0.25
                alfa_b=10
                discount=0.97
                prev_value=0
                if used_Qfactors:
                    #Find the existing QfactorGrid value of the current state
                    for keys in QFactorGrid.keys():
                        if keys==current_state:
                            if QFactorGrid[keys][1]==selected_menu:
                                prev_value=QFactorGrid[keys][0]

                # if not found it is prev_value=0
                if current_state not in QFactorGridVisits.keys(): QFactorGridVisits[current_state]=0
                QFactorGridVisits[current_state]=QFactorGridVisits[current_state]+1
                value=0
                alfa=alfa_a/(alfa_b+QFactorGridVisits[current_state])
                value=prev_value+alfa*(reward+pow(discount,step)*min_newstate-prev_value)
                QFactorGrid[current_state]=[value, selected_menu, remaining_budget]
                #if (current_state not in QFactorGridPrev.keys()) and episode<11: QFactorGridPrev[current_state]=[-1*episode]
                xfixes[selected_menu,step]=1

                if debug_mode:
                    #print("at the end of day ",step, " current_state:",current_state)
                    #print("----------------------------------------------------")
                    print("at the end of day ",step, " new_state:",new_state)
                    print("----------------------------------------------------")
                    #print("at the end of day ",step, " QFactorGrid:",QFactorGrid)
                    #print("----------------------------------------------------")
                current_state=new_state
            ############################################# Auxilary-Daily Values #############################################
            Total_days=Total_days+1
            Total_spent=Total_spent+shopping_cost
            Total_wasted=Total_wasted+dailywasted
            Total_nutrition=Total_nutrition+NutritionalValue[selected_menu]
            MTotal_spent=MTotal_spent+shopping_cost
            MTotal_wasted=MTotal_wasted+dailywasted
            MTotal_nutrition=MTotal_nutrition+NutritionalValue[selected_menu]
            time_to_train=time.time()-start_time
        '''
        if episode % 10 == 0 or episode==num_episodes:
            for key in QFactorGrid:
                if (key not in QFactorGridPrev.keys()): QFactorGridPrev[key]=[-1*episode]
                QFactorGridPrev[key].append(QFactorGrid[key][0])
        '''
        if episode % 10 == 0:
            #print("there are ",len(QFactorGridPrev), " keys")
            print("randomly selected", num_random_selection, " total infeasible:", total_infeas)
            #sorted_QFactorGridVisits = dict(sorted(QFactorGridVisits.items(), key=lambda x: x[1], reverse=True))
            #threshold = 3
            #keys_above_threshold = list(filter(lambda key: sorted_QFactorGridVisits[key] > threshold, sorted_QFactorGridVisits.keys()))
            file_name="QFactorGrid_adv_norm"+str(episode)+".pkl"
            save_object(QFactorGrid, file_name)
            #for key in keys_above_threshold:
            #    print(QFactorGridPrev[key])


    ############################################# Auxilary-Heatmap #############################################

    if debug_mode:
        pdb.set_trace()
    
    valueheatmap=np.random.random((lenOfEpisode+1,num_menuoptions+1))
    for menuid in range(1,num_menuoptions+1):
        for day in range(1,lenOfEpisode+1):
            for key in QFactorGrid:
                if menuid==QFactorGrid[key][1] and day==key[1]:
                    valueheatmap[day][menuid]=valueheatmap[day][menuid]+QFactorGrid[key][0]
    df = pd.DataFrame(valueheatmap) #, columns=["a","b","c","d","e","f","g","h","i","j"]
    sns.heatmap(df, cmap="YlGnBu")
    plt.savefig('QMatrix.png')
                  
    '''
    f = open('results_qlearn.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['Total_days,Total_spent,Total_wasted,Total_nutrition,Daily_spent,Daily_wasted,Daily_nutrition'])
    f.close()
    f2 = open('monthly_qlearn.csv', 'w', newline='')
    writer2 = csv.writer(f2)
    writer2.writerow(['Episode,MonthTotal_spent,MonthTotal_wasted,MonthTotal_nutrition,MonthlyDayAve_spent,MonthlyDayAve_wasted,MonthlyDayAve_nutrition'])
    f2.close()
    '''
    return QFactorGrid, QFactorGridPrev, time_to_train
############################################################################################################################
############################################# Auxilary-Performance Assessment #############################################
############################################################################################################################
def simulateQLearn(num_episodes, QFactorGrid, lenOfEpisode, scarcitypricecoeff, scarce_ingredients, distribution, rng, initial_budget):
    global num_MCepisodes
    global total_nutrition, total_waste, total_cost
    global cheapest_menu, most_exp_menu
    global max_pack_size, usagetime
    global numberOfIntervals, menu_with_max_ingredients, max_waste
    global debug_mode, Recipes, num_menuoptions, Prices, num_ingredients, NutritionalValue
    
    remaining_budget=0
    #pdb.set_trace()
    episode_summary={}
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
        temp_list=[]
        temp_list.append(int((remaining_budget-shopping_cost)//cheapest_menu))
        temp_list.append(current_state[1]+1)
        remaining_budget=remaining_budget-shopping_cost
        for i in range(1,num_ingredients+1):
            stock=0
            for age in range(1,usagetime+1):
                stock=stock+unused[i,age]
            stock=int(stock)
            temp_list.append(stock)
        current_state=tuple(temp_list)           

        for step in range(1,lenOfEpisode+1):

            for ingre in range(1,num_ingredients+1):
                for age in range(usagetime,0,-1):
                    if age>1:
                        unused[ingre,age]=unused[ingre,age-1]
                    else:
                        unused[ingre,age]=0
            if debug_mode:
                print("at the beginning of day ",step, " current_state:", current_state)
                print("----------------------------------------------------")                        
            #################NEXT ACTION########################################################################################
            used_Qfactors=False
            use_best=False
            selected_menu=0
            #################NEXT ACTION########################################################################################
            used_Qfactors=False
            maxvalue=-100000000
            for Qkeys in QFactorGrid:
                if Qkeys==current_state:
                    if float(QFactorGrid[Qkeys][0])>maxvalue:
                        maxvalue=float(QFactorGrid[Qkeys][0])
                        selected_menu=int(QFactorGrid[Qkeys][1])                        
                        print("Found from Qfactor matrix at key: ", Qkeys)
                        used_Qfactors=True
            if used_Qfactors:
                proposed_menu.append(selected_menu)
            #if it is not found in the QFactor then allow it to enter to the random selection part below
            if (not use_best) or (selected_menu==0):
                selected_menu=random.randint(1, num_menuoptions)
                random_menu.append(use_best)
            if debug_mode:
                print("selected menu:",selected_menu)
                print("Day:",step)
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
                        else: currentprice=rng.weibull(0.5)*Prices[ingre] #weibull
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

            #################NEXT STATE using transition########################################################################################
            #if debug_mode:
            #    pdb.set_trace()
            temp_list=[]
            temp_list.append(int((remaining_budget-shopping_cost)//cheapest_menu))
            #temp_list.append(current_state[1]+1)
            remaining_budget=remaining_budget-shopping_cost
            for i in range(1,num_ingredients+1):
                stock=0
                for age in range(1,usagetime+1):
                    stock=stock+unused[i,age]
                    if (age==usagetime):
                        Total_wasted=Total_wasted+unused[i,age]
                stock=int(stock)
                temp_list.append(stock)
            new_state=tuple(temp_list)
            current_state=new_state


            #################ACTUAL VALUE########################################################################################
            Total_spent=Total_spent+shopping_cost
            Total_nutrition=Total_nutrition+NutritionalValue[selected_menu]        
        

        episode_summary[episode].append(Total_spent/lenOfEpisode)
        episode_summary[episode].append(Total_wasted/lenOfEpisode)
        episode_summary[episode].append(Total_nutrition/lenOfEpisode)
    
    return episode_summary
            

    

