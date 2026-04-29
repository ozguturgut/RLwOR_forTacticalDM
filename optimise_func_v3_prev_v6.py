from __future__ import division
from pyomo.environ import *
import csv
import numpy as np
from numpy import array
from random import random
from itertools import permutations
import pickle
import pdb
import time
import os
from pyomo.opt import TerminationCondition
def deterministic_equival( ttt, xfixes, price_dict, recipe_dict, nutri_dict, num_packsizeoptions, num_DaysToPlan, unused, scarcitypricecoeff, scarce_ingredients) :
    num_DaysToPlan=15
    num_Ingredients=25
    num_MenuItems=50
    max_total_cost=41
    min_cost=0
    max_nutrional_rate=10
    min_nutrition=5
    menu_with_max_ingredients=13
    max_waste=(menu_with_max_ingredients*4)
    min_waste=0
    num_packsizeoptions=1
    usagetime=3
    max_units_required=6
    max_pack_size=4
    discount=0.01
    #print("Start...")

    #MENU ITEMS: 1-Pasta 2-..
    #Breakfast options: 1(Basic):  2(Average):  3(Rich):
    #INGREDIENTS LIST: 1:pasta; 2:tomato; 3:patoto; 4:oil; 5:cheese; 6:chicken;
    #Recipe=[(1,1):2, (1,2):1, (1,3):0, (1,4):1, (1,5):1, (1,6):0, (2,1):1, (2,2):1, (2,3):0, (2,4):1, (2,5):1, (2,6):1, (3,1):0, (3,2):1, (3,3):1, (3,4):1, (3,5):0, (3,6):1], [2, 2, 1, 1, 1, 1], [0, 1, 1, 1, 0, 2]]
    #How much needed from each ingredient for each menu item; four sizes are possible (2 indicating use of half of ordering portion;
    #  4 indicating use of whole  ordering portion)
    #Initial_stock={1:0,2:0,3:0,4:0,5:0,6:0} #for each INGREDIENTS

    model_master = AbstractModel()
    data = DataPortal()

    model_master.MENU = Set(ordered=True,initialize=range(1,num_MenuItems+1))  # index set for first_echelon_routes
    model_master.INGREDIENTS = Set(ordered=True,initialize=range(1,num_Ingredients+1))
    model_master.TIME_DAY = Set(ordered=True,initialize=range(1,num_DaysToPlan+1))
    model_master.PACKSIZE = Set(ordered=True,initialize=range(1,num_packsizeoptions+1))
    #data.load(filename='Sets_ROUTE_VISITS_IC3.tab',format="set", set=model_master.ROUTE_VISITS)
    model_master.Price = Param(model_master.INGREDIENTS, model_master.PACKSIZE,initialize=price_dict, within=Any, mutable=True)
    model_master.Recipe = Param(model_master.MENU, model_master.INGREDIENTS,  initialize=recipe_dict, within=Any, mutable=True)
    model_master.Nutrition = Param(model_master.MENU, initialize=nutri_dict, mutable=True)

    instance = model_master.create_instance(data)
    model_master.LB = Param(initialize=-1*usagetime*max_units_required)
    model_master.UB = Param(initialize=usagetime*max_pack_size)
    model_master.usagetime = Param(initialize=usagetime)
    #model_master.Price = Param(model_master.INGREDIENTS, model_master.PACKSIZE, mutable=True)  # Should be the price for commercial package
    #model_master.Recipe = Param(model_master.MENU,model_master.INGREDIENTS, mutable=True)
    #model_master.Nutrition = Param(model_master.MENU, mutable=True)
    model_master.Initial_stock = Param(model_master.INGREDIENTS,default=0)
    model_master.Psize = Param(model_master.PACKSIZE,default=4) #number of unit production inside the package
    #data.load(filename='RecipesFull.tab', param=model_master.Recipe, format="table")
    #data.load(filename='nutrition.tab', param=model_master.Nutrition, format="table")
    #data.load(filename='price_v1.tab', param=model_master.Price, format="table")


    #pdb.set_trace()
    model_master.x = Var(model_master.MENU,model_master.TIME_DAY,domain=Binary) #selected menu
    model_master.stock = Var(model_master.INGREDIENTS,model_master.TIME_DAY,domain=NonNegativeReals) #,domain=PositiveIntegers
    model_master.order = Var(model_master.INGREDIENTS,model_master.TIME_DAY,model_master.PACKSIZE,domain=NonNegativeReals)
    model_master.dailyneed = Var(model_master.INGREDIENTS,model_master.TIME_DAY,domain=NonNegativeReals)
    model_master.unused = Var(model_master.INGREDIENTS,model_master.TIME_DAY,domain=NonNegativeReals)
    model_master.od = Var(model_master.INGREDIENTS,model_master.TIME_DAY,domain=NonNegativeReals)
    model_master.dn = Var(model_master.INGREDIENTS,model_master.TIME_DAY,domain=NonNegativeReals)
    model_master.d1 = Var(model_master.INGREDIENTS,model_master.TIME_DAY,domain=Binary)
    model_master.d2 = Var(model_master.INGREDIENTS,model_master.TIME_DAY,domain=Binary)


    instance = model_master.create_instance(data)
    ##################-------------Objective------------#############################################
    def obj_expression_cost(model_master):
        return sum((model_master.order[i,t,ps]*model_master.Price[i,ps]) for i in model_master.INGREDIENTS  for t in model_master.TIME_DAY for ps in model_master.PACKSIZE)

    def obj_expression_waste(model_master):
        return sum((model_master.unused[i,t]) for i in model_master.INGREDIENTS  for t in model_master.TIME_DAY)

    def obj_expression_nutrition(model_master):
        return sum((model_master.x[m,t]*-1*model_master.Nutrition[m]) for m in model_master.MENU  for t in model_master.TIME_DAY)

    def obj_expression_multi(model_master):
        return sum(((1-(1/pow(1+discount,t)))*(model_master.order[i,t,ps]*model_master.Price[i,ps]+(model_master.unused[i,t]*model_master.Price[i,ps]))) for i in model_master.INGREDIENTS  for t in model_master.TIME_DAY for ps in model_master.PACKSIZE)+ \
            sum(((model_master.x[m,t]*-1*model_master.Nutrition[m]))/(10) for m in model_master.MENU  for t in model_master.TIME_DAY)
        #return sum((((1-(1/pow(1+discount,t)))*model_master.order[i,t,ps]*model_master.Price[i,ps])-min_cost)/(max_total_cost-min_cost)+((model_master.unused[i,t]-min_waste)/(max_waste-min_waste)) for i in model_master.INGREDIENTS  for t in model_master.TIME_DAY for ps in model_master.PACKSIZE)+ \
        #   sum(((model_master.x[m,t]*-1*model_master.Nutrition[m])-min_nutrition)/(max_nutrional_rate-min_nutrition) for m in model_master.MENU  for t in model_master.TIME_DAY)
    model_master.OBJ = Objective(rule=obj_expression_multi, sense=minimize)
    ##################--------------------------------------------------------#############################################

    def calculate_daily_need(model_master, i,t):
        return sum(model_master.x[m,t] * model_master.Recipe[m,i] for m in model_master.MENU)-model_master.dailyneed[i,t] == 0
    model_master.calculate_daily_need = Constraint(model_master.INGREDIENTS,model_master.TIME_DAY, rule=calculate_daily_need)

    # goes back in the "model_master.usagetime" many days
    def stock_definition(model_master,i,t):
        if t==1:
            return model_master.stock[i,1] == model_master.Initial_stock[i]
        elif t==2:
            return sum((sum(model_master.Psize[ps]*model_master.order[i,tt,ps] for ps in model_master.PACKSIZE)-model_master.dailyneed[i,tt]) for tt in range(t-1,t))+model_master.Initial_stock[i]-model_master.stock[i,t]== 0
        else:
            return sum((sum(model_master.Psize[ps]*model_master.order[i,tt,ps] for ps in model_master.PACKSIZE)-model_master.dailyneed[i,tt]) for tt in range(t-2,t))-model_master.stock[i,t]== 0
    model_master.stock_definition = Constraint(model_master.INGREDIENTS,model_master.TIME_DAY,rule=stock_definition)

    def order_definition(model_master,i,t):
        if t==1:
            return sum(model_master.Psize[ps]*model_master.order[i,t,ps] for ps in model_master.PACKSIZE)-model_master.dailyneed[i,t] +model_master.Initial_stock[i] >= 0
        else:
            return sum(model_master.Psize[ps]*model_master.order[i,t,ps] for ps in model_master.PACKSIZE)-model_master.dailyneed[i,t] +model_master.stock[i,t]>= 0

    model_master.order_definition = Constraint(model_master.INGREDIENTS,model_master.TIME_DAY,rule=order_definition)

    def select_one_menu(model_master, t):
        return sum(model_master.x[m,t] for m in model_master.MENU) <= 1
    model_master.select_one_menu = Constraint(model_master.TIME_DAY, rule=select_one_menu)

    def diversemenufor3days_rule(model_master, m, t):
        if t==1:
            return Constraint.Skip
        elif t==2:
            return sum((model_master.x[m,tt]) for tt in range(t-1,t+1))<= 1
        elif t==3:
            return sum((model_master.x[m,tt]) for tt in range(t-2,t+1))<= 1
        elif t==4:
            return sum((model_master.x[m,tt]) for tt in range(t-3,t+1))<= 1                
        else:
            return sum((model_master.x[m,tt]) for tt in range(t-4,t+1))<= 1

    model_master.diversemenufor3days = Constraint(model_master.MENU, model_master.TIME_DAY, rule=diversemenufor3days_rule)
    ##################------------fix ORDER amount-------------#############################################
    def fix_prev_menu(model_master, m, t):
        if (m,t) in xfixes.keys():
            return model_master.x[m,t] == xfixes[m,t]
        else:
            return Constraint.Skip
    model_master.fix_prev_menu_rule = Constraint(model_master.MENU, model_master.TIME_DAY, rule=fix_prev_menu)
    ##################------------fix STOCK amount-------------#############################################
    def fix_stock(model_master, i, t):
        if (i,t) in unused.keys():
            if unused[i,t]>0: return model_master.stock[i,t] == unused[i,t]  #since "unused" is a dict. that we always created as opposed to "xfixes"
            else: return Constraint.Skip
        else:
            return Constraint.Skip
    model_master.fix_stock_rule = Constraint(model_master.INGREDIENTS,model_master.TIME_DAY, rule=fix_stock)
    ##################------------Constrain budget-------------#############################################
    def restrict_budget(model_master):
        return sum(model_master.Psize[ps]*model_master.order[i,t,ps] for i in model_master.INGREDIENTS for t in model_master.TIME_DAY for ps in model_master.PACKSIZE ) <= 3000
    model_master.restrict_budget_rule = Constraint( rule=restrict_budget)
    
    ##################--------------------------------------------------------#############################################
    ##################--------------To model WASTE amount------------#############################################
    def auxpieceOD_definition(model_master,i,t):
        if t<model_master.usagetime:
            return Constraint.Skip
        else:
            return sum(model_master.Psize[ps]*model_master.order[i,t-2,ps] for ps in model_master.PACKSIZE)- model_master.od[i,t]==0

    model_master.auxpieceOD= Constraint(model_master.INGREDIENTS,model_master.TIME_DAY,rule=auxpieceOD_definition)

    def auxpieceDN_definition(model_master,i,t):
        if t==1:
            return sum((model_master.dailyneed[i,tt]) for tt in range(t,t+1))-model_master.dn[i,t]== 0
        elif t==2:
            return sum((model_master.dailyneed[i,tt]) for tt in range(t-1,t+1))-model_master.dn[i,t]== 0
        else:
            return sum((model_master.dailyneed[i,tt]) for tt in range(t-2,t+1))-model_master.dn[i,t]== 0

    model_master.auxpieceDN = Constraint(model_master.INGREDIENTS,model_master.TIME_DAY,rule=auxpieceDN_definition)

    # rest of the eqn set should start at the "model_master.usagetime"  days
    def unused_lb2(model_master,i,t):
        if t<model_master.usagetime:
            return Constraint.Skip
        else:
            return model_master.od[i,t-2] -model_master.dn[i,t] - model_master.unused[i,t]<=0

    model_master.unused_lb2 = Constraint(model_master.INGREDIENTS,model_master.TIME_DAY,rule=unused_lb2)

    def unused_ub1(model_master,i,t):
        if t<model_master.usagetime:
            return Constraint.Skip
        else:
            return model_master.unused[i,t]-(model_master.UB*(1-model_master.d1[i,t]))<=0

    model_master.unused_ub1 = Constraint(model_master.INGREDIENTS,model_master.TIME_DAY,rule=unused_ub1)

    def unused_ub2(model_master,i,t):
        if t<model_master.usagetime:
            return Constraint.Skip
        else:
            return model_master.unused[i,t]-(model_master.od[i,t-2] -model_master.dn[i,t])-((model_master.UB-model_master.LB)*(1-model_master.d2[i,t]))<=0

    model_master.unused_ub2 = Constraint(model_master.INGREDIENTS,model_master.TIME_DAY,rule=unused_ub2)

    def select_one(model_master,i,t):
        if t<model_master.usagetime:
            return Constraint.Skip
        else:
            return model_master.d1[i,t]+model_master.d2[i,t]-1 ==0

    model_master.select_one = Constraint(model_master.INGREDIENTS,model_master.TIME_DAY,rule=select_one)
    ##################-------------------Integer variable Bounds-----------------#############################################
    def bound_integer(model_master,i,t,ps):
        return model_master.order[i,t,ps] <= 2
    model_master.bound_integer = Constraint(model_master.INGREDIENTS,model_master.TIME_DAY,model_master.PACKSIZE,rule=bound_integer)
    ##################--------------------------------------------------------#############################################


    #pdb.set_trace()
    instance = model_master.create_instance(data)
    #instance.write('mykitchen.lp', io_options={'symbolic_solver_labels': True})
    opt = SolverFactory("cplex", Verbose=False)
    opt.options['mip_tolerances_mipgap'] = 0.33
    #opt.options['MIPGap'] = 0.05
    #opt.options['TimeLimit'] = 30
    results = opt.solve(instance, tee=False, options_string="timelimit=30") #, options_string="timelimit=30 "ng="timelimit=10")
    #if (results.solver.termination_condition not in [TerminationCondition.optimal, TerminationCondition.feasible]):
    #pdb.set_trace()
    #print("termination condition:", results.solver.termination_condition)
    if results.solver.termination_condition in [TerminationCondition.infeasible]:
        return [10e10, 0, 10e10, 0, 1]
    if results.solver.termination_condition in [
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.maxTimeLimit,   # CBC timelimit reached
        TerminationCondition.maxIterations,
        TerminationCondition.other            # catch odd CBC flags
    ]:
        if (hasattr(results.solution, 'Status') and len(results.solution) > 0): 
            total_spent=0
            total_nutrition=0
            total_wasted=0
            total_spent=0    
            #pdb.set_trace()
            for m in instance.MENU:
                for t in instance.TIME_DAY:
                    var_value=instance.x[m,t]
                    menu_offer=0
                    if var_value.value is not None:
                        total_nutrition=total_nutrition+value(instance.Nutrition[m])*var_value.value
            nonzero_keys = [(m, t) for (m, t) in instance.x if value(instance.x[m, t]) > 1e-6]
            menu_offer = [m for (m, t) in nonzero_keys if t == ttt]   
            if len(menu_offer)==0:  menu_offer=[0]        
            total_spent=0
            for i in instance.INGREDIENTS:
                for t in instance.TIME_DAY:
                    for ps in instance.PACKSIZE:
                        var_value=instance.order[i,t,ps]
                        if var_value.value is not None:
                            total_spent=total_spent+value(instance.Price[i,ps])*var_value.value
                        var_value=instance.unused[i,t]
                        if var_value.value is not None:
                            total_wasted=total_wasted+var_value.value*value(instance.Price[i,ps])
        else:
            return [10e10, 0, 10e10, 0, 0]
    else:
      return [10e10, 0, 10e10, 0, 0]  
    
    #print("Appr. Optimum cost:", total_spent," nutrition:", total_nutrition," waste:", total_wasted)
    return [total_spent, total_nutrition, total_wasted, menu_offer[0], 0]

