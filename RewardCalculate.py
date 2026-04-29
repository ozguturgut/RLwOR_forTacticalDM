
import pdb

def rewardcalculate(k: int,
Recipes:dict, 
Prices:dict,
scenunused:dict, 
term_volatility: float,
scarce_ingredients: list[int],
num_ingredients:float,
usagetime:float, max_pack_size:float, scarcitypricecoeff:float) :
    shop_list=Recipes[k]    
    qntty_ordered={}
    shopping_cost=0
    dailywasted=0 #checks expiration only for the ingredients those are inside the shopping list
    #pdb.set_trace()
    for ingre in range(1,num_ingredients+1):
        requirement=int(shop_list[ingre-1])
        
        if requirement>0:
            useabletotal=0
            for age in range(1,usagetime+1):
                useabletotal=useabletotal+scenunused[ingre,age]
            
            if requirement-useabletotal>0:
                qntty_ordered[ingre]=max_pack_size  #a whole tradeble package consists of 4 portions
                scenunused[ingre,1]=qntty_ordered[ingre]
            
            for age in range(usagetime,0,-1): #usage FIFO (oldest ingredient is used first)
                if scenunused[ingre,age]-requirement>0:
                    scenunused[ingre,age]=scenunused[ingre,age]-requirement
                    requirement=0
                    if age==usagetime:
                        dailywasted=dailywasted+scenunused[ingre,age]
                        scenunused[ingre,age]=0
                elif scenunused[ingre,age]>0 and requirement>0:
                    requirement=requirement-scenunused[ingre,age]
                    scenunused[ingre,age]=0
                if requirement==0:
                    break        
    for ingre in range(1,num_ingredients+1):
        if int(shop_list[ingre-1])>0:
            if (ingre not in scarce_ingredients) :  
                recent_price=Prices[int(shop_list[ingre-1])]*term_volatility
            else: # scarcity price is applied only after 12th day
                recent_price=Prices[int(shop_list[ingre-1])]*scarcitypricecoeff
            if ingre in qntty_ordered:
                shopping_cost=shopping_cost+recent_price*qntty_ordered[ingre]
    return [shopping_cost, scenunused]