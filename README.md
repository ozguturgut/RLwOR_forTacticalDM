From https://doi.org/10.1007/s10479-026-07164-3

"Maintest.py" has the control panel, i.e. "__main__" function.
Default version of this file has all the algorithm calls  (including benchmarks); these lines are open (i.e.uncommented), and allows you call them in the existing order of "__main__". 
If you want to read and compare existing output files w/out running the algorithms, open the summary reading lines which are under the "# Load" lines of each algorithm block; and comment out the rest until "RESULTS Baseline" part.

Following .py files are imported for each method:
-import QlearnExp_v14_func_SimOpt as Qlearn
-import QlearnExp_v14_func_SimOptRedS as QlearnRedS
-import Rollout_v4_MC_func_OptRoll_par_rest as RolloutMCrest
-import Rollout_v4_MC_func_OptRoll_par as RolloutMC
-import kitchenheuristic as Heur
-import QlearnExp_v14_func_Greed as Qlearn_base


INPUT FILES: Recipes.csv, QFactorwarmstart.csv, Pricesv1.csv, Nutrition.csv


All of the files use "optimise_func_v3_prev_v6.py" to call the approximated optimization results. 
Currently "CPLEX" is the default solver. If one wants to use different solvers, this file after line 215 can be modified. (Options for Gurobi is available as commented out)

"Analyze.py" contains plotting functions.

Existing picled (.pkl) files are the original results of "lognormal" price tests.
