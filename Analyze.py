import csv
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pdb
from typing import Any
############################################################################################# SAVE AND LOAD ###################################################################################

def load_object(filename: str) -> Any:
    """Load Python object from file (binary)."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

############################################################################################# PLOT ###################################################################################
def plot_bubble(summary_Qlearn, summary_RollMC, summary_RollMC_reds, summary_Heuristic, summary_Qlearn_base, summary_Qlearn_reds, summary_Qlearn_base10x):
    plt.figure(figsize=(8, 6))

    def avg_extract(summary, label, color):
        costs = []
        wastes = []
        nutritions = []
        for _, vals in summary.items():
            costs.append(vals[0])     # cost
            wastes.append(vals[1])    # waste
            nutritions.append(vals[2])# nutrition
        avg_cost = sum(costs) / len(costs) if costs else 0
        avg_waste = sum(wastes) / len(wastes) if wastes else 0
        avg_nutrition = sum(nutritions) / len(nutritions) if nutritions else 0

        plt.scatter(avg_cost, avg_waste, s=avg_nutrition * 500, alpha=0.5, color=color)
        plt.text(avg_cost, avg_waste, label, fontsize=9, ha="left", va="center")  # <-- add name next to bubble

    avg_extract(summary_Qlearn, "QlearnAdv", "blue")
    avg_extract({k: [v[0], v[1], v[2]] for k, v in summary_RollMC.items()}, "RollMC", "green")
    avg_extract({k: [v[0], v[1], v[2]] for k, v in summary_RollMC_reds.items()}, "RollMC_reds", "pink")
    avg_extract({k: [v[0], v[1], v[2]] for k, v in summary_Heuristic.items()}, "Heuristic", "red")
    avg_extract({k: [v[0], v[1], v[2]] for k, v in summary_Qlearn_base.items()}, "QlearnBase", "orange")
    avg_extract({k: [v[0], v[1], v[2]] for k, v in summary_Qlearn_reds.items()}, "QlearnAdvReds", "purple")
    avg_extract({k: [v[0], v[1], v[2]] for k, v in summary_Qlearn_base10x.items()}, "QlearnBase10x", "brown")

    plt.xlabel("Average Cost per Episode")
    plt.ylabel("Average Waste per Episode")
    plt.title("Average Cost vs Average Waste vs Average Nutrition (Bubble size = Avg Nutrition)")
    # plt.legend()  # <-- removed legend
    plt.savefig('bubble_base.png')

def plot_box(summary_Qlearn, summary_RollMC, summary_RollMC_reds, summary_Heuristic, summary_Qlearn_base, summary_Qlearn_reds, summary_Qlearn_base10x):
    data = []

    def avg_metric(label, cost_idx, waste_idx, nutrition_idx, summary):
        costs = [v[cost_idx] for v in summary.values()]
        wastes = [v[waste_idx] for v in summary.values()]
        nutritions = [v[nutrition_idx] for v in summary.values()]
        data.append([label, "Cost", sum(costs)/len(costs) if costs else 0])
        data.append([label, "Waste", sum(wastes)/len(wastes) if wastes else 0])
        data.append([label, "Nutrition", sum(nutritions)/len(nutritions) if nutritions else 0])

    avg_metric("QlearnAdv", 0, 1, 2, summary_Qlearn)
    avg_metric("RollMC", 0, 1, 2, summary_RollMC)
    avg_metric("RollMC_reds", 0, 1, 2, summary_RollMC_reds)
    avg_metric("Heuristic", 0, 1, 2, summary_Heuristic)
    avg_metric("QlearnBase", 0, 1, 2, summary_Qlearn_base)
    avg_metric("QlearnReds", 0, 1, 2, summary_Qlearn_reds)
    avg_metric("QlearnBase10x", 0, 1, 2, summary_Qlearn_base10x)

    df = pd.DataFrame(data, columns=["Method", "Metric", "Value"])
    plt.figure(figsize=(10, 6))

    ax = sns.boxplot(data=df, x="Metric", y="Value", hue="Method")

    # Remove legend
    if ax.legend_ is not None:
        ax.legend_.remove()

    # Add labels above each box using line collections instead of patches
    metrics = df["Metric"].unique()
    methods = df["Method"].unique()
    
    # Calculate positions for each box
    num_methods = len(methods)
    width = 0.8 / num_methods  # Total width divided by number of methods
    
    for i, metric in enumerate(metrics):
        for j, method in enumerate(methods):
            # Calculate x position for this box
            x = i + (j - num_methods/2 + 0.5) * width
            
            # Get the corresponding value
            value = df[(df["Metric"] == metric) & (df["Method"] == method)]["Value"].values[0]
            
            # Add text label
            ax.text(x, value, method, ha="center", va="bottom", fontsize=8, rotation=0)

    plt.title("Average Performance Comparison Across Methods")
    plt.tight_layout()
    plt.savefig('box_base.png')




def merge_summaries(summary_Qlearn, summary_RollMC, summary_RollMC_reds, summary_Heuristic, summary_Qlearn_base, summary_Qlearn_reds, summary_Qlearn_base10x, filename="summary_all_base.csv"):
    merged_rows = []

    # Qlearn format: [cost, waste, nutrition]
    for ep, vals in summary_Qlearn.items():
        merged_rows.append([ep, "Qlearn", vals[0], vals[1], vals[2]])

    # RollMC format: [monthly_selection, reward_details, cost, waste, nutrition, duration]
    for ep, vals in summary_RollMC.items():
        merged_rows.append([ep, "RollMC", vals[0], vals[1], vals[2]])

    # Heuristic format: same as Heuristic
    for ep, vals in summary_Heuristic.items():
        merged_rows.append([ep, "Heuristic", vals[0], vals[1], vals[2]])
    
    # Heuristic format: same as RollMC reduced control
    for ep, vals in summary_RollMC_reds.items():
        merged_rows.append([ep, "RollMC_reds", vals[0], vals[1], vals[2]])        

    # Qlearn_base format: same as Qlearn Greedy
    for ep, vals in summary_Qlearn_base.items():
        merged_rows.append([ep, "Qlearn_base", vals[0], vals[1], vals[2]])     

    # Qlearn_base format: same as Qlearn reduced states
    for ep, vals in summary_Qlearn_reds.items():
        merged_rows.append([ep, "Qlearn_reds", vals[0], vals[1], vals[2]]) 

    # Qlearn_base10x format: same as Qlearn reduced states
    for ep, vals in summary_Qlearn_base10x.items():
        merged_rows.append([ep, "Qlearn_base10x", vals[0], vals[1], vals[2]]) 
    # Write merged file
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Method", "Cost", "Waste", "Nutrition"])
        writer.writerows(merged_rows)

    print(f"Merged summary saved to {filename}")     

#####################################################################################################################
def dicts_to_matrix(*algos):
    episodes = sorted(algos[0].keys())
    num_algos = len(algos)
    num_criteria = 3  # cost, waste, nutrition
    
    matrix = np.zeros((len(episodes), num_algos, num_criteria))
    
    for e_idx, ep in enumerate(episodes):
        for a_idx, algo in enumerate(algos):
            vals = algo[ep]
            # Flatten each value if it's a list, otherwise use as-is
            for c_idx in range(num_criteria):
                val = vals[c_idx]
                # Unwrap if nested in a list
                matrix[e_idx, a_idx, c_idx] = val[0] if isinstance(val, list) else val
    
    return matrix
# "algo_A, algo_B, algo_C" are dictionaries of lists (keys are episode index)
# Example usage
# matrix = dicts_to_matrix(algo_A, algo_B, algo_C)
# print(matrix.shape)  # (30, 3, 3)
# matrix[e, a, c] = criterion c for algorithm a at episode e

'''
#####################################################################################################################

# -----------------------------
# Example dummy data
# -----------------------------
np.random.seed(42)
algo_A = {i: np.random.rand(3).tolist() for i in range(30)}
algo_B = {i: np.random.rand(3).tolist() for i in range(30)}
algo_C = {i: np.random.rand(3).tolist() for i in range(30)}

'''
def radar_preprint (algo_A, algo_B, algo_C, algo_D, algo_E, algo_F, algo_G):
    algorithms = ["Qlearn_adv", "Rollout", "Rollout_reds", "Heuristic", "Qlearn_base", "Qlearn_reds", "Qlearn_base10x"]
    criteria = ["Cost", "Waste", "Nutrition"]
    # -----------------------------
    # Step 1: Convert dicts to matrix (episodes × algorithms × criteria)
    # -----------------------------
    def dicts_to_matrix(*algos):
        episodes = sorted(algos[0].keys())
        num_algos = len(algos)
        num_criteria = 3  # cost, waste, nutrition
        
        matrix = np.zeros((len(episodes), num_algos, num_criteria))
        
        for e_idx, ep in enumerate(episodes):
            for a_idx, algo in enumerate(algos):
                vals = algo[ep]
                # Flatten each value if it's a list, otherwise use as-is
                for c_idx in range(num_criteria):
                    val = vals[c_idx]
                    # Unwrap if nested in a list
                    matrix[e_idx, a_idx, c_idx] = val[0] if isinstance(val, list) else val
        
        return matrix

    matrix = dicts_to_matrix(algo_A, algo_B, algo_C, algo_D, algo_E, algo_F, algo_G)

    # -----------------------------
    # Step 2: Average over episodes
    # -----------------------------
    avg_matrix = matrix.mean(axis=0)  # (algorithms × criteria)

    # -----------------------------
    # Step 3: Normalize criteria (min–max per column)
    # -----------------------------
    def normalize_matrix(mat, maximize=[True, False, True]):
        normed = np.zeros_like(mat)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            if col.max() > col.min():
                col_norm = (col - col.min()) / (col.max() - col.min())
            else:
                col_norm = np.zeros_like(col)
            # If maximize is True, higher values → higher scores (0 to 1)
            # If maximize is False, lower values → higher scores (flip: 1 - norm)
            if not maximize[j]:
                col_norm = 1 - col_norm
            normed[:, j] = col_norm
        return normed

    # Cost ↓ (lower is better), Waste ↓ (lower is better), Nutrition ↑ (higher is better)
    avg_matrix_norm = normalize_matrix(avg_matrix, maximize=[False, False, True])

    # -----------------------------
    # Step 4a: Radar Plot
    # -----------------------------
    def plot_radar(values, algos, criteria):
        N = len(criteria)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        values = np.concatenate((values, values[:,[0]]), axis=1)  # close loop
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        for i, algo in enumerate(algos):
            ax.plot(angles, values[i], label=algo)
            ax.fill(angles, values[i], alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria)
        ax.set_title("Radar Chart – Average Performance")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.savefig('radar_comparison.png')
        plt.show()

    plot_radar(avg_matrix_norm, algorithms, criteria)

    # -----------------------------
    # Step 4b: Heatmap
    # -----------------------------
    plt.figure(figsize=(6,4))
    sns.heatmap(avg_matrix_norm, annot=True, cmap="YlGnBu",
                xticklabels=criteria, yticklabels=algorithms)
    plt.title("Heatmap – Average Normalized Performance")
    plt.savefig('heatmap_comparison.png')
    plt.show()

    # -----------------------------
    # Step 5: Numeric Summary Table
    # -----------------------------
    df_raw = pd.DataFrame(avg_matrix, index=algorithms, columns=criteria)
    df_norm = pd.DataFrame(avg_matrix_norm, index=algorithms, columns=criteria)

    print("\n=== Raw Average Performance ===")
    print(df_raw.round(3))

    print("\n=== Normalized Average Performance (0=worst, 1=best) ===")
    print(df_norm.round(3))
#######################################################################################Parameter Tuning plots are inside AnalyseParam.py #############################################
def compare_trainDuration():


    def dicts_to_matrix(*algos):
        episodes = sorted(algos[0].keys())
        num_algos = len(algos)
        num_criteria = 3  # cost, waste, nutrition
        
        matrix = np.zeros((len(episodes), num_algos, num_criteria))
        
        for e_idx, ep in enumerate(episodes):
            for a_idx, algo in enumerate(algos):
                vals = algo[ep]
                # Flatten each value if it's a list, otherwise use as-is
                for c_idx in range(num_criteria):
                    val = vals[c_idx]
                    # Unwrap if nested in a list
                    matrix[e_idx, a_idx, c_idx] = val[0] if isinstance(val, list) else val
        
        return matrix

    # --------------------------------
    # Load all algorithms
    # --------------------------------
    algorithms = ["Qlearn_adv_100", "Qlearn_adv_200", "Qlearn_adv_300", "Qlearn_adv_400", "Qlearn_adv_500","Qlearn_adv_600","Qlearn_adv_700","Qlearn_adv_800","Qlearn_adv_900","Qlearn_adv_1000",
        "Qlearn_adv_1100", "Qlearn_adv_1200", "Qlearn_adv_1300", "Qlearn_adv_1400", "Qlearn_adv_1500", "Qlearn_adv_1600", "Qlearn_adv_1700", "Qlearn_adv_1800", "Qlearn_adv_1900", "Qlearn_adv_2000"] #, "Qlearn_adv_2500", "Qlearn_adv_3000"
    #algorithms = ["Qlearn_adv_reds500","Qlearn_adv_reds1000","Qlearn_adv_reds1500", "Qlearn_adv_reds2000"] #, "Qlearn_adv_2500", "Qlearn_adv_3000"
    criteria = ["Cost", "Waste", "Nutrition"]

    Qlearn_adv = {}
    for i, name in enumerate(algorithms, start=1):
        filename="summary_QFactorGrid_adv_log"+str(i*100)+".pkl"
        Qlearn_adv[i] = load_object(filename)   # your function

    # --------------------------------
    # Convert to mean performance table
    # --------------------------------
    matrix = dicts_to_matrix(*Qlearn_adv.values())

    # Average over episodes (axis=0)
    mean_matrix = matrix.mean(axis=0)

    # Build DataFrame: rows=algorithms, cols=criteria
    df_means = pd.DataFrame(mean_matrix, index=algorithms, columns=criteria)

    print(df_means.round(3))


    return df_means

#######################################################################################Parameter Tuning plots are inside AnalyseParam.py #############################################
'''
def plot_lines(summary_Qlearn, summary_RollMC, summary_RollOpt):
    plt.figure(figsize=(8, 6))

    def avg_series(summary, idx):
        return sum([v[idx] for v in summary.values()]) / len(summary) if summary else 0

    episodes = ["Qlearn", "RollMC", "RollOpt"]
    avg_costs = [
        avg_series(summary_Qlearn, 0),
        avg_series(summary_RollMC, 0),
        avg_series(summary_RollOpt, 0)
    ]
    plt.plot(episodes, avg_costs, marker='o', label="Average Cost", color='blue')

    plt.xlabel("Method")
    plt.ylabel("Average Cost")
    plt.title("Average Cost Across Methods")
    plt.legend()
    plt.savefig('line_base.png')
'''    


def merge_summaries2 (*summary_dicts, filename="summary_all_base.csv"):
    """
    Merge multiple summary dictionaries into a single CSV file.
    
    Args:
        *summary_dicts: Tuples of (method_name, summary_dict)
                       e.g., ("Qlearn", summary_Qlearn), ("RollMC", summary_RollMC), ...
        filename: Output CSV filename (keyword argument)
    
    Usage:
        merge_summaries(
            ("Qlearn", summary_Qlearn),
            ("RollMC", summary_RollMC),
            ("Heuristic", summary_Heuristic),
            filename="output.csv"
        )
    """
    merged_rows = []
    
    # Process each (name, dictionary) pair
    for method_name, summary_data in summary_dicts:
        # Each dictionary format: {episode: [cost, waste, nutrition], ...}
        for ep, vals in summary_data.items():
            merged_rows.append([ep, method_name, vals[0], vals[1], vals[2]])
    
    # Write merged file
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Method", "Cost", "Waste", "Nutrition"])
        writer.writerows(merged_rows)
    
    print(f"Merged summary saved to {filename}")

'''
# Example usage with your 20 dictionaries:
merge_summaries(
    ("Qlearn_adv_100", summary_Qlearn_adv_100),
    ("Qlearn_adv_200", summary_Qlearn_adv_200),
    ("Qlearn_adv_300", summary_Qlearn_adv_300),
    ("Qlearn_adv_400", summary_Qlearn_adv_400),
    ("Qlearn_adv_500", summary_Qlearn_adv_500),
    ("Qlearn_adv_600", summary_Qlearn_adv_600),
    ("Qlearn_adv_700", summary_Qlearn_adv_700),
    ("Qlearn_adv_800", summary_Qlearn_adv_800),
    ("Qlearn_adv_900", summary_Qlearn_adv_900),
    ("Qlearn_adv_1000", summary_Qlearn_adv_1000),
    ("Qlearn_adv_1100", summary_Qlearn_adv_1100),
    ("Qlearn_adv_1200", summary_Qlearn_adv_1200),
    ("Qlearn_adv_1300", summary_Qlearn_adv_1300),
    ("Qlearn_adv_1400", summary_Qlearn_adv_1400),
    ("Qlearn_adv_1500", summary_Qlearn_adv_1500),
    ("Qlearn_adv_1600", summary_Qlearn_adv_1600),
    ("Qlearn_adv_1700", summary_Qlearn_adv_1700),
    ("Qlearn_adv_1800", summary_Qlearn_adv_1800),
    ("Qlearn_adv_1900", summary_Qlearn_adv_1900),
    ("Qlearn_adv_2000", summary_Qlearn_adv_2000),
    filename="summary_all_qlearn_adv.csv"
)
'''


  