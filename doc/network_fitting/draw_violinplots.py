

import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats 
import sys
import ringity as rng
rng.set_theme()

try:
    input_file = sys.argv[1]
except IndexError:
    input_file = "data/homophily_scores.csv"
    

df = pd.read_csv(input_file, index_col=0)

assert (df.columns == ['network_file', 'embedding_file', 'embedding_iterations',
       'randomization', 'analyzer_interaction_width',
       'analyzer_max_probability']).all()

df["network"] = df['network_file'].map(lambda x : x.split("/")[-1].split(".")[0])

for network in df["network"].unique():
    
    df_network_select = df.loc[df["network"] == network]
    print(df_network_select["randomization"].unique())
    
    config_model_homophily_score = df_network_select.loc[df_network_select["randomization"] == "configuration"].analyzer_interaction_width.values
    true_homophily_score = df_network_select.loc[df_network_select["randomization"] == "False"].analyzer_interaction_width.values
    
    fig,ax = plt.subplots(figsize=(10,6))
    
    res = scipy.stats.mannwhitneyu(config_model_homophily_score,true_homophily_score)
    sns.violinplot([config_model_homophily_score,true_homophily_score],ax=ax,color="#AAAAAA")
    
    ax.set_xticklabels(["Configuration\nModel\nControl", "True\nModel"])
    ax.set_xlabel("Model")
    ax.set_ylabel("Homophily Score")
    plt.suptitle(network+str(res.pvalue))
    
    plt.show()
    
    
    