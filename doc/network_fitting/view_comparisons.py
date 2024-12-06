
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats



COLOR_SCHEME = {"Jasmine"    :"#ffd07b",
                "Glaucous"   :"#577399",
                "Dark purple":"#412234",
                "Moss green" :"#748e54",
                "Keppel"     :"#44bba4"
               }


COLOR_DICT = {
"immune"   :COLOR_SCHEME["Keppel"],
"fibro"    :COLOR_SCHEME["Glaucous"],
"gene"     :COLOR_SCHEME["Moss green"],
"lipid"    :COLOR_SCHEME["Jasmine"],
"soil"     :COLOR_SCHEME["Dark purple"],
"gene_corrected" :COLOR_SCHEME["Moss green"]
}


def main():
    
    df = pd.read_csv("data/network_fitting_stats.csv",
                     index_col=0
                     )
    
    name = "gene_corrected"
    
    particular_network_df = df.loc[df["parameters_choice_network_name"]==name,:]
    
    color = COLOR_DICT[name]
    
    tmp=particular_network_df['parameters_choice_network_model']=="configuration"
    configuration_values = list(particular_network_df.loc[tmp,"fitter_w"].values)
    
    tmp=particular_network_df['parameters_choice_network_model']=="this<_model"
    this_paper_values    = list(particular_network_df.loc[tmp,"fitter_w"].values)
    
    tmp=particular_network_df['parameters_choice_network_model']=="none"
    true_value = particular_network_df.loc[tmp,"fitter_w"].mean()
    print(configuration_values, this_paper_values, true_value, color)
    draw_figure(configuration_values, this_paper_values, true_value, color)
    
    
    
    
    
    
    
import seaborn as sns


def draw_figure(configuration_values, this_paper_values, true_value,color='k'):

    test = scipy.stats.ttest_ind(true_value,
                                 configuration_values
                                 )
    
    
    
    fig,ax = plt.subplots()
    sns.violinplot([this_paper_values,configuration_values],color=color,inner="quart")
    ax.scatter([2],[true_value],c=color,s=200,edgecolor='k')

    fontsize = 20
    ax.set_ylabel("Fit Intercept",fontsize=fontsize)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["Our Model", "Configuration\n Model", "True"],fontsize=fontsize)
    
    x1, x2 = 1,2   
    y, h = np.max(configuration_values) + 0.2, 0.5
    draw_significance_stars(ax,test.pvalue,x1,x2,y,h)
    fig.savefig(f"test/comparison.png", dpi=500)
    
    fp = open(f"test/pvalue.txt", 'w')
    fp.write(str(test.pvalue))
    fp.close()



    
def draw_significance_stars(ax,pvalue,x1,x2,y,h,fontsize=20):
    
    col='k'
    
    if pvalue < 0.001:
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col,fontsize=fontsize)

    elif pvalue < 0.01:
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col,fontsize=fontsize)
        
    elif pvalue < 0.05:
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=fontsize)

    ymin,ymax = ax.get_ylim()
    ax.set_ylim((ymin,ymax+1))
    
    

main()
    
    