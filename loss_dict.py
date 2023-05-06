import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle



def plot_curve(ax, x, y, out_file, fig_title, measure):
    color_list = ["b", "c", "r", "m", "y", "k", "g"]
    label_name = measure

    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.tick_params(direction = "out", length = 3, width = 1)
    
    for i in range(len(y[0])):
       ax.plot(x, y[:,i], linewidth = 1, label = label_name[i], color=color_list[i], marker= None, markersize = 6, markeredgecolor = color_list[i], markeredgewidth=1, markerfacecolor='w')   

    x_min_val = 0
    x_max_val = 500
    y_min_val = 0
    y_max_val = 5
    y_scale = 10
    
    x_index = [i for i in range(x_min_val, x_max_val+1, int(x_max_val/10))]
    y_index = [i/y_scale for i in range(y_min_val*y_scale, y_max_val*y_scale+1, int(y_max_val*y_scale/10))] 
    ax.set_xticks(x_index)
    ax.set_xticklabels(x_index, fontsize = 16)
    ax.set_xlim([0, x_max_val*1.01])
    ax.set_yticks(y_index)
    ax.set_yticklabels(y_index, fontsize = 16)
    ax.set_ylim([y_min_val, y_max_val*1.01])
        
    ax.set_xlabel("Epoch", fontsize = 20, labelpad = 2)
    ax.set_ylabel("Measure", fontsize = 20, labelpad = 2)
    
    ax.legend(bbox_to_anchor=(0.6, 1), loc='upper right', borderaxespad=1, fontsize=12)


########################################################################################
if __name__ == '__main__':

    out_file = './dataplot/result.png'
    fig_title = 'loss curve'
    measure =['Cumulative_loss X 0.1','FAPE','RMSD','plddt_loss','pTM_loss','LDDT','pTM-Score']

    with open('./example/output/result.pkl','rb') as f:
        output_dict = pickle.load(f)
    y = output_dict['loss_curve']
    print(y)
    

    fig = plt.figure(figsize = (10,10))
    plt.rcParams["figure.subplot.left"] = 0.1 
    plt.rcParams["figure.subplot.bottom"] = 0.1
    plt.rcParams["figure.subplot.right"] = 0.9 
    plt.rcParams["figure.subplot.top"] = 0.9

    ax = fig.add_subplot(1,1,1)

    
    x = [i for i in range(len(y))]
    x = np.array(x)
    y = np.array(y)
    """
    y1=[]
    for i in range(len(y)):
        y1.append(y[i][1])
    print(y1)
    """
    

    
    print(x)
  
         
    plot_curve(ax, x, y, out_file, fig_title, measure)


    fig.subplots_adjust(bottom=0.1, left=0.1, top=0.9, right=0.9, hspace=0.3)

    plt.savefig(out_file, dpi=300) 
    plt.show()
    plt.clf()
    plt.close()
