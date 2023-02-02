# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:07:31 2023

@author: AJShah
"""
import seaborn as sns
import matplotlib.pyplot as plt
from results_analysis import *

RESULT_DPATH = './figures'
rc = {'axes.labelsize': 28, 'axes.titlesize': 32, 'legend.fontsize': 24, 'xtick.labelsize': 24, 'ytick.labelsize': 22}

def plot_symbolic_accuracies1():
    df = create_symbolic_accuracies_table(test_types = SYMBOLIC_TEST_TYPES, model_types = SYMBOLIC_MODEL_TYPES)
    with sns.plotting_context('poster', rc=rc):
       plt.figure(figsize=[12, 10])
       sns.barplot(data = df, x = 'Model',y = 'Accuracy',hue = 'Test Type', alpha = 0.85, capsize = 0.1)
       plt.ylim([0,1])
       plt.savefig('figures/symbolic_accuracies.jpg', dpi=400, bbox_inches='tight')
       plt.show()


def plot_symbolic_accuracies2():
    df = create_symbolic_accuracies_table(test_types = SYMBOLIC_TEST_TYPES, model_types = SYMBOLIC_MODEL_TYPES)
    with sns.plotting_context('poster', rc=rc):
       plt.figure(figsize=[12, 10])
       sns.barplot(data = df, x = 'Test Type',y = 'Accuracy',hue = 'Model', alpha = 0.85, capsize = 0.1)
       plt.ylim([0,1])
       plt.savefig('figures/symbolic_accuracies2.jpg', dpi=400, bbox_inches='tight')
       plt.show()

    
if __name__ == '__main__':
    plot_symbolic_accuracies1()
    plot_symbolic_accuracies2()