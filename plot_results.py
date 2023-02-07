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

def plot_osm_accuracies():
    df = create_osm_accuracies_table()
    with sns.plotting_context('poster', rc=rc):
       plt.figure(figsize=[12, 10])
       sns.barplot(data = df, x = 'Test Type',y = 'Accuracy',hue = 'Model', alpha = 0.85, capsize = 0.1)
       plt.ylim([0,1])
       plt.savefig('figures/osm_accuracies.jpg', dpi=400, bbox_inches='tight')
       plt.show()

def plot_type_accuracies():
    for test_type in SYMBOLIC_TEST_TYPES:
        for model_type in SYMBOLIC_MODEL_TYPES:
            #print(test_type, model_type)
            accs = parse_per_type_accuracies(test_type, model_type)
            #print(accs)
            with sns.plotting_context(context = 'poster', rc = rc):
                sns.set_color_codes('muted')
                plt.figure(figsize = [12,10])
                sns.barplot(data = accs, x = 'Accuracy', y = 'Formula Type', color = 'b', ci = None)
                plt.savefig(f'figures/type_accs_{test_type}_{model_type}.jpg', dpi = 400, bbox_inches = 'tight')
    
def plot_n_prop_accuracies():
    for test_type in SYMBOLIC_TEST_TYPES:
        for model_type in SYMBOLIC_MODEL_TYPES:
            #print(test_type, model_type)
            accs = parse_n_prop_accuracies(test_type, model_type)
            #print(accs)
            with sns.plotting_context(context = 'poster', rc = rc):
                sns.set_color_codes('muted')
                plt.figure(figsize = [12,10])
                sns.barplot(data = accs, x = 'N Propositions', y = 'Accuracy', color = 'b')
                plt.ylim([0,1])
                plt.savefig(f'figures/n_prop_accs_{test_type}_{model_type}.jpg', dpi = 400, bbox_inches = 'tight')
    
    
if __name__ == '__main__':
    #plot_symbolic_accuracies1()
    #plot_symbolic_accuracies2()
    #plot_osm_accuracies()
    plot_type_accuracies()
    plot_n_prop_accuracies()