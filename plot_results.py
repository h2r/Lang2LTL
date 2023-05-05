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
                plt.xlim([0,1])
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
                sns.barplot(data = accs, x = 'N Propositions', y = 'Accuracy', color = 'b', capsize = 0.1)
                plt.ylim([0,1])
                plt.savefig(f'figures/n_prop_accs_{test_type}_{model_type}.jpg', dpi = 400, bbox_inches = 'tight')

def plot_per_city_accuracy():
    for test_type in SYMBOLIC_TEST_TYPES:
        data = parse_per_city_accs(test_type)
        with sns.plotting_context(context = 'poster',rc = rc):
            sns.set_color_codes('muted')
            plt.figure(figsize=[10,12])
            sns.barplot(data = data, x = 'Accuracy',y = 'City', color = 'b', capsize = 0.1)
            plt.xlim([0,1])
            plt.savefig(f'figures/city_accuracies_{test_type}.jpg', dpi = 400, bbox_inches = 'tight')
    
def symbolic_error_piechart():
    data = [1688, 690, 205, 524, 212]
    with sns.plotting_context(context = 'poster', rc =rc):
        plt.figure(figsize=[12,10])
        labels = ['Misclassified Template','Incorrect Propositions', 'Incorrect Permutation', 'Unknown Templates', 'Invalid LTL']
        colors = sns.color_palette('muted',n_colors=5)
        plt.pie(data, labels=labels, colors = colors)
        plt.savefig('figures/symbolic_piechart.jpg',dpi=400, bbox_inches = 'tight')
    
def symbolic_error_piechart_finetuned_formula():
    data = [4539, 4794, 178, 2326, 6830]
    with sns.plotting_context(context = 'poster', rc =rc):
        plt.figure(figsize=[12,10])
        labels = ['Misclassified Template','Incorrect Propositions', 'Incorrect Permutation', 'Unknown Templates', 'Invalid LTL']
        colors = sns.color_palette('muted',n_colors=5)
        plt.pie(data, labels=labels, colors = colors)
        plt.savefig('figures/symbolic_piechart_finetuned_formula.jpg',dpi=400, bbox_inches = 'tight')

def symbolic_error_piechart_finetuned_type():
    data = [47204, 11, 7, 1310, 733]
    with sns.plotting_context(context = 'poster', rc =rc):
        plt.figure(figsize=[12,10])
        labels = ['Misclassified Template','Incorrect Propositions', 'Incorrect Permutation', 'Unknown Templates', 'Invalid LTL']
        colors = sns.color_palette('muted',n_colors=5)
        plt.pie(data, labels=labels, colors = colors)
        plt.savefig('figures/symbolic_piechart_finetuned_type.jpg',dpi=400, bbox_inches = 'tight')


def plot_full_system_piechart(test_type):
    symbolic_errors, RER_errors = get_full_system_piechart(test_type)
    with sns.plotting_context(context='poster',rc = rc):
        plt.figure(figsize=[12,10])
        labels = ['Symbolic Translation Error', 'Proposition Grounding Error']
        colors = sns.color_palette('muted',n_colors = 2)
        plt.pie([symbolic_errors, RER_errors], labels = labels, colors = colors)
        plt.savefig(f'figures/full_system_{test_type}.jpg',dpi=400, bbox_inches = 'tight')

def plot_bar_charts():
    data = create_osm_accuracies_table()
    data = data.loc[data['Test Type'] == 'Utterance']
    entry = {}
    entry['Model ID'] = [0]
    entry['Accuracy'] = [0.38]
    entry['Model'] = ['Prompt GPT-3']
    entry['Test Type'] = ['Utterance']
    data = pd.concat([data, pd.DataFrame(entry)], axis=0, ignore_index=True)
    
    with sns.plotting_context(context='poster', rc = rc):
        plt.figure(figsize=[12,10])
        sns.barplot(data = data, x = 'Model', y = 'Accuracy', ci=None)
        plt.ylim([0,1])
        plt.savefig(f'figures/full_system_accuracy.jpg',dpi=400, bbox_inches = 'tight')

if __name__ == '__main__':
    #plot_symbolic_accuracies1()
    #plot_symbolic_accuracies2()
    #plot_osm_accuracies()
    #plot_type_accuracies()
    #plot_n_prop_accuracies()
    #plot_per_city_accuracy()
    #ymbolic_error_piechart()
    #plot_full_system_piechart('utt_holdout')
    #plot_full_system_piechart('formula_holdout')
    #plot_bar_charts()
    symbolic_error_piechart_finetuned_formula()
    symbolic_error_piechart_finetuned_type()