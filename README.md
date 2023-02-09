# Installation
```
conda create -n lang2ltl python=3.9 dill matplotlib plotly scipy scikit-learn pandas tenacity
conda activate lang2ltl
pip install openai
pip install nltk seaborn
conda install pytorch torchtext torchdata -c pytorch
pip install tensorboard
pip install transformers datasets evaluate
```

To install LTL/Automaton library Spot
```
conda install -c conda-forge spot
```
or follow the installation instructions [here](https://spot.lre.epita.fr/install.html).

Optional. Install if use transformer model T5
```
pip install sentencepiece
```

Optional. Install if use spaCy
```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```

# Files
```experiment.py```: main function to start running all experiments for evaluation full translation system.

```lang2llt.py```: modules of Lang2LTL translation system and an API.

```formula_sampler.py```: sample symbolic LTL formulas given formula type and the number of propositions.

```utils.py```: utility functions, e.g. substitute, name_to_prop.

```s2s_sup.py```: generic supervised sequence-to-sequence model.

```s2s_hf_transformers.py```: finetune pretrained transformer models from Hugging Face.

```s2s_pt_transformer.py```: train from scratch transformer sequence-to-sequence model implemented in PyTorch.

```gpt3.py```: interface to GPT-3 model.

```get_embed.py```: interface to GPT-3 embedding feature.

```dataset_symbolic.py```: construct symbolic train and test sets for evaluating symbolic translation module.

```dataset_grounded.py```: construct grounded train and test sets using OSM or CleanUp landmarks for evaluation full translation system.

```dataset_filtered.py```: import test sets from Gopalan et al. 18 and Berg et al. 20.

```data_collection.py```: clean the collected symbolic dataset of utterances, LTL formulas.

```evaluation.py```: functions to evaluate translation and planning.

```analyze_results.py```: scripts to analyze results, e.g. confusion matrix, misclassification.

```tester.py```: unittests.

```dataset_corlw.py```: construct grounded train and test sets for CoRL22-W.


# Run Experiments
Temporarily set environment variables for API key and organization ID
```
export OPENAI_API_KEY=<YOUR_API_KEY>
export ORG_ID=<YOUR_ORG_ID>
```
Or permanently set the above environment variables in your ```~/.bash_profile``` or ```~/.bashrc```.

Create embeddings for known landmarks or objects in the given environment.
```
python get_emebd.py
```

To generate symbolic dataset, train test splits for training symbolic translation module and prompts for off-the-shelf GPT-3 with permuted propositions and update existing symbolic dataset for batch 1 and 2 data.
```
python dataset_symbolic.py --perm --update --merge
```

Generate grounded dataset from symbolic dataset
```
python dataset_grounded.py --env=osm --city=CITYNAME
```
where CITYNAME is the name of a file in the directory ```data/osm/osm_lmks``` without .json file extension.

Use Lang2LTL as an API
```
from lang2ltl import lang2ltl
out_ltl = lang2ltl(utt, lmk2sem, result_dpath)
```

Run experiments for the Lang2LTL modular system
```
python run_experiment.py
```

Run experiments for the end-to-end GPT-3 translation system with prompt
```
python run_experiment.py --full_e2e
```

Run experiments for the modular_ner approach to translate language to LTL translation
```
python run_experiment.py --translate_e2e
```


# Datasets
## Symbolic
```symbolic_no_perm.csv``` contains pairs of utterances and LTL formulas whose propositions are symbolic, e.g. a, b, c, etc, used for training symbolic translation module.

```symbolic_perm.csv``` augments ```symbolic_no_perm.csv``` with permutations of propositions in utterances and their corresponding LTL formulas.

## OpenStreetMap (OSM)
```osm_corlw.csv``` generated from ```providence_500.csv``` by running the ```create_osm_dataset``` function in ```dataset.py```.

## Cleanup World
```cleanup_raw.csv``` contains the raw [Gopalan et al. 18 dataset](https://github.com/h2r/language_datasets/tree/master/RSS_2018_Gopalan_et_al)
for language commands paired LTL expressions, converted to 1 csv file from 2 txt files, ```hard_pc_src.txt``` and ```hard_pc_tar.txt```.

```cleanup_fixed.csv``` after fixing mistakes (e.g. mislabel, incorrect language, typos).

```cleanup_cleaned.csv``` after data cleaning (e.g. remove ambiguous language, manipulation).

```cleanup_corlw.csv``` generated from ```cleanup_cleaned.csv``` by running the ```generate_tar_file``` function in ```dataset.py```.
Convert propositions in target LTLs from letters to words joined by underscores.
