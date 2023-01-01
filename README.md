# Installation
```
conda create -n lang2ltl python=3.9 dill matplotlib plotly scipy scikit-learn pandas tenacity
conda activate lang2ltl
pip install openai
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
```experiment.py```: main function to start running all experiments

```formula_sampler.py```: sample symbolic LTL formulas given formula type and the number of propositions

```s2s_sup.py```: for inference pretrained supervised sequence-to-sequence model

```s2s_hf_transformers.py```: to finetune pretrained transformer models from Hugging Face

```s2s_pt_transformer.py```: to train transformer sequence-to-sequence model implemented in PyTorch

```gpt3.py```: interface to GPT-3 model

```get_embed.py```: interface to GPT-3 embedding feature

```dataset.py```: construct training and test datasets

```data_collection.py```: clean the collected symbolic dataset of utterances, LTL formulas.


# Run Experiments
Temporarily set environment variables for API key and organization ID
```
export OPENAI_API_KEY=<YOUR_API_KEY>
export ORG_ID=<YOUR_ORG_ID>
```
Or permanently set the above environment variables in your ```~/.bash_file``` or ```~/.bashrc```.

Run end-to-end language to LTL translation using GPT-3
```
python run_experiment.py --full_e2e
```
Run the modular_ner approach to language to LTL translation
```
python run_experiment.py --translate_e2e
```
Run the modular_ner+placeholders approach to language to LTL translation
```
python run_experiment.py
```

# Datasets
## Cleanup World
```cleanup_raw.csv``` contains the raw [Gopalan et al. 18 dataset](https://github.com/h2r/language_datasets/tree/master/RSS_2018_Gopalan_et_al)
for language commands paired LTL expressions, converted to 1 csv file from 2 txt files, ```hard_pc_src.txt``` and ```hard_pc_tar.txt```.

```cleanup_fixed.csv``` after fixing mistakes (e.g. mislabel, incorrect language, typos).

```cleanup_cleaned.csv``` after data cleaning (e.g. remove ambiguous language, manipulation).

```cleanup_corlw.csv``` generated from ```cleanup_cleaned.csv``` by running the ```generate_tar_file``` function in ```dataset.py```.
Convert propositions in target LTLs from letters to words joined by underscores.

## OpenStreetMap (OSM)
```osm_corlw.csv``` generated from ```providence_500.csv``` by running the ```create_osm_dataset``` function in ```dataset.py```.

## Symbolic
```symbolic_no_perm.csv``` contains pairs of utterances and LTL formulas whose propositions are symbolic, e.g. a, b, c, etc, used for training symbolic translation module.

```symbolic_perm.csv``` augments ```symbolic_no_perm.csv``` with permutations of propositions in utterances and their corresponding LTL formulas.
