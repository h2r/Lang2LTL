# Installation
```
conda create -n lang2ltl python=3.9 dill matplotlib plotly scipy scikit-learn pandas tenacity
conda activate lang2ltl
pip install openai
```

To install LTL/Automaton library Spot
```
conda install -c conda-forge spot
```
or follow the installation instructions [here](https://spot.lre.epita.fr/install.html).

Optional if use Sequence-to-Sequence model
```
conda install pytorch torchtext torchdata -c pytorch
```

Optional if use spaCy
```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```

# Run Experiments
Set environment variable for API key and organization ID
```
export OPENAI_API_KEY=<YOUR_API_KEY>
export ORG_ID=<YOUR_ORG_ID>
```
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
