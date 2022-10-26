# Installation
```
conda create -n lang2ltl python=3.9 dill matplotlib plotly scipy scikit-learn pandas tenacity
conda activate lang2ltl
pip install openai
conda install -c conda-forge spot
```
Follow Spot installation instruction [here](https://spot.lre.epita.fr/install.html).

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
python run_experiment.py --overall_e2e
```
Run the modular approach to language to LTL translation
```
python run_experiment.py
```

# Datasets
```test_src_raw.txt``` and ```test_tar_raw.txt``` are the raw [Gopalan et al. 18 dataset](https://github.com/h2r/language_datasets/tree/master/RSS_2018_Gopalan_et_al) for language commands paired LTL expressions, renamed from ```hard_pc_src.txt``` and ```hard_pc_tar.txt```. 

```test_src_fixed.txt``` and ```test_tar_fixed.txt``` after fixing mistakes (e.g. mislabel, incorrect language, typos).

```test_src_cleaned.txt``` and ```test_tar_cleaned.txt``` after data cleaning (e.g. remove ambiguous language, manipulation).
