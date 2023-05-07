# Installation
```
conda create -n lang2ltl python=3.9 dill matplotlib plotly scipy scikit-learn pandas tenacity
conda activate lang2ltl
pip install openai tiktoken 
pip install nltk seaborn
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  # GPU
conda install pytorch torchtext torchdata -c pytorch  # CPU
pip install tensorboard transformers datasets evaluate
```

If needed, upgrade openai package (e.g., to use GPT-4)
```
pip install openai --upgrade
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

## Install LLaMA
Make sure to complete the installation instructions above.

Then too install [LLaMA](https://arxiv.org/abs/2302.13971), please first fill this [form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform) to request access to model weights. After downloaded, the weights can be converted and loaded with huggingface model classes. Please refer to the [huggnigface documentation](https://huggingface.co/docs/transformers/main/model_doc/llama) for more instructions.



# Files
```experiment.py```: main function to start running all experiments for evaluation full translation system.

```lang2llt.py```: modules of Lang2LTL translation system and an API.

```formula_sampler.py```: sample symbolic LTL formulas given formula type and the number of propositions.

```utils.py```: utility functions, e.g., substitute, name_to_prop.

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

```analyze_results.py```: scripts to analyze results, e.g., confusion matrix, misclassification.

```tester.py```: unittests.

```dataset_corlw.py```: construct grounded train and test sets for CoRL22-W.



# Finetuning GPT-3
*Make sure you have set OpenAI API keys before running following steps*
## Data Formatting
1. Prepend a prefix `'Utterance: '` and append a separator `'\nLTL: '` to each of the input query, so your input will look like `'Utterance: {input_sequence}\nLTL: '`
2. Prepend a whitespace and append a stop word `'\n'` to each of the completion, so your output will look like `' {output_sequence}\n'`

## Prepare Data
Convert your formatted dataset into `.jsonl` type with CLI data preparation tool provided by OpenAI:
```
$openai tools fine_tunes.prepare_data -f your_file
```
Follow the directions and answer the promtped questions. For reproducing our results, DO NOT remove duplicates, DO NOT lowercase, and DO NOT split into training set and validation set.

## Create Finetuned Model
You'll need to submit finetuning jobs to OpenAI to get the finetuned model. Using the `.jsonl` file we just got converted:
```
$openai api fine_tunes.create -t prepared_file.jsonl -m base_model --suffix "your_desired_model_name"
```
For reproducing our results, please use `davinci` as the base model and use default hyperparameters: epoch=4, batch_size=0.2%*training_set_size, learning_rate_multiplier=0.1
## Use the model
You can check the status of a finetuning job by:
```
$openai api fine_tunes.list
```
When a job is finished (`"status": "processed"`), you will be able to find the name of that finetuned model in the list, and you can then use the finetuned model the same way as other OpenAI models through the API.

For more info, please refer to the [official document](https://platform.openai.com/docs/guides/fine-tuning)


# Finetuning T5 Models
Besides finetuning GPT models, we can also finetune [T5 models](https://arxiv.org/abs/1910.10683) for symbolic translation using our dataset.
T5 models are open source, so we can customize the model architecture and have ownership of the finetuned model weights.
The datasets for reproducing all holdout test results are stored under `data/holdout_split_batch12_perm/`.
For finetuning, run
```
python s2s_hf_transformers.py --model=t5-base --data data/holdout_split_batch12_perm/{DESIRED_HOLDOUT_FOLD}.pkl
```
To reproduce the exact results presented in our paper, please use the following hyperparameters,
```
MAX_SRC_LEN = 512
MAX_TAR_LEN = 256
BATCH_SIZE = 40
learning_rate = 1e-4
weight_decay = 0.01
num_train_epochs = 5
```
We use a single NVIDIA GeForce RTX 3090 (24GB) for finetuning, and completing one fold of the holdout test set takes ~2 hours.



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
```symbolic_no_perm.csv``` contains pairs of utterances and LTL formulas whose propositions are symbolic, e.g., a, b, c, etc, used for training symbolic translation module.

```symbolic_perm.csv``` augments ```symbolic_no_perm.csv``` with permutations of propositions in utterances and their corresponding LTL formulas.

## OpenStreetMap (OSM)
```osm_corlw.csv``` generated from ```providence_500.csv``` by running the ```create_osm_dataset``` function in ```dataset.py```.

## Cleanup World
```cleanup_raw.csv``` contains the raw [Gopalan et al. 18 dataset](https://github.com/h2r/language_datasets/tree/master/RSS_2018_Gopalan_et_al)
for language commands paired LTL expressions, converted to 1 csv file from 2 txt files, ```hard_pc_src.txt``` and ```hard_pc_tar.txt```.

```cleanup_fixed.csv``` after fixing mistakes (e.g., mislabel, incorrect language, typos).

```cleanup_cleaned.csv``` after data cleaning (e.g., remove ambiguous language, manipulation).

```cleanup_corlw.csv``` generated from ```cleanup_cleaned.csv``` by running the ```generate_tar_file``` function in ```dataset.py```.
Convert propositions in target LTLs from letters to words joined by underscores.

## Compose Datasets
To construct composed dataset for zero shot transfer, utterance and formula holdout,
first run 
```
python compose.py
```
then run
```
python dataset_composed.py
```

Composed dataset for zero shot transfer: dictionary of data and meta

Composed dataset for utterance and formula holdout: dictionary of train data, train meta data, test data, test meta data and information of base datasets used to construct the composed dataset.


## Large datasets
Large datasets are stored on [Google Drive](https://drive.google.com/drive/folders/1ept4vnvlUevzqUellFt938vV2VDcgdwb?usp=sharing),
e.g., composed datasets.



# Baselines
## CopyNet
Grounding Language to Landmarks in Arbitrary Outdoor Environments [Berg et al. 18](https://h2r.cs.brown.edu/wp-content/uploads/berg20.pdf),
[code](https://github.com/jasonxyliu/magic-skydio)

## GRU-based Sequence-to-Sequence-Attention
Sequence-to-Sequence Language Grounding of Non-Markovian Task Specifications [Gopalan et al. 18](https://h2r.cs.brown.edu/wp-content/uploads/gopalan18.pdf)
[code](https://github.com/h2r/lggltl/tree/master/models)

## Code as Policies
Code as Policies: Language Model Programs for Embodied Control [Liang, et al. 22](https://arxiv.org/abs/2209.07753)
[code](https://colab.research.google.com/drive/1UgMpP-b-TnSs4pgpTUj63sSSJsXRTgxC?usp=sharing)
