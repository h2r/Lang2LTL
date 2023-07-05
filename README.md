# Installation
```
conda create -n lang2ltl python=3.9 dill matplotlib plotly scipy scikit-learn pandas tenacity
conda activate lang2ltl
pip install openai tiktoken 
pip install nltk seaborn pyyaml
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  # GPU
conda install pytorch torchdata -c pytorch  # CPU
pip install tensorboard transformers datasets evaluate torchtext
```

To install LTL/Automaton library [Spot](https://spot.lre.epita.fr/), use command below or follow the installation instructions [here](https://spot.lre.epita.fr/install.html).
```
conda install -c conda-forge spot
```

To upgrade openai package (e.g., to use GPT-4),
```
pip install openai --upgrade
```

To use transformer model T5,
```
pip install sentencepiece
```

## Install LLaMA (Optional)
Complete the installation instructions above.

To install [LLaMA](https://arxiv.org/abs/2302.13971), please first fill this [form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform) to request access to the model weights.
After downloading the model weights, convert and load them using HuggingFace's model classes. Please refer to the [documentation](https://huggingface.co/docs/transformers/main/model_doc/llama) for more instructions.



# Files
```lang2ltl.py```: modules and API of the Lang2LTL language grounding system.

```formula_sampler.py```: sample lifted LTL formulas given a formula type and the number of propositions.

```s2s_sup.py```: generic supervised sequence-to-sequence model.

```s2s_hf_transformers.py```: finetune pretrained transformer models from HuggingFace.

```s2s_pt_transformer.py```: train from scratch transformer encoder-to-decoder model implemented in PyTorch.

```gpt.py```: interface to GPT-3 and 4 model.

```get_embed.py```: interface to GPT-3 embedding.

```utils.py```: utility functions, e.g., build_placeholder_map, substitute, name_to_prop, etc.

```eval.py```: functions to evaluate translation and planning.

```exp_full.py```: main function to start running all experiments for evaluating the full language grounding system.

```dataset_symbolic.py```: construct lifted train and test sets for evaluating the lifted translation module.

```dataset_grounded.py```: construct grounded train and test sets using OSM or CleanUp landmarks for evaluation full translation system.

```dataset_filtered.py```: import test sets from Gopalan et al. 18 and Berg et al. 20.

```data_collection.py```: clean the collected lifted dataset of utterances, LTL formulas.

```analyze_results.py```: scripts to analyze results, e.g., confusion matrix, misclassification.

```tester.py```: unit tests.


# Finetuning T5 Models
Following instructions below to finetune [T5 models](https://arxiv.org/abs/1910.10683) on our dataset for lifted translation.
The datasets for reproducing all holdout test results are stored under `data/holdout_split_batch12_perm/`.
```
python s2s_hf_transformers.py --model=t5-base --data data/holdout_split_batch12_perm/{DESIRED_HOLDOUT_FOLD}.pkl
```
To reproduce the results presented in our paper, please use the following hyperparameters,
```
MAX_SRC_LEN = 512
MAX_TAR_LEN = 256
BATCH_SIZE = 40
learning_rate = 1e-4
weight_decay = 0.01
num_train_epochs = 5
```
We use a single NVIDIA GeForce RTX 3090 (24GB) for finetuning, and one fold of the holdout test set takes ~2 hours.



# Finetuning GPT-3
Set environment variable for OpenAI API key for current shell
```
export OPENAI_API_KEY=<YOUR_API_KEY>
```
Or permanently set the environment variable in your ```~/.bash_profile``` or ```~/.bashrc```.

## Data Formatting
1. Prepend a prefix `'Utterance: '` and append a separator `'\nLTL: '` to each input query, so your input will look like `'Utterance: {INPUT_UTTERANCE}\nLTL: '`
2. Prepend a whitespace and append a stop word `'\n'` to each completion, so your output will look like `' {OUTPUT_LTL}\n'`

## Prepare Data
Convert your formatted dataset into `.jsonl` type with CLI data preparation tool provided by OpenAI:
```
$openai tools fine_tunes.prepare_data -f your_file
```
Follow the directions and answer the prompted questions. For reproducing our results, DO NOT remove duplicates, DO NOT lowercase, and DO NOT split into training set and validation set.

## Create Finetuned Model
You need to submit finetuning jobs to OpenAI. Using the `.jsonl` file obtained from last step:
```
$openai api fine_tunes.create -t prepared_file.jsonl -m base_model --suffix "{MODEL_NAME}"
```
To reproduce our results, please use `davinci` as the base model and use default hyperparameters: epoch=4, batch_size=0.2%*training_set_size, learning_rate_multiplier=0.1

## Use the finetuned model
You can check the status of a finetuning job by:
```
$openai api fine_tunes.list
```
When a fintuning job is finished (i.e., `"status": "processed"`), you can find the name of that finetuned model in the list, and you can then use the finetuned model the same way as other OpenAI models through its API.

For more info, please refer to the [official document](https://platform.openai.com/docs/guides/fine-tuning).




# Model Weights
All model weights are stored on [Google Drive link](https://drive.google.com/drive/folders/1Rk_JICbHOArWZE6TRxwJZnHVd4wQ5abL?usp=sharing).

The model trained on the composed dataset is at [Google Drive link](https://drive.google.com/drive/folders/1rZl8tblyVj-pZZW4OgbO1NJwMIT2fwx9?usp=sharing)

## Download files and folders from Google Drive
```
pip install gdown
gdown <SHARED_LINK_FOR_FILE>
gdown --folder <SHARED_LINK_FOR_FOLDER>
```



# Run Experiments
Set environment variable for OpenAI API key for current shell
```
export OPENAI_API_KEY=<YOUR_API_KEY>
```
Or permanently set the environment variable in your ```~/.bash_profile``` or ```~/.bashrc```.

Create embeddings for the known landmarks or objects in the given environment.
```
python get_emebd.py
```

To generate lifted dataset, train test splits for training lifted translation module and prompts for off-the-shelf GPT-3 with permuted propositions and update existing symbolic dataset for batch 1 and 2 data.
```
python dataset_lifted.py --perm --update --merge
```

To generate grounded dataset from lifted dataset
```
python dataset_grounded.py --env=osm --city={CITYNAME}
```
where CITYNAME is the name of a file in the directory ```data/osm/osm_lmks``` without .json file extension.

To use Lang2LTL as an API
```
from lang2ltl import lang2ltl
out_ltl = lang2ltl(utt, lmk2sem, result_dpath)
```

To run experiments for the Lang2LTL modular system
```
python exp_full.py
```

To run experiments for the end-to-end translation with prompting GPT-3 or GPT-4
```
python exp_full.py --full_e2e
```

To run experiments for the modular_ner approach to translate language to LTL
```
python exp_full.py --translate_e2e
```


# Datasets
All data is stored on [Google Drive](https://drive.google.com/drive/folders/1ept4vnvlUevzqUellFt938vV2VDcgdwb?usp=sharing).

## Lifted Dataset
```symbolic_no_perm.csv``` contains pairs of utterances and LTL formulas whose propositions are symbolic, e.g., a, b, c, etc, used for training symbolic translation module.

```symbolic_perm.csv``` augments ```symbolic_no_perm.csv``` with permutations of propositions in utterances and their corresponding LTL formulas.

## Grounded Dataset: OpenStreetMap (OSM)
```osm_corlw.csv``` generated from ```providence_500.csv``` by running the ```create_osm_dataset``` function in ```dataset.py```.

## Grounded Dataset: Cleanup World
```cleanup_raw.csv``` contains the raw [Gopalan et al. 18 dataset](https://github.com/h2r/language_datasets/tree/master/RSS_2018_Gopalan_et_al)
for language commands paired LTL expressions, converted to 1 csv file from 2 txt files, ```hard_pc_src.txt``` and ```hard_pc_tar.txt```.

```cleanup_fixed.csv``` after fixing mistakes (e.g., mislabel, incorrect language, typos).

```cleanup_cleaned.csv``` after data cleaning (e.g., remove ambiguous language, manipulation).

```cleanup_corlw.csv``` generated from ```cleanup_cleaned.csv``` by running the ```generate_tar_file``` function in ```dataset.py```.
Convert propositions in target LTLs from letters to words joined by underscores.

## Compose Datasets
 (WIP) To construct training and test set of a composed dataset,
run
```
python dataset_composed_new.py
```
Composed dataset pkl file: dictionary of train data, train meta data, test data, test meta data and information used for construction.


## Large datasets
Large datasets are stored on [Google Drive](https://drive.google.com/drive/folders/1ept4vnvlUevzqUellFt938vV2VDcgdwb?usp=sharing),
e.g., composed datasets.




# Baselines
## CopyNet
Grounding Language to Landmarks in Arbitrary Outdoor Environments ([Berg et al. 18](https://h2r.cs.brown.edu/wp-content/uploads/berg20.pdf))
([code](https://github.com/jasonxyliu/magic-skydio))

## Code as Policies
Code as Policies: Language Model Programs for Embodied Control ([Liang, et al. 22](https://arxiv.org/abs/2209.07753))
([code](https://colab.research.google.com/drive/1UgMpP-b-TnSs4pgpTUj63sSSJsXRTgxC?usp=sharing))

## GRU-based Sequence-to-Sequence with Attention
Sequence-to-Sequence Language Grounding of Non-Markovian Task Specifications ([Gopalan et al. 18](https://h2r.cs.brown.edu/wp-content/uploads/gopalan18.pdf))
([code](https://github.com/h2r/lggltl/tree/master/models))



# Citation
[arXiv paper](https://arxiv.org/abs/2302.11649)
```
@inproceedings{liu23lang2ltl,
  title={Lang2LTL: Translating Natural Language Commands to Temporal Robot Task Specification},
  author={Liu, Jason Xinyu and Yang, Ziyi and Idrees, Ifrah and Liang, Sam and Schornstein, Benjamin and Tellex, Stefanie and Shah, Ankit},
  booktitle={arXiv preprint arXiv:2302.11649},
  year={2023}
}
```
