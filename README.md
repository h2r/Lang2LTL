# Installation
```
$conda create -n lang2ltl python=3.9 dill matplotlib plotly scipy scikit-learn pandas tenacity
$conda activate lang2ltl
$pip install openai
```
Optional if use spaCy
```
$pip install -U pip setuptools wheel
$pip install -U spacy
$python -m spacy download en_core_web_sm
```

# Run Experiments
Set environment variable for API key and orgnization ID
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
