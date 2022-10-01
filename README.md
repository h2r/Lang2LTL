# Installation
```
$conda create -n lang2ltl python=3.9 dill
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
Set environment variable for API key
```
export OPENAI_API_KEY=<YOUR_API_KEY>
```
