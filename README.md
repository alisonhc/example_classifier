# Introduction
This repo is to build an example classifier with a fancy deep learning framework, [AllenNLP](https://github.com/allenai/allennlp), for research usage.

In short, the example classifier aims at classifying sentences into good or bad sentences for English learners.  
For example:
  - `In this article, I will first provide an overview of the system of accreditation and then discuss issues of accreditation as they apply to these contemporary American educational programs in Japan.` is a bad sentence.
  - `Town meetings should be held to discuss issues.` is a good sentence.

AllenNLP is a PyTorch framework for NLP. It abstracts the process of NLP tasks into several components so that we can reduce a lot of duplicated and messy works, such as the training loop, gradient clipping, or passing data among components. If you are not familiar with AllenNLP, please refer to their fantastic [tutorial](https://guide.allennlp.org/).

# Setup
> The python version used is **3.8**

First, create a new virtual environment (recommended) and install the dependencies.
```bash
pip install -r requirements.txt
```

Second, download the [data](https://drive.google.com/file/d/1eijo9i2Erg0ZS9FjX1fieNyDKbbz4Y3p/view?usp=sharing) for modeling.

Third, create a directory for storing models.
```
mkdir models
```

Once we have done, the structure of the project should look like
```
├── classifier  
├── model_config  
├── models  
├── data 
│   ├── v1 
│   ├── raw 
│   └── test
├── requirements.txt   
├── README.md 
└── .gitignore
```

# Training
We can train a model using allennlp train command with a configuration file easily.
```bash
allennlp train model_config/v1_bert.jsonnet -s models/v1_bert -f --include-package classifier
```
> Note: The existing models/v1_bert will be replaced if you run the above command. You can change the `-s` argument to keep the original model.

# Prediction
## File
Once we have got a model, we can perform classification on a set of sentences. The output would contain raw sentences and their probabilities of being good sentences.
```bash
allennlp predict models/v1_bert/model.tar.gz data/test/3000_general_wordlist.jsonl --output-file 3000_general_wordlist.pred.jsonl --include-package classifier --predictor example_predictor --batch-size 1024 --cuda 1 --silent
```
> Note: You can change the argument [--cuda 1] to [--cuda -1] if you don't want to use a GPU.

## A Single Sentence
Also, we can use the model in programs to estimate the probability of being a good sentence.
```python
from classifier.predictor import ExamplePredictor

clr = ExamplePredictor.from_path('models/v1_bert/model.tar.gz', 'example_predictor')

good_sentence = 'Town meetings should be held to discuss issues.'
bad_sentence = 'In this article, I will first provide an overview of the system of accreditation and then discuss issues of accreditation as they apply to these contemporary American educational programs in Japan.'


good_sent_probs = clr.predict_probs({'text': good_sentence})
bad_sent_probs = clr.predict_probs({'text': bad_sentence})

print(f'good sentence: {round(good_sent_probs, 2)}')
>>> 0.98

print(f'bad sentence: {round(bad_sent_probs, 2)}')
>>> 0.1
```