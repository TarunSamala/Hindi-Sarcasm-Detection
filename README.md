# Hindi-Sarcasm-Detection

This repository contains implementations of various machine learning and deep learning models for detecting sarcasm in Hindi tweets. The models used include Bi-LSTM, Bi-GRU, Random Forest, and BERT, leveraging pre-trained Hindi word embeddings and a dataset from Kaggle.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Embeddings](#embeddings)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Introduction

Sarcasm detection is a challenging task in natural language processing, especially in social media contexts where the tone and intent can be ambiguous. This project aims to detect sarcasm in Hindi tweets using different machine learning approaches.

## Dataset

The dataset used in this project is the [Hindi Tweets Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/pragyakatyayan/hindi-tweets-dataset-for-sarcasm-detection) from Kaggle. It contains labeled tweets in Hindi, indicating whether each tweet is sarcastic or not.

To use the dataset, download it from Kaggle and place it in the `data/` directory.

## Models

The following models are implemented:

- **Bi-LSTM**: Bidirectional Long Short-Term Memory network, using pre-trained fastText embeddings.
- **Bi-GRU**: Bidirectional Gated Recurrent Unit network, using pre-trained fastText embeddings.
- **Random Forest**: Ensemble learning method for classification, using features derived from the text.
- **BERT**: Bidirectional Encoder Representations from Transformers, using the multilingual cased model.

## Embeddings

Pre-trained word embeddings for Hindi are used for the Bi-LSTM and Bi-GRU models. Specifically, the [fastText](https://fasttext.cc/docs/en/crawl-vectors.html) word vectors trained on Common Crawl and Wikipedia are utilized. These embeddings are 300-dimensional and are loaded using the `fasttext` library.

For BERT, the multilingual cased model is used, which supports Hindi among other languages.

## Setup

To set up the environment, install the required libraries:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:

- `torch`
- `transformers`
- `scikit-learn`
- `fasttext`
- `pandas`
- `numpy`

Download the pre-trained fastText embeddings for Hindi:

```python
import fasttext.util
fasttext.util.download_model('hi', if_exists='ignore')
```

## Usage

1. **Data Preparation**: Load and preprocess the dataset.
2. **Model Training**: Train the models using the training scripts.
3. **Evaluation**: Evaluate the models on the test set.

Example commands:

```bash
python train_bilstm.py
python train_bigru.py
python train_randomforest.py
python train_bert.py
```

## Results

[To be added once experiments are conducted]

## References

- [fastText Word Vectors](https://fasttext.cc/docs/en/crawl-vectors.html)
- [BERT Multilingual Model](https://huggingface.co/bert-base-multilingual-cased)
- [Hindi Tweets Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/pragyakatyayan/hindi-tweets-dataset-for-sarcasm-detection)