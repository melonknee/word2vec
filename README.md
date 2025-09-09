# Word2Vec - A Simple Word2Vec Implementation

## Overview
A lightweight Python-based Word2Vec model showcasing how to convert text into vector representations using the skip-gram model. This project aims to explore word embeddings and the training process of how machines begin to form an understanding of the semantics of words through rich meaningful embeddings.

### Skip-Gram
[Diagram]

## Stack:
Python, Jupyter Notebook, Matplotlib, sci-kit learn, NumPy, PyTorch

## Embedding Techniques
### Subsampling

 # Results
## Visualisations
### t-SNE (t-Distributed Stochastic Neighbour Embedding)
This non-linear technique is used for visualising high-dimensional data. It converts high-dimensional distances between data points into probabilities, then creating a low-dimensional map where similar high-dimensional points are modeled as nearby points in the low-dimensional space.

### PCA (Principal Component Analysis)
This linear technique reduces the dimensionality of the embeddings, whilst preserving global data structures and variance, allowing us to create visualisations that show interpretable, deterministic results.
It is less effective for non-linear data, hence why I used it to explore the arithmetic relationship of the words.

## Statistics
### Most Similar Words


## Limitations
### Polysemy: words may have multiple meanings e.g. coach = big bus||to teach someone||the brand "Coach". In this model, each word only has one embedding to represent it, meaning it may not be able to encapsulate these multiple meanings
### Out of Vocabulary (OOV) Words
### Extensions
At the end of the day, this version of Word2Vec is just a very simple static word embedding model. In more complex applications, contextualised word embeddings are usually used, as they provide dynamic, context-aware representations. These


### Set Up


## Training the Model
