# Word2Vec - A Simple Word2Vec Implementation

## Overview
A lightweight Python-based Word2Vec model showcasing how to convert text into vector representations using the skip-gram technique. This project aims to explore word embeddings and the training process of how machines begin to form an understanding of the semantics of words through rich meaningful embeddings. It is inspired by and closely follows the NLP research developed by Tomáš Mikolov, Kai Chen, Greg Corrado, Ilya Sutskever and Jeff Dean at Google.

### Skip-Gram
This is a technique for creating word embeddings that focuses on predicting surrounding words (the "context words") based on a specific word (the "target word").

<img src="https://github.com/melonknee/word2vec/blob/readme-updates/docs/images/skipgram.png" alt="Structure of the Skip-Gram Model" style="width:50%; height:auto;">![Skip-Gram IO](https://github.com/melonknee/word2vec/blob/readme-updates/docs/images/skipgram_io.png)
The left diagram is extracted from the research paper ["Distributed Representations fo Words and Phrases and their Compositionality"](https://arxiv.org/pdf/1310.4546), and the right diagram is a simplified version I created to show an example. In both diagrams, the window size has been set to 2: the window size determines the number of context words that should be considered before and after the target word (how many context words to the left and the right of the target word to predict).

Imagine you're reading a sentence and can guess the words (output) that come before and after a particular word (input).
For example: in a sentence "I found sunny days again", using the input "sunny", this model may output an example of the surrounding words of ["I", "found", "days", "again"].

![Skip-Gram Example](https://github.com/melonknee/word2vec/blob/readme-updates/docs/images/skipgram_eg.png)


With all these pairs of target-context words generated from the text corpus, we can use them to train our Word2Vec model.

## Word2Vec
Word2Vec is a simple neural network with a single hidden layer. The initial weights are randomly generated and then updated with the iterations of training with the goal of reducing the loss function. The final output are the weights of the network which are then used as word vectors (embeddings) which can further be used in other tasks.

## What do these "embeddings" actually mean?
The goal of all of this is to create word embeddings that can somehow encapsulate semantic and syntactic relationships in the English language.
A visual representation of vector relationships in word embeddings.

<img src="https://github.com/melonknee/word2vec/blob/readme-updates/docs/images/gender_forms.png" alt="Gender Form Pairs" style="width:50%; height:auto;"><img src="https://github.com/melonknee/word2vec/blob/readme-updates/docs/images/verb_forms.png" alt="Verb Form Pairs" style="width:40%; height:auto;">
## Stack
Python, Jupyter Notebook, Matplotlib, sci-kit learn, NumPy, PyTorch

## Embedding Techniques
To improve the efficiency and quality of the word represenations, I implemented techniques like subsampling and negative sampling (inspired by the ["Distributed Representations fo Words and Phrases and their Compositionality"](https://arxiv.org/pdf/1310.4546) paper.
### Subsampling

 # Results
## Visualisations
### t-SNE (t-Distributed Stochastic Neighbour Embedding)
This non-linear technique is used for visualising high-dimensional data. It converts high-dimensional distances between data points into probabilities, then creating a low-dimensional map where similar high-dimensional points are modeled as nearby points in the low-dimensional space.

![Displaying the clustering of words in similar topics](https://github.com/melonknee/word2vec/blob/readme-updates/docs/images/vis_t-sne_categorical_clustering.png)
### PCA (Principal Component Analysis)
This linear technique reduces the dimensionality of the embeddings, whilst preserving global data structures and variance, allowing us to create visualisations that show interpretable, deterministic results.
It is less effective for non-linear data, hence why I used it to explore the arithmetic relationship of the words.

## Statistics
### Most Similar Words


## Limitations
### Polysemy
Some words may have multiple meanings e.g. coach = big bus||to teach someone||the brand "Coach". In this model, each word only has one embedding to represent it, meaning it may not be able to encapsulate these multiple meanings
### Out of Vocabulary (OOV) Words
If a word is not
### Extensions
At the end of the day, this version of Word2Vec is just a very simple static word embedding model. In more complex applications, contextualised word embeddings are usually used, as they provide dynamic, context-aware representations. These


## Set Up

[comment]: <> (uv)

## Training the Model
