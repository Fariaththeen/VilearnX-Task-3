# VilearnX-Task-3
This project analyzes movie reviews using NLP to classify sentiments as positive or negative. It includes data cleaning, TF-IDF feature extraction, and a Naive Bayes classifier, along with visualizations to illustrate sentiment distribution and trends.


# Sentiment Analysis of Movie Reviews

This project performs sentiment analysis on a dataset of movie reviews to classify them as positive or negative. The code is written in Python and utilizes various libraries for data preprocessing, feature extraction, model training, and visualization.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Extraction](#feature-extraction)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Visualizations](#visualizations)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Sentiment analysis is a crucial task in natural language processing (NLP) that aims to determine the sentiment expressed in a piece of text, such as positive, negative, or neutral. This project focuses on classifying movie reviews as either positive or negative using machine learning techniques.

## Dataset
The dataset used in this project is the IMDb Movie Reviews dataset, which contains 50,000 highly polar movie reviews for training, and 50,000 for testing. The dataset is available on [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Installation
To run this project, you need to have Python and the following libraries installed:
- pandas
- nltk
- scikit-learn
- matplotlib
- seaborn
- wordcloud

You can install these libraries using pip:
```bash
pip install pandas nltk scikit-learn matplotlib seaborn wordcloud