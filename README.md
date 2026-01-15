SMS Spam Detection System
This project implements a Machine Learning model to classify SMS messages as either Spam or Ham (not spam). The project involves data preprocessing, exploratory data analysis (EDA), and training a classifier using Natural Language Processing (NLP) techniques.

Table of Contents
Project Overview

Dataset

Installation

Project Steps

Data Cleaning

Exploratory Data Analysis

Data Preprocessing

Model Building

Results

Project Overview
The goal of this project is to build a reliable classifier that can identify spam messages with high precision. By using the ExtraTreesClassifier, the model effectively filters out unwanted messages while minimizing the false detection of legitimate texts.

Dataset
The dataset used is sms-spam.csv, which contains 5,572 rows of SMS messages.

v1: The label (ham/spam).

v2: The raw text of the SMS message.

Installation
To run this notebook, you need the following Python libraries:

Bash

pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud
Project Steps
Data Cleaning
Removed unnecessary columns (Unnamed: 2, Unnamed: 3, Unnamed: 4).

Renamed columns for better readability: result and input.

Encoded target labels: 0 for Ham and 1 for Spam.

Removed duplicate records, reducing the dataset from 5,572 to 5,169 rows.

EDA (Exploratory Data Analysis)
Analyzed the distribution of classes: ~12.6% Spam and ~87.4% Ham.

Engineered new features using nltk:

Number of characters in the message.

Number of words.

Number of sentences.

Visualized the differences between Spam and Ham messages using histograms and pie charts, showing that spam messages generally contain more characters and words.

Data Preprocessing
Converted text to lowercase.

Tokenized the text into individual words.

Removed special characters, punctuation, and stop words.

Applied Porter Stemming to reduce words to their root forms.

Converted processed text into numerical vectors using TfidfVectorizer.

Model Building
The project evaluated several algorithms, including:

Naive Bayes (Gaussian, Multinomial, Bernoulli)

ExtraTreesClassifier (ETC)

Based on evaluation metrics (Accuracy and Precision), the ExtraTreesClassifier was chosen as the final model.

Results
The model and the vectorizer are saved as .pkl files for deployment:

model.pkl: The trained ExtraTreesClassifier.

vectorizer.pkl: The TfidfVectorizer instance used during training.
