# Fake-News-Detection-System

📰 Fake News Detection System

Fake News Detection System is an intelligent machine learning project designed to identify whether a news article is real or fake.
Using advanced NLP techniques and multiple classification models, this tool analyzes text patterns, language cues, and semantics to detect misinformation efficiently.

📋 Project Overview

The Fake News Detection System processes real-world news datasets to help users:

Detect fake or misleading news articles.

Understand how ML models can classify text-based data.

Visualize results through confusion matrices and performance metrics.

This project aims to promote media literacy and demonstrate the power of natural language processing in combating fake news.

🔑 Features

1️⃣ Text Preprocessing

Cleans and normalizes text data (lowercasing, stopword removal, lemmatization).

Removes URLs, HTML tags, and unwanted characters.

2️⃣ Model Training & Evaluation

Trains three ML models: Logistic Regression, Naive Bayes, and Linear SVM.

Evaluates models with accuracy, precision, recall, and F1-score metrics.

Visualizes confusion matrices for better understanding.

3️⃣ Prediction & Export

Predicts whether new text is real or fake.

Saves best model and vectorizer using Joblib for future use.

🛠️ Tech Stack
Component	Technology
Language	Python
Libraries	scikit-learn, NLTK, Pandas, NumPy
Visualization	Matplotlib, Seaborn
Model Storage	Joblib
📊 How It Works

1️⃣ Data Loading: Reads and merges Fake.csv and True.csv datasets.
2️⃣ Preprocessing: Cleans and tokenizes news content.
3️⃣ Feature Extraction: Converts text into TF-IDF vectors.
4️⃣ Model Training: Trains multiple models and evaluates performance.
5️⃣ Prediction: Classifies unseen news as Real or Fake.

🙌 Contributing

Contributions are welcome!

Fork this repository

Create a new branch for your feature

Submit a pull request describing your improvements
