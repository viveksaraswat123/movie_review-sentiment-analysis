# Movie Review Sentiment Analysis

A machine learning project that classifies movie reviews into positive or negative sentiments using Natural Language Processing (NLP) techniques and logistic regression.

---

## Project Overview

Sentiment analysis is a popular NLP task that involves classifying text based on the expressed sentiment. This project focuses on analyzing movie reviews from the IMDB dataset to determine whether a review is positive or negative.

The model leverages TF-IDF vectorization to convert text into meaningful numerical features and uses Logistic Regression for classification. The program also provides an interactive command-line interface to test the model with custom reviews.

---

## Features

- **Data Preprocessing:** Cleans text by removing HTML tags, punctuation, and converting to lowercase.
- **Vectorization:** Uses TF-IDF to represent text data efficiently.
- **Classification:** Implements Logistic Regression to classify sentiments.
- **Model Evaluation:** Outputs accuracy and detailed classification reports on test data.
- **Interactive Testing:** Allows users to input their own movie reviews for sentiment prediction.

---

## Dataset

The project uses the [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), which contains 50,000 highly polarized movie reviews split evenly into positive and negative categories.

---

## Installation and Setup

1. **Clone the repository**

git clone https://github.com/viveksaraswat123/movie_review-sentiment-analysis.git
cd movie_review-sentiment-analysis
Create a virtual environment (optional but recommended)

bash
Copy code
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

Install dependencies - pip install -r requirements.txt
Download the dataset

Download the IMDB Dataset.csv file from here and place it in the project directory.

# Usage
Run the main script: python main.py
The program will:

# Train the logistic regression model on the dataset.

# Output the model accuracy and classification report.

# Allow you to enter your own movie reviews and predict their sentiment interactively.

# Type exit to quit the interactive mode.


Model Performance
The logistic regression model with TF-IDF features achieves an accuracy of approximately 90-92% on the test set, providing reliable sentiment classification for movie reviews.

Future Improvements
# Incorporate advanced NLP models like LSTM, BERT for improved accuracy.

# Use more complex text preprocessing like lemmatization and stopword removal.

# Implement hyperparameter tuning for better model performance.

Add support for multi-class sentiment analysis (e.g., neutral class).

Build a web app or API to make the model accessible online.

Technologies Used
Python 3.x
pandas
scikit-learn
regex (re)

License
This project is open-source and available under the MIT License.

Contact
For questions or collaborations, feel free to reach out:

GitHub: viveksaraswat123
