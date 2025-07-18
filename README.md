#📰 Fake News Classifier using NLP & Machine Learning
This project builds a Fake News Classifier using a real-world dataset and applies Natural Language Processing (NLP) techniques with three ML models: Multinomial Naive Bayes, Bernoulli Naive Bayes, and Logistic Regression. The final model can predict if a news article is real or fake.

📂 Dataset
Source: WELFake Dataset on Kaggle

Columns:

title: News title

text: News body

label: 1 for real news, 0 for fake

🧰 Libraries Used
pandas, numpy

seaborn, matplotlib, plotly

nltk, wordcloud

scikit-learn: TfidfVectorizer, train_test_split, classifiers, evaluation metrics

📊 Workflow
Data Loading & Cleaning

Loaded WELFake_Dataset.csv

Handled nulls, dropped unnamed columns

Visualization

Pie chart for real vs fake distribution using Plotly

WordClouds for both classes

Text Preprocessing

Lowercasing, punctuation removal, tokenization

Stopword removal

Lemmatization using WordNetLemmatizer

Feature Extraction

TF-IDF vectorization of preprocessed text

Model Training

Train/test split (80/20)

Trained:

Multinomial Naive Bayes

Bernoulli Naive Bayes

Logistic Regression

Evaluation

Accuracy, Confusion Matrix

ROC Curve using RocCurveDisplay

User Input Prediction

Custom input text passed for prediction using trained model

Model Saving

Final model saved for future inference

🧪 Sample Prediction Code
python
Copy
Edit
text = "Breaking news: Scientists discover new AI method to prevent fake news."
processed = preprocess_text(text)
vectorized = tfidf_vectorizer.transform([processed])
print(model.predict(vectorized))
💾 Files
Fake_news_classifier_.ipynb – main notebook

WELFake_Dataset.csv – dataset (external from Kaggle)

model.pkl – (if saved) trained model file

🚀 How to Run
Clone the repo

Install dependencies: pip install -r requirements.txt

Run the notebook: Fake_news_classifier_.ipynb
