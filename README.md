# NLP_Email_Verifications
This project demonstrates how to build a Spam Message Detector using Natural Language Processing (NLP) and Machine Learning with Python. The model classifies text messages (SMS) as either Spam or Ham (Not Spam) using a Multinomial Naive Bayes classifier.


It uses the SMS Spam Collection Dataset, applies text preprocessing (tokenization, stopword removal, TF-IDF transformation), and visualizes data distributions.

🧠 Project Overview

The goal of this project is to automatically identify whether an SMS message is spam or not using Machine Learning and Text Mining techniques.

We:

Load and explore the dataset

Preprocess text (cleaning, removing punctuation & stopwords)

Convert text into numerical features using Bag of Words (BoW) and TF-IDF

Train a Naive Bayes classifier

Evaluate the model using precision, recall, and F1-score

📂 Dataset Information

Dataset Name: SMS Spam Collection

Source: UCI Machine Learning Repository

File Name: SMSSpamCollection

Format: Tab-separated (label \t message)

Labels:

ham → Not Spam

spam → Spam message

Example:

ham   Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam  Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005...

⚙️ Installation & Setup
1. Clone this Repository
git clone https://github.com/your-username/sms-spam-detection.git
cd sms-spam-detection

2. Install Dependencies

Make sure you have Python 3.x installed.
Install the required libraries using pip:

pip install numpy pandas matplotlib scikit-learn nltk

3. Download NLTK Data

Open Python shell and run:

import nltk
nltk.download('stopwords')

4. Place Dataset

Place the SMSSpamCollection file in your desired directory (update the path in your code).

🧩 Project Workflow
1. Import Libraries

Essential Python libraries for data analysis, visualization, and NLP are imported:

import nltk, pandas as pd, numpy as np, matplotlib.pyplot as plt, string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords

2. Load and Explore Dataset

Reads the SMS dataset from a text file.

Displays message count and sample entries.

Converts it into a Pandas DataFrame for easier manipulation.

msgs = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
print(msgs.head())
print(msgs.groupby('label').describe())

3. Data Visualization

Plots the distribution of message lengths.

Compares spam vs ham message lengths.

msgs['length'].plot.hist(bins=70)
msgs.hist(column='length', by='label', bins=60, figsize=(12,4))
plt.show()

4. Text Preprocessing

A function is created to:

Remove punctuation

Remove stopwords

Tokenize text

def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

5. Feature Extraction

CountVectorizer: Converts messages into a Bag of Words matrix.

TF-IDF Transformer: Normalizes the frequency counts.

bow_trans = CountVectorizer(analyzer=text_process).fit(msgs['message'])
msg_bow = bow_trans.transform(msgs['message'])
tfi = TfidfTransformer().fit(msg_bow)
msg_tfidf = tfi.transform(msg_bow)

6. Model Training (Naive Bayes)

Train a Multinomial Naive Bayes classifier using TF-IDF features:

spam_detect_model = MultinomialNB().fit(msg_tfidf, msgs['label'])

7. Model Evaluation

Split the dataset into training and testing sets:

msg_train, msg_test, label_train, label_test = train_test_split(
    msgs['message'], msgs['label'], test_size=0.3, random_state=42)


Build a pipeline for automation:

pipe = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])
pipe.fit(msg_train, label_train)
pred = pipe.predict(msg_test)
print(classification_report(label_test, pred))

📊 Results
Metric	Ham	Spam
Precision	~0.98	~0.96
Recall	~0.99	~0.93
F1-score	~0.99	~0.94

✅ The model performs very well with high accuracy and precision, effectively distinguishing spam messages.

📘 Concepts Used

Text Preprocessing: Removing punctuation and stopwords

Bag of Words (BoW): Representing text as numerical word frequencies

TF-IDF (Term Frequency–Inverse Document Frequency): Weighting words based on importance

Naive Bayes Classifier: Probabilistic model for text classification

Pipeline in scikit-learn: Automates feature extraction and model training

📈 Visualizations

Distribution of message lengths

Comparison between spam and ham message lengths

These visualizations help understand how spam messages differ in structure and size.

🚀 Future Improvements

Use Word Embeddings (Word2Vec, GloVe, BERT) for richer representations

Try Deep Learning models like LSTM or BERT-based classifiers

Build a Web Interface (e.g., Flask/Streamlit app) for real-time SMS classification

🧑‍💻 Author
Danish Khatri
📘 Student & Machine Learning Enthusiast
