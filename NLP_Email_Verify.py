# Import necessary libraries

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords

''''CountVectorizer – converts text into a matrix of word counts (how many times each word appears).
    TfidfTransformer – converts those word counts into weighted scores (TF-IDF) to show how important each word is in a message.'''


# Load the SMS Spam Collection dataset
msgs = [line.rstrip() for line in open(r'C:\Users\DANISH\Downloads\SMSSpamCollection')]
print("Total messages:", len(msgs))
print("Example message:\n", msgs[50])

# Print first 10 messages with numbering
for mess_no, msg in enumerate(msgs[:10]):
    print(mess_no, msg)
    print('\n')

# Read dataset as a DataFrame (tab-separated)
msgs = pd.read_csv(r'C:\Users\DANISH\Downloads\SMSSpamCollection',
                   sep='\t', names=['label', 'message'])

# Display first few rows
print(msgs.head())

# Summary of dataset
print(msgs.describe())

# Count of spam vs ham messages
print(msgs.groupby('label').describe())

# Add a new column for message length
msgs['length'] = msgs['message'].apply(len)
print(msgs.head())

# Visualize message lengths
msgs['length'].plot.hist(bins=70)
plt.title("Distribution of Message Lengths")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()

# Descriptive stats for message length
print(msgs['length'].describe())

# Find the longest message
print("Longest message:\n", msgs[msgs['length'] == 910]['message'].iloc[0])

# Compare message lengths between spam and ham
msgs.hist(column='length', by='label', bins=60, figsize=(12,4))
plt.show()

# Text Preprocessing Example
mess = 'Sample message! Notice: it has punctuation.'
print("All punctuations:", string.punctuation)

# Remove punctuation
nopunc = [c for c in mess if c not in string.punctuation]
print("Without punctuation:", nopunc)

# Join characters back into string
nopunc = ''.join(nopunc)

# Split into individual words
print("Split words:", nopunc.split())

# Remove stopwords (common words like 'the', 'and', etc.)
print("English stopwords:", stopwords.words('english'))
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
print("After removing stopwords:", clean_mess)

# Define text preprocessing function for entire dataset
def text_process(mess):
    """
    This function removes punctuation and stopwords from a message.
    Returns a list of cleaned words.
    """
    # Remove punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    # Remove stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Test the function on first 5 messages
print(msgs['message'].head(5).apply(text_process))

# Convert text messages into numerical vectors using CountVectorizer
bow_trans = CountVectorizer(analyzer=text_process).fit(msgs['message'])
print("Total unique words (vocabulary size):", len(bow_trans.vocabulary_))

# Example transformation on message 4
mess4 = msgs['message'][3]
print("Example message:\n", mess4)

bow4 = bow_trans.transform([mess4])
print("Bag of Words for message 4:\n", bow4)
print("Shape of this vector:", bow4.shape)

# Show the word corresponding to a specific index
print("Word at index 4068:", bow_trans.get_feature_names_out()[4068])

# Transform all messages into Bag of Words format
msg_bow = bow_trans.transform(msgs['message'])
print("Shape of sparse matrix:", msg_bow.shape)
print("Total non-zero values (nnz):", msg_bow.nnz)

# Normalization using TF-IDF (Term Frequency - Inverse Document Frequency)
tfi = TfidfTransformer().fit(msg_bow)
tfi4 = tfi.transform(bow4)

# Example: TF-IDF score for a word like 'university'
print("TF-IDF score for 'university':", tfi.idf_[bow_trans.vocabulary_['university']])

# Transform all messages using TF-IDF
msg_tfidf = tfi.transform(msg_bow)

# Train Naive Bayes classifier
spam_detect_model = MultinomialNB().fit(msg_tfidf, msgs['label'])

# Predict single example
print("Prediction for example 4:", spam_detect_model.predict(tfi4)[0])

# Predict all messages
all_pred = spam_detect_model.predict(msg_tfidf)
print("All predictions:", all_pred)

# Split dataset into training and test sets (70% train, 30% test)
msg_train, msg_test, label_train, label_test = train_test_split(
    msgs['message'], msgs['label'], test_size=0.3, random_state=42)

# Build a Pipeline (automates preprocessing + model training)
pipe = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # Step 1: Bag of Words
    ('tfidf', TfidfTransformer()),                    # Step 2: TF-IDF normalization
    ('classifier', MultinomialNB())                   # Step 3: Naive Bayes classifier
])

# Train the pipeline
pipe.fit(msg_train, label_train)

# Predict on test set
pred = pipe.predict(msg_test)

# Print classification report
print(classification_report(label_test, pred))
