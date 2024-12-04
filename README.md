# Sentiment-Analysis-Chatbot

This project implements a chatbot that predicts the sentiment of a user's input text. It categorizes the sentiment into Positive, Negative, or Neutral using natural language processing (NLP) techniques and machine learning models. The chatbot leverages advanced text preprocessing, word embeddings, and an LSTM (Long Short-Term Memory) neural network to achieve accurate sentiment predictions.

Key Features
Text Preprocessing:

Cleans text by removing digits, URLs, mentions, hashtags, special characters, and HTML tags.
Expands contractions (e.g., "don't" → "do not").
Normalizes elongated words (e.g., "soooo" → "soo").
Handles negations in sentences for better sentiment understanding.
Retains only relevant parts of speech (nouns, adjectives, verbs, and adverbs).
Filters out non-English words for language consistency.


Word Embedding:

Uses Word2Vec to generate vector representations of words, capturing their semantic context.


Sentiment Categorization:

Assigns numerical labels: Positive (1), Neutral (0), and Negative (-1).


Machine Learning Model:

Employs an LSTM neural network with pre-trained word embeddings for sequential data analysis.
Trains the model using Keras with a focus on accuracy and efficiency.


User Interaction:

Includes a Streamlit interface for easy interaction.
Provides real-time sentiment analysis for user inputs.
