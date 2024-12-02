#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words
import pymorphy2
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import rbo  # Make sure to install this package
import numpy as np

# Initialize the lemmatizer
morph = pymorphy2.MorphAnalyzer()

# Path to the JSON file
json_file = '/Users/anastasyarussu/Documents/thesis_hse/data+code/code/dataset_figurka_30k/figurka_30000_filtered_messages.json'  # Replace with your actual file path

# Stop phrases to exclude
stop_phrases = [
    "у меня", "я не", "так и не", "до сих пор", "и не", "не знаю", "не будет", "о том что", 'мороженое увеличил карму', 'увеличил карму'
    
]

# Load the messages
def load_messages(json_file):
    messages = []
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                # Handle if the JSON is a list
                for item in data:
                    msg = item.get('message', '')
                    if msg:
                        messages.append(msg)
            elif isinstance(data, dict):
                # Handle if the JSON is a dictionary
                for key, value in data.items():
                    msg = value.get('message', '')
                    if msg:
                        messages.append(msg)
            else:
                print("Unsupported JSON structure. Expected a list or dictionary.")
        print(f"Total messages loaded: {len(messages)}")
    except FileNotFoundError:
        print(f"The file {json_file} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    return messages

# Remove stop phrases from messages
def filter_stop_phrases(messages, stop_phrases):
    filtered_messages = []
    for message in messages:
        if not any(stop_phrase in message for stop_phrase in stop_phrases):
            filtered_messages.append(message)
    return filtered_messages

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text)  # Keep only Cyrillic characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    tokens = word_tokenize(text, language='russian')  # Tokenize
    stop_words = set(get_stop_words('russian'))  # Get Russian stopwords
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [morph.normal_forms(word)[0] for word in tokens]  # Lemmatize
    return tokens

# Analyze messages for most common phrases
def get_most_frequent_phrases(messages, n=2, top_k=10):
    all_ngrams = Counter()
    for message in messages:
        preprocessed = preprocess_text(message)
        n_grams = nltk.ngrams(preprocessed, n)
        all_ngrams.update(n_grams)
    return all_ngrams.most_common(top_k)

# Function to convert document topics to vector
def topics_to_vector(doc_topics, num_topics):
    vector = [0] * num_topics
    for topic_num, prob in doc_topics:
        vector[topic_num] = prob
    return vector

# Main processing
messages = load_messages(json_file)

# Remove messages with stop phrases
filtered_messages = filter_stop_phrases(messages, stop_phrases)
print(f"Total messages after stop phrase removal: {len(filtered_messages)}")

# Preprocess text for LDA
processed_texts = [preprocess_text(msg) for msg in filtered_messages]

# Most frequent bigrams
most_frequent_bigrams = get_most_frequent_phrases(filtered_messages, n=2, top_k=10)
print("\nMost Frequent Phrases (Bigrams):")
for phrase, count in most_frequent_bigrams:
    print(f"{' '.join(phrase)}: {count}")

# Most frequent trigrams
most_frequent_trigrams = get_most_frequent_phrases(filtered_messages, n=3, top_k=10)
print("\nMost Frequent Phrases (Trigrams):")
for phrase, count in most_frequent_trigrams:
    print(f"{' '.join(phrase)}: {count}")

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(processed_texts)
print(f"Number of unique tokens before filtering: {len(dictionary)}")

# Filter extremes to remove very common and very rare words
dictionary.filter_extremes(no_below=15, no_above=0.5)
print(f"Number of unique tokens after filtering: {len(dictionary)}")

# Create a bag-of-words representation of the documents
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# Save the dictionary and corpus for later use
with open('dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)
with open('corpus.pkl', 'wb') as f:
    pickle.dump(corpus, f)
with open('processed_texts.pkl', 'wb') as f:
    pickle.dump(processed_texts, f)

# Split the data into training and test sets
corpus_train, corpus_test, texts_train, texts_test = train_test_split(
    corpus, processed_texts, test_size=0.2, random_state=42
)

# Function to compute metrics
def compute_metrics(dictionary, corpus_train, corpus_test, texts_train, texts_test, limit, start=2, step=3):
    coherence_values = []
    perplexity_values = []
    inverted_rbo_values = []
    cosine_sim_values = []
    model_list = []
    topics_list = []
    doc_topic_dists = []

    for num_topics in range(start, limit, step):
        print(f"\nBuilding LDA model with {num_topics} topics...")
        model = LdaModel(
            corpus=corpus_train,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True,
        )
        model_list.append(model)

        # Compute Coherence Score
        coherencemodel = CoherenceModel(
            model=model, texts=texts_train, dictionary=dictionary, coherence='c_v'
        )
        coherence = coherencemodel.get_coherence()
        coherence_values.append(coherence)
        print(f"Coherence Score: {coherence:.4f}")

        # Compute Perplexity on test set
        perplexity = model.log_perplexity(corpus_test)
        perplexity_values.append(perplexity)
        print(f"Perplexity: {perplexity:.4f}")

        # Store topics for RBO computation
        current_topics = []
        for topic_id in range(num_topics):
            terms = [word for word, prob in model.show_topic(topic_id, topn=10)]
            current_topics.append(terms)
        topics_list.append(current_topics)

        # Compute document-topic distributions for test set
        doc_topics_test = [model.get_document_topics(bow, minimum_probability=0) for bow in corpus_test]
        doc_topic_dists.append(doc_topics_test)

    # Compute Inverted RBO between models
    inverted_rbo_values.append(None)  # First model has no previous model to compare
    for i in range(1, len(topics_list)):
        previous_topics = topics_list[i - 1]
        current_topics = topics_list[i]
        total_rbo = 0
        num_pairs = min(len(previous_topics), len(current_topics))
        for j in range(num_pairs):
            rbo_val = rbo.RankingSimilarity(previous_topics[j], current_topics[j]).rbo()
            total_rbo += rbo_val
        avg_rbo = total_rbo / num_pairs
        inverted_rbo = 1 - avg_rbo
        inverted_rbo_values.append(inverted_rbo)
        print(
            f"Inverted RBO between models with {start + (i - 1) * step} and {start + i * step} topics: {inverted_rbo:.4f}"
        )

    # Compute Cosine Similarity between document-topic distributions across models
    cosine_sim_values.append(None)  # First model has no previous model to compare
    for i in range(1, len(doc_topic_dists)):
        num_docs = len(doc_topic_dists[i])
        num_topics_prev = start + (i - 1) * step
        num_topics_curr = start + i * step

        # Convert doc-topic distributions to vectors
        vectors_prev = [
            topics_to_vector(doc, num_topics_prev) for doc in doc_topic_dists[i - 1]
        ]
        vectors_curr = [
            topics_to_vector(doc, num_topics_curr) for doc in doc_topic_dists[i]
        ]

        # Pad vectors to the same length
        max_topics = max(num_topics_prev, num_topics_curr)
        vectors_prev_padded = [
            v + [0] * (max_topics - len(v)) for v in vectors_prev
        ]
        vectors_curr_padded = [
            v + [0] * (max_topics - len(v)) for v in vectors_curr
        ]

        # Compute cosine similarity between corresponding documents
        cosine_sim = [
            cosine_similarity([vectors_prev_padded[j]], [vectors_curr_padded[j]])[0][0]
            for j in range(num_docs)
        ]
        avg_cosine_sim = np.mean(cosine_sim)
        cosine_sim_values.append(avg_cosine_sim)
        print(
            f"Cosine Similarity between models with {start + (i - 1) * step} and {start + i * step} topics: {avg_cosine_sim:.4f}"
        )

    return model_list, coherence_values, perplexity_values, inverted_rbo_values, cosine_sim_values

# Compute metrics
limit = 50
start = 2
step = 2
(
    model_list,
    coherence_values,
    perplexity_values,
    inverted_rbo_values,
    cosine_sim_values,
) = compute_metrics(
    dictionary=dictionary,
    corpus_train=corpus_train,
    corpus_test=corpus_test,
    texts_train=texts_train,
    texts_test=texts_test,
    start=start,
    limit=limit,
    step=step,
)

# Plot the metrics
x = range(start, limit, step)

# Coherence Score Plot
plt.figure(figsize=(10, 6))
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score (c_v)")
plt.title("Coherence Scores by Number of Topics")
plt.show()

# Perplexity Plot
plt.figure(figsize=(10, 6))
plt.plot(x, perplexity_values)
plt.xlabel("Number of Topics")
plt.ylabel("Perplexity")
plt.title("Perplexity by Number of Topics")
plt.show()

# Inverted RBO Plot
plt.figure(figsize=(10, 6))
plt.plot(x, inverted_rbo_values)
plt.xlabel("Number of Topics")
plt.ylabel("Inverted RBO")
plt.title("Inverted RBO by Number of Topics")
plt.show()

# Cosine Similarity Plot
plt.figure(figsize=(10, 6))
plt.plot(x, cosine_sim_values)
plt.xlabel("Number of Topics")
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity by Number of Topics")
plt.show()

# Find and print the optimal number of topics based on Coherence Score
if coherence_values:
    max_coherence = max(coherence_values)
    optimal_index = coherence_values.index(max_coherence)
    optimal_model = model_list[optimal_index]
    optimal_num_topics = start + optimal_index * step
    print(f"Optimal number of topics based on Coherence Score: {optimal_num_topics}")
    topics = optimal_model.print_topics(num_words=10)
    print("\nTopics discovered by the LDA model:")
    for topic_no, topic in topics:
        print(f"Topic {topic_no}: {topic}")

# Save the optimal model with user-specified name
if 'optimal_model' in locals():
    model_name = input("Enter the name to save the LDA model: ")
    optimal_model.save(model_name)
    print(f"Model saved as {model_name}")


# In[ ]:


# Coherence Score Plot
plt.figure(figsize=(10, 6))
plt.ylim([0,1])
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score (c_v)")
plt.title("Coherence Scores by Number of Topics (fegurka, 30k)")
plt.show()

# Perplexity Plot
plt.figure(figsize=(10, 6))
plt.plot(x, perplexity_values)
plt.xlabel("Number of Topics")
plt.ylabel("Perplexity")
plt.title("Perplexity by Number of Topics (fegurka, 30k)")
plt.show()

# Inverted RBO Plot
plt.figure(figsize=(10, 6))
plt.ylim([0,1])
plt.plot(x, inverted_rbo_values)
plt.xlabel("Number of Topics")
plt.ylabel("Inverted RBO")
plt.title("Inverted RBO by Number of Topics (fegurka, 30k)")
plt.show()

# Cosine Similarity Plot
plt.figure(figsize=(10, 6))
plt.ylim([0,1])
plt.plot(x, cosine_sim_values)
plt.xlabel("Number of Topics")
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity by Number of Topics (fegurka, 30k)")
plt.show()


# In[ ]:


import pyLDAvis
import pyLDAvis.gensim_models as gensimvis  # Module for Gensim
import pickle
from gensim.models import LdaModel

# Generate the pyLDAvis visualization for the LDA model
def generate_lda_visualization(lda_model, corpus, dictionary, output_html='lda_vis.html'):
    """
    Generates a pyLDAvis interactive visualization and saves it as an HTML file.
    
    Parameters:
    - lda_model: The trained LDA model
    - corpus: The corpus used for the LDA model
    - dictionary: The dictionary used for the LDA model
    - output_html: Path to save the HTML file with the visualization
    """
    # Prepare the visualization
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
    
    # Save to an HTML file
    pyLDAvis.save_html(vis_data, output_html)
    print(f"Interactive LDA visualization saved to {output_html}")

    # Optionally display in a notebook (if running in Jupyter)
    return vis_data

# Ensure an optimal model is available
if 'optimal_model' in locals():
    # Generate and display the visualization
    lda_vis = generate_lda_visualization(optimal_model, corpus, dictionary)
    
    # To view in a notebook, uncomment the line below
    pyLDAvis.display(lda_vis)
else:
    print("No optimal model available. Please ensure the LDA model is trained.")

