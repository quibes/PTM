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
            if isinstance(data, list):  # Check if the data is a list
                for entry in data:
                    msg = entry.get('message', '')
                    if msg:
                        messages.append(msg)
            elif isinstance(data, dict):  # If it's a dictionary, process as before
                for key, value in data.items():
                    msg = value.get('message', '')
                    if msg:
                        messages.append(msg)
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
import pickle
with open('dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)
with open('corpus.pkl', 'wb') as f:
    pickle.dump(corpus, f)
with open('processed_texts.pkl', 'wb') as f:
    pickle.dump(processed_texts, f)

# Function to compute coherence score for different numbers of topics
# Function to compute coherence score for different numbers of topics
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print(f"Building LDA model with {num_topics} topics...")
        model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model,
                                        texts=texts,
                                        dictionary=dictionary,
                                        coherence='c_v')
        coherence = coherencemodel.get_coherence()  # Compute coherence score
        coherence_values.append(coherence)  # Append coherence to the list
        print(f"Coherence Score: {coherence:.4f}")
    return model_list, coherence_values


# Compute coherence scores
limit = 50
start = 2
step = 2
model_list, coherence_values = compute_coherence_values(dictionary=dictionary,
                                                        corpus=corpus,
                                                        texts=processed_texts,
                                                        start=start,
                                                        limit=limit,
                                                        step=step)

# Plot the coherence scores
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.title("Coherence Scores by Number of Topics")
plt.show()

# Find and print the optimal number of topics
if coherence_values:
    max_coherence = max(coherence_values)
    optimal_index = coherence_values.index(max_coherence)
    optimal_model = model_list[optimal_index]
    optimal_num_topics = start + optimal_index * step
    print(f"Optimal number of topics: {optimal_num_topics}")
    topics = optimal_model.print_topics(num_words=10)
    print("\nTopics discovered by the LDA model:")
    for topic_no, topic in topics:
        print(f"Topic {topic_no}: {topic}")

# Save the model
if 'optimal_model' in locals():
    optimal_model.save('lda_model_russian')


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
    #vis_data = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary, mds='mmds') 

    
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

