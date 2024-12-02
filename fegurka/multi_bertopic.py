#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Add comments here to explain the code below
import pandas as pd
import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from nltk.corpus import stopwords
from bertopic import BERTopic
from gensim.models import CoherenceModel
from gensim import corpora
import re
import os
import pymorphy2  # Russian lemmatizer
from razdel import tokenize  # Better tokenization for Russian
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
# Load the JSON file
with open('/Users/anastasyarussu/Documents/thesis_hse/data+code/code/dataset_figurka_30k/figurka_30000_filtered_messages.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Convert JSON to DataFrame and extract the 'message' field
df = pd.DataFrame(data)
df = df[df['message'].notnull() & (df['message'] != '')]  # Filter non-null and non-empty messages

# Convert messages to a list for BERTopic
documents = df['message'].tolist()

# Verifications similar to the cookbook notebook
print(len(documents))  # Number of messages
print(type(documents))  # Data type
print(documents[:1])  # First instance of the content

# Load stop words for Russian
nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")

# Stop phrases to exclude
stop_phrases = [
    "у меня", "я не", "так и не", "до сих пор", "и не", "не знаю", "не будет",
    "о том что", "мороженое увеличил карму", "увеличил карму", "добавить карма", "карма", 
    'это', 'еще', 'ещё', 'очень', 'неа', 'есс', 'вообще', 'просто', 'вроде'
]

# Initialize Russian lemmatizer
morph = pymorphy2.MorphAnalyzer()

# Preprocess the dataset
filtered_text = []

for message in documents:
    # Convert to lowercase
    message = message.lower()

    # Remove stop phrases
    for phrase in stop_phrases:
        message = message.replace(phrase, "")

    # Tokenize the Russian text using razdel
    tokens = [token.text for token in tokenize(message)]

    # Remove punctuation and non-alphabetic tokens
    tokens = [re.sub(r'\W+', '', token) for token in tokens if token.isalpha()]

    # Lemmatize and filter out stopwords
    lemmatized_tokens = [
        morph.parse(token)[0].normal_form
        for token in tokens
        if token not in russian_stopwords and len(token) > 2
    ]

    lemmatized_message = " ".join(lemmatized_tokens)

    # Remove English words using regex
    lemmatized_message = re.sub(r'\b[A-Za-z]+\b', '', lemmatized_message)

    # Strip extra whitespace and append
    if lemmatized_message.strip():
        filtered_text.append(lemmatized_message.strip())

print(filtered_text[:1])  # Check the first preprocessed message

# Step 2.1 - Extract embeddings using a multilingual SentenceTransformer
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ... [Your other pipeline components remain unchanged] ...

print("Pipeline components are ready for BERTopic.")

# Define a range of topic numbers to evaluate
topic_numbers = [2, 4,6,8,10,20,30, 40,50]

coherence_scores_cv = []
coherence_scores_umass = []
irbo_scores = []
average_cosine_similarities = []

# Store topic-word distributions for IRBO computation
previous_topic_word_distributions = None
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    metric='cosine',
    random_state=42
)

# Initialize HDBSCAN for clustering
hdbscan_model = HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# Initialize the vectorizer for text representation
vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),  # Use unigrams and bigrams
    stop_words=stopwords.words('russian'),  # Include Russian stopwords
    max_df=0.99,  # Ignore terms that appear in more than 90% of documents
    min_df=2     # Ignore terms that appear in less than 5 documents
)
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
for nr_topics in tqdm(topic_numbers):
    # Create a new BERTopic model with the specified number of topics
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        nr_topics=nr_topics
    )
    
    # Fit the model to the filtered text
    topics, probabilities = topic_model.fit_transform(filtered_text)
    
    # Prepare documents for topic analysis
    documents = pd.DataFrame({
        "Document": filtered_text,
        "ID": range(len(filtered_text)),
        "Topic": topics
    })
    
    # Aggregate documents per topic for coherence evaluation
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)
    
    # Extract vectorizer and analyzer from BERTopic
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    
    # Tokenize the cleaned documents for coherence evaluation
    tokens = [analyzer(doc) for doc in cleaned_docs]
    
    # Create a dictionary and corpus for Gensim coherence evaluation
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    
    # Extract topic words for each topic
    topic_words = [
        [words for words, _ in topic_model.get_topic(topic)]
        for topic in range(len(set(topics)) - 1)
        if topic_model.get_topic(topic)  # Ensure the topic is not empty
    ]

    
    # Evaluate coherence using the 'c_v' metric
    coherence_model_cv = CoherenceModel(
        topics=topic_words,
        texts=tokens,
        corpus=corpus,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_cv = coherence_model_cv.get_coherence()
    coherence_scores_cv.append(coherence_cv)
    
    # Evaluate coherence using the 'u_mass' metric
    coherence_model_umass = CoherenceModel(
        topics=topic_words,
        texts=tokens,
        corpus=corpus,
        dictionary=dictionary,
        coherence='u_mass'
    )
    coherence_umass = coherence_model_umass.get_coherence()
    coherence_scores_umass.append(coherence_umass)
    
    # Calculate IRBO if previous topic-word distributions exist
    current_topic_word_distributions = topic_model.get_topic_info()
    if previous_topic_word_distributions is not None:
        # Compute IRBO between previous and current topic models
        # Here we use Jaccard similarity as a placeholder for IRBO computation
        # You may need to implement or use a library function for exact IRBO calculation
        overlap_scores = []
        for prev_topic_words in previous_topic_word_distributions['Name']:
            prev_words_set = set(prev_topic_words.split('_'))
            max_overlap = 0
            for curr_topic_words in current_topic_word_distributions['Name']:
                curr_words_set = set(curr_topic_words.split('_'))
                overlap = len(prev_words_set & curr_words_set) / len(prev_words_set | curr_words_set)
                max_overlap = max(max_overlap, overlap)
            overlap_scores.append(max_overlap)
        irbo_score = np.mean(overlap_scores)
        irbo_scores.append(irbo_score)
    else:
        irbo_scores.append(None)  # No IRBO score for the first iteration

    # Update previous topic-word distributions
    previous_topic_word_distributions = current_topic_word_distributions

    # Calculate average cosine similarity between topic embeddings
    topic_embeddings = topic_model.topic_embeddings_
    if topic_embeddings is not None and len(topic_embeddings) > 1:
        cosine_sim_matrix = cosine_similarity(topic_embeddings)
        # Exclude self-similarity by masking the diagonal
        np.fill_diagonal(cosine_sim_matrix, np.nan)
        average_cosine_similarity = np.nanmean(cosine_sim_matrix)
        average_cosine_similarities.append(average_cosine_similarity)
    else:
        average_cosine_similarities.append(None)

    print(f"nr_topics={nr_topics} => Coherence Score (c_v): {coherence_cv}, (u_mass): {coherence_umass}, IRBO: {irbo_scores[-1]}, Avg Cosine Similarity: {average_cosine_similarities[-1]}")

# Plot the coherence scores
plt.figure(figsize=(12, 6))
plt.plot([str(n) for n in topic_numbers], coherence_scores_cv, marker='o', label='c_v Coherence')
plt.plot([str(n) for n in topic_numbers], coherence_scores_umass, marker='x', label='u_mass Coherence')
plt.plot([str(n) for n in topic_numbers], irbo_scores, marker='s', label='IRBO Score')
plt.plot([str(n) for n in topic_numbers], average_cosine_similarities, marker='^', label='Avg Cosine Similarity')
plt.title('Evaluation Metrics for Different Numbers of Topics')
plt.xlabel('Number of Topics')
plt.ylabel('Score')
plt.legend()
plt.show()

