## Introduction

This project investigates two probabilistic topic modeling algorithms—LDA and BERTopic—to assess their effectiveness in handling thematic coherence, interpretability, and computational efficiency when applied to large-scale unstructured Russian-language text data from Telegram channels.

## Methodology

### Data Collection

Data was collected from three public sports-themed Russian-language Telegram chats over a one-month period (October 1 to November 1, 2024). The selected chats are:

1. **Матч! Чат**: Over 22,000 members, generating 3,000 to 10,000 messages daily.
2. **FEGURKA / Фигурное катание**: Over 2,200 members, producing about 3,000 messages daily.
3. **БАРСЕЛОНА | BARCELONA**: Over 22,500 members, with around 10,000 messages daily.

Data was fetched using the Telethon Python library, adhering to ethical research guidelines.

### Data Preprocessing

#### LDA Preprocessing

1. **Data Loading and Deduplication**: Loaded JSON-formatted messages and removed duplicates, capping at 300,000 messages.
2. **Stop Phrase and Noise Removal**: Excluded messages with domain-specific stop phrases and removed non-textual noise (e.g., URLs, mentions).
3. **Tokenization and Lemmatization**: Tokenized text, removed Russian stopwords, and lemmatized using `pymorphy2`.
4. **Data Normalization**: Converted text to lowercase and removed non-Cyrillic characters.
5. **Identification of Common Phrases**: Identified and ranked bigrams and trigrams.
6. **Dictionary Creation and Corpus Representation**: Created a Gensim dictionary, filtered extreme frequencies, and transformed the corpus into a bag-of-words representation.
7. **Coherence-Based Optimization of Topics**: Tested topic counts (2 to 40) and selected the model with the highest coherence score.

#### BERTopic Preprocessing

1. **Data Loading**: Loaded JSON data into a pandas DataFrame, extracting and filtering the 'message' field.
2. **Text Normalization**: Lowercased text, removed stop phrases, tokenized using `razdel`, and excluded non-alphabetic tokens.
3. **Lemmatization**: Applied `pymorphy2` for morphological normalization.
4. **Stopword Removal**: Excluded tokens in the Russian stopword list from NLTK and ensured a minimum token length of three characters.
5. **Quality Enhancement**: Removed English words and messages lacking meaningful content post-processing.

## Implementation

### LDA Implementation

1. **Dictionary and Corpus Preparation**: Created a dictionary of unique tokens and constructed a bag-of-words representation.
2. **Model Tuning**: Generated models with topic counts ranging from 2 to 50, evaluated using coherence scores, and selected the optimal model.
3. **Parameter Adjustment**: Set alpha and beta parameters to 'auto' for dynamic optimization.
4. **Evaluation**: Plotted coherence scores against topic counts and extracted top words for each topic.

### BERTopic Implementation

1. **Embeddings**: Generated dense vector representations using `paraphrase-multilingual-sbert_large_nlu_ru` and `MiniLM-L12-v2`.
2. **Dimensionality Reduction**: Applied UMAP with optimized parameters for Russian text data.
3. **Clustering**: Utilized HDBSCAN with parameters balancing topic granularity and cluster stability.
4. **Vectorization and Topic Representation**: Employed `CountVectorizer` and `ClassTFIDFTransformer` for interpretable topic representations.

## Results

The comparative analysis of LDA and BERTopic on the Russian-language Telegram dataset revealed insights into thematic coherence, interpretability, and computational efficiency. Detailed results are documented in the thesis.

## Repository Structure

- `Barcelona/`: Data and analysis related to the БАРСЕЛОНА | BARCELONA chat.
- `MatchTV/`: Data and analysis related to the Матч! Чат.
- `fegurka/`: Data and analysis related to the FEGURKA / Фигурное катание chat.
- `environment_info/`: Environment configuration and dependencies.
- `README.md`: This README file.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/quibes/PTM.git
   cd PTM
