import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample citizen inquiries
citizen_inquiries = [
    "How do I apply for a building permit?",
    "What are the requirements for a business license?",
    "Is there any financial aid available for small businesses?"
]

# Responses from the government service
government_responses = [
    "To apply for a building permit, please visit our website and fill out the online form.",
    "Requirements for a business license include...",
    "Yes, there are various financial aid programs available for small businesses. Please visit our financial aid page for more information."
]

# Tokenization and TF-IDF vectorization
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

citizen_inquiries_preprocessed = [preprocess_text(inquiry) for inquiry in citizen_inquiries]
government_responses_preprocessed = [preprocess_text(response) for response in government_responses]

tfidf_vectorizer = TfidfVectorizer()
citizen_inquiries_tfidf = tfidf_vectorizer.fit_transform(citizen_inquiries_preprocessed)
government_responses_tfidf = tfidf_vectorizer.transform(government_responses_preprocessed)

# Matching citizen inquiries with government responses
def get_most_similar_response(user_inquiry):
    user_inquiry_preprocessed = preprocess_text(user_inquiry)
    user_inquiry_tfidf = tfidf_vectorizer.transform([user_inquiry_preprocessed])
    
    similarities = cosine_similarity(user_inquiry_tfidf, government_responses_tfidf)
    most_similar_index = np.argmax(similarities)
    
    return government_responses[most_similar_index]

# Example usage
user_inquiry = "How do I apply for a building permit?"
response = get_most_similar_response(user_inquiry)
print(response)