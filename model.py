import json
import random
import nltk
import numpy as np
import os
import logging
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
def download_nltk_data():
    try:
        for resource in ['punkt', 'stopwords', 'wordnet']:
            nltk.download(resource, quiet=True)
        logger.info("NLTK data downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        raise

download_nltk_data()

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load intents
def load_intents():
    try:
        with open("intents.json", "r", encoding='utf-8') as file:
            data = json.load(file)
        logger.info(f"Loaded intents.json with {len(data['intents'])} intents.")
        return data
    except FileNotFoundError:
        logger.error("intents.json not found in project directory.")
        raise
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in intents.json.")
        raise

data = load_intents()

# Preprocessing function
def preprocess_text(text):
    """Preprocess text using NLTK: tokenize, lemmatize, remove stopwords."""
    try:
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in stop_words]
        return " ".join(tokens)
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return text.lower()

# Train and save model
def train_model():
    """Train MultinomialNB model and save to model.pkl and vectorizer.pkl."""
    logger.info("Starting model training...")
    patterns = []
    tags = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            processed = preprocess_text(pattern)
            if processed:  # Skip empty patterns
                patterns.append(processed)
                tags.append(intent["tag"])

    if not patterns:
        logger.error("No valid patterns found for training.")
        raise ValueError("No valid patterns for training.")

    # Vectorize and split data
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(patterns).toarray()
    y = np.array(tags)

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model trained with accuracy: {accuracy:.4f}")

    # Save model and vectorizer
    try:
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info("Model and vectorizer saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save model/vectorizer: {e}")
        raise

    return model, vectorizer

# Load or train model
def load_or_train_model():
    """Load model and vectorizer if they exist, otherwise train and save."""
    model_path = "model.pkl"
    vectorizer_path = "vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)
            logger.info("Loaded existing model and vectorizer.")
            return model, vectorizer
        except Exception as e:
            logger.warning(f"Failed to load model/vectorizer: {e}. Retraining model.")
    
    return train_model()

# Load model and vectorizer
model, vectorizer = load_or_train_model()

# Initialize OpenAI client
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.warning("OPENAI_API_KEY not found in .env. Falling back to local model only.")
    openai.api_key = None

def chatbot_reply(user_msg):
    """Generate a response for user input using OpenAI API or local model."""
    if not user_msg or not isinstance(user_msg, str):
        logger.warning("Invalid user message received.")
        return "Please enter a valid message."

    # Preprocess user input
    processed_msg = preprocess_text(user_msg)
    if not processed_msg:
        logger.warning("Empty processed message after preprocessing.")
        return "I didn't understand that. Can you rephrase?"

    # Try OpenAI API
    if openai.api_key:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use gpt-4o if your plan supports it
                messages=[
                    {"role": "system", "content": "You are a helpful chatbot assistant created by Swalha Sakeer, a Data Engineering student. Provide concise, friendly responses about AI, data engineering, or Swalha's work. If unsure, suggest rephrasing."},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=50,
                temperature=0.7
            )
            api_reply = response.choices[0].message['content'].strip()
            logger.info("OpenAI API response received.")
            return api_reply
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")

    # Fallback to local model
    try:
        user_vec = vectorizer.transform([processed_msg]).toarray()
        predicted_tag = model.predict(user_vec)[0]
        for intent in data["intents"]:
            if intent["tag"] == predicted_tag:
                response = random.choice(intent["responses"])
                logger.info(f"Local model predicted intent: {predicted_tag}")
                return response
        logger.warning(f"No matching intent found for predicted tag: {predicted_tag}")
    except Exception as e:
        logger.error(f"Error in local model prediction: {e}")
    
    return "I didn't understand that. Can you rephrase?"

if __name__ == "__main__":
    # Test the model with sample inputs
    test_inputs = ["Hello", "Tell me about Swalha", "What is AI?", "Tell me a joke"]
    for user_msg in test_inputs:
        response = chatbot_reply(user_msg)
        print(f"Input: {user_msg}\nResponse: {response}\n")