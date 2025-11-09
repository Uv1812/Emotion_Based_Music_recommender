import logging
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import os
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("train_model.log", encoding="utf-8"),
    logging.StreamHandler()
])

def train_emotion_model():
    try:
        logging.info("Loading emotion dataset from data/emotion_dataset.csv")
        data = pd.read_csv('data/emotion_dataset.csv')
        if data.empty:
            raise ValueError("Emotion dataset is empty")

        logging.info("Vectorizing text data with TF-IDF")
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data['text'])
        y = data['emotion']

        logging.info("Training LinearSVC model")
        model = LinearSVC()
        model.fit(X, y)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/emotion_classifier.pkl')
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
        logging.info("Model and vectorizer saved successfully to models/")
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_emotion_model()


# train_emotion_model.py mein yeh function add karo
def retrain_model():
    """Retrain model with updated dataset"""
    try:
        train_emotion_model()  # Original training function
        logger.info("Model retrained successfully with new data")
        return True
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        return False

# active_learning.py mein retrain functionality add karo
def add_to_dataset(self, text, emotion):
    """Add new text-emotion pair to dataset and retrain"""
    try:
        # Load existing data
        try:
            df = pd.read_csv(self.dataset_path)
        except:
            df = pd.DataFrame(columns=['text', 'emotion'])
        
        # Check if already exists
        if text in df['text'].values:
            logger.info("Text already exists in dataset")
            return True
            
        # Add new entry
        new_entry = pd.DataFrame({'text': [text], 'emotion': [emotion]})
        df = pd.concat([df, new_entry], ignore_index=True)
        
        # Save updated dataset
        df.to_csv(self.dataset_path, index=False)
        logger.info(f"Added new entry: '{text}' -> {emotion}")
        
        # Retrain model if dataset grown significantly
        if len(df) % 10 == 0:  # Every 10 new entries
            logger.info("Dataset grown significantly, retraining model...")
            from train_emotion_model import retrain_model
            retrain_model()
        
        return True
        
    except Exception as e:
        logger.error(f"Error adding to dataset: {e}")
        return False