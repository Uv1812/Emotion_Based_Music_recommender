# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import classification_report
# import joblib
# import os
# from pathlib import Path

# def create_default_dataset():
#     """Create default dataset if file doesn't exist or is empty"""
    # data = {
    #     'text': [
    #         "I feel great today! 😄", 
    #         "This makes me so angry 😠",
    #         "I'm devastated by the news 😢",
    #         "Such a peaceful evening 🌙",
    #         "I'm deeply in love ❤️",
    #         "Feeling excited about the trip 🎉",
    #         "This situation is frustrating 😣",
    #         "I feel anxious and worried 😟",
    #         "What a beautiful day 🌞",
    #         "Heartbroken after the breakup 💔",
    #         "Smile :) 😊",
    #         "Pleading Face 🥺",
    #         "Rolling on the Floor Laughing 🤣",
    #         "Smiling Face with Heart-Eyes 😍",
    #         "Folded Hands 🙏",
    #         "I love listening to thunder by Imagine Dragons 🎵",
    #         "Sad love songs 😢",
    #         "Golden ages old songs 🎻",
    #         "Feel like I'm about to get explosive emotions 💥",
    #         "So much inner healing 🌿"
    #     ],
    #     'emotion': [
    #         'happy', 'angry', 'sad', 'calm', 'love', 
    #         'excited', 'frustrated', 'anxious', 'happy', 'sad',
    #         'happy', 'sad', 'happy', 'love', 'calm',
    #         'happy', 'sad', 'calm', 'angry', 'calm'
    #     ]
    # }
#     return pd.DataFrame(data)

# def train_and_save_model():
#     # Create data directory if needed
#     os.makedirs('data', exist_ok=True)
    
#     try:
#         df = pd.read_csv('data/emotion_dataset.csv')
#         if df.empty:
#             raise pd.errors.EmptyDataError
#     except (FileNotFoundError, pd.errors.EmptyDataError):
#         print("Creating default dataset...")
#         df = create_default_dataset()
#         df.to_csv('data/emotion_dataset.csv', index=False)
    
#     # Create and train model
#     model = Pipeline([
#         ('tfidf', TfidfVectorizer(max_features=5000)),
#         ('clf', LinearSVC(dual='auto', C=1.0))  # Tuned C parameter
#     ])
    
#     # Perform cross-validation
#     scores = cross_val_score(model, df['text'], df['emotion'], cv=5, scoring='accuracy')
#     print(f"Cross-validation accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    
#     # Train model on full dataset
#     model.fit(df['text'], df['emotion'])
    
#     # Evaluate model
#     y_pred = model.predict(df['text'])
#     print("Model evaluation on training data:")
#     print(classification_report(df['emotion'], y_pred))
    
#     # Save model
#     os.makedirs('models', exist_ok=True)
#     model_path = Path('models/emotion_classifier.pkl')
#     joblib.dump(model, model_path, protocol=4)  # Protocol 4 for compatibility
    
#     print(f"Model trained and saved to {model_path}")

# if __name__ == "__main__":
#     train_and_save_model()


# """Train the emotion classification model."""
# import logging
# import joblib
# import pandas as pd
# import sqlite3
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.metrics import classification_report
# from pathlib import Path
# import os

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# def create_default_dataset() -> pd.DataFrame:
#     """Create default emotion dataset if needed."""
#     data = {
#         'text': [
#             "I feel great today! 😄", 
#             "This makes me so angry 😠",
#             "I'm devastated by the news 😢",
#             "Such a peaceful evening 🌙",
#             "I'm deeply in love ❤️",
#             "Feeling excited about the trip 🎉",
#             "This situation is frustrating 😣",
#             "I feel anxious and worried 😟",
#             "What a beautiful day 🌞",
#             "Heartbroken after the breakup 💔",
#             "Smile :) 😊",
#             "Pleading Face 🥺",
#             "Rolling on the Floor Laughing 🤣",
#             "Smiling Face with Heart-Eyes 😍",
#             "Folded Hands 🙏",
#             "I love listening to thunder by Imagine Dragons 🎵",
#             "Sad love songs 😢",
#             "Golden ages old songs 🎻",
#             "Feel like I'm about to get explosive emotions 💥",
#             "So much inner healing 🌿"
#         ],
#         'emotion': [
#             'happy', 'angry', 'sad', 'calm', 'love', 
#             'excited', 'frustrated', 'anxious', 'happy', 'sad',
#             'happy', 'sad', 'happy', 'love', 'calm',
#             'happy', 'sad', 'calm', 'angry', 'calm'
#         ]
#     }
#     df = pd.DataFrame(data)
#     df.drop_duplicates(subset=['text'], inplace=True)
#     return df

# def load_dataset(db_path: str = 'data/emotions.db') -> pd.DataFrame:
#     """Load dataset from SQLite."""
#     conn = sqlite3.connect(db_path)
#     df = pd.read_sql_query("SELECT * FROM emotions", conn)
#     conn.close()
#     if df.empty:
#         df = create_default_dataset()
#         df.to_sql('emotions', sqlite3.connect(db_path), if_exists='replace', index=False)
#     return df

# def train_and_save_model():
#     """Train and save the model with evaluation."""
#     df = load_dataset()
    
#     # Split for validation
#     X_train, X_val, y_train, y_val = train_test_split(df['text'], df['emotion'], test_size=0.2)
    
#     model = Pipeline([
#         ('tfidf', TfidfVectorizer(max_features=5000)),
#         ('clf', LinearSVC(dual='auto', C=1.0))
#     ])
    
#     # Cross-val
#     scores = cross_val_score(model, X_train, y_train, cv=5)
#     logger.info(f"CV Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    
#     model.fit(X_train, y_train)
    
#     # Evaluate
#     y_pred = model.predict(X_val)
#     logger.info("Validation Report:\n" + classification_report(y_val, y_pred))
    
#     os.makedirs('models', exist_ok=True)
#     joblib.dump(model, 'models/emotion_classifier.pkl', protocol=4)
#     logger.info("Model saved.")

# def retrain_model():
#     """Retrain with current dataset."""
#     try:
#         train_and_save_model()
#         logger.info("Retrained successfully.")
#         return True
#     except Exception as e:
#         logger.error(f"Retrain error: {e}")
#         return False

# if __name__ == "__main__":
#     train_and_save_model()

# """Train the emotion classification model."""
# import logging
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.metrics import classification_report
# from pathlib import Path
# import os

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# def create_default_dataset() -> pd.DataFrame:
#     """Create default emotion dataset if needed."""
#     data = {
#         'text': [
#             "I feel great today! 😄", 
#             "This makes me so angry 😠",
#             "I'm devastated by the news 😢",
#             "Such a peaceful evening 🌙",
#             "I'm deeply in love ❤️",
#             "Feeling excited about the trip 🎉",
#             "This situation is frustrating 😣",
#             "I feel anxious and worried 😟",
#             "What a beautiful day 🌞",
#             "Heartbroken after the breakup 💔",
#             "Smile :) 😊",
#             "Pleading Face 🥺",
#             "Rolling on the Floor Laughing 🤣",
#             "Smiling Face with Heart-Eyes 😍",
#             "Folded Hands 🙏",
#             "I love listening to thunder by Imagine Dragons 🎵",
#             "Sad love songs 😢",
#             "Golden ages old songs 🎻",
#             "Feel like I'm about to get explosive emotions 💥",
#             "So much inner healing 🌿"
#         ],
#         'emotion': [
#             'happy', 'angry', 'sad', 'calm', 'love', 
#             'excited', 'frustrated', 'anxious', 'happy', 'sad',
#             'happy', 'sad', 'happy', 'love', 'calm',
#             'happy', 'sad', 'calm', 'angry', 'calm'
#         ]
#     }
#     df = pd.DataFrame(data)
#     df.drop_duplicates(subset=['text'], inplace=True)
#     return df

# def load_dataset() -> pd.DataFrame:
#     """Load dataset from CSV file."""
#     try:
#         # Try to load from CSV file
#         df = pd.read_csv('emotion_dataset.csv')
#         logger.info(f"Loaded dataset from CSV with {len(df)} rows")
        
#         # Check if the CSV has the required columns
#         if 'text' not in df.columns or 'emotion' not in df.columns:
#             logger.warning("CSV file doesn't have 'text' and 'emotion' columns. Using default dataset.")
#             df = create_default_dataset()
            
#     except FileNotFoundError:
#         logger.warning("emotion_dataset.csv not found. Using default dataset.")
#         df = create_default_dataset()
    
#     except Exception as e:
#         logger.error(f"Error loading CSV: {e}. Using default dataset.")
#         df = create_default_dataset()
    
#     return df

# def load_music_dataset() -> pd.DataFrame:
#     """Load music dataset from CSV file."""
#     try:
#         music_df = pd.read_csv('music_dataset.csv')
#         logger.info(f"Loaded music dataset with {len(music_df)} rows")
#         return music_df
#     except Exception as e:
#         logger.error(f"Error loading music dataset: {e}")
#         return pd.DataFrame()  # Return empty DataFrame

# def train_and_save_model():
#     """Train and save the model with evaluation."""
#     # Load emotion dataset
#     df = load_dataset()
    
#     # Load music dataset (for reference, you might want to use it later)
#     music_df = load_music_dataset()
    
#     # Display dataset info
#     logger.info(f"Training dataset shape: {df.shape}")
#     logger.info(f"Emotions distribution:\n{df['emotion'].value_counts()}")
    
#     if not music_df.empty:
#         logger.info(f"Music dataset shape: {music_df.shape}")
#         logger.info(f"Music dataset columns: {music_df.columns.tolist()}")
    
#     # Split for validation
#     X_train, X_val, y_train, y_val = train_test_split(
#         df['text'], df['emotion'], test_size=0.2, random_state=42, stratify=df['emotion']
#     )
    
#     model = Pipeline([
#         ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
#         ('clf', LinearSVC(dual='auto', C=1.0, random_state=42))
#     ])
    
#     # Cross-validation
#     scores = cross_val_score(model, X_train, y_train, cv=5)
#     logger.info(f"CV Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    
#     # Train the model
#     model.fit(X_train, y_train)
    
#     # Evaluate
#     y_pred = model.predict(X_val)
#     logger.info("Validation Report:\n" + classification_report(y_val, y_pred))
    
#     # Save model
#     os.makedirs('models', exist_ok=True)
#     joblib.dump(model, 'models/emotion_classifier.pkl', protocol=4)
#     logger.info("Model saved to models/emotion_classifier.pkl")
    
#     # Test the model with some examples
#     test_texts = [
#         "I feel amazing today!",
#         "This is so frustrating",
#         "I'm feeling peaceful",
#         "I love this song"
#     ]
    
#     predictions = model.predict(test_texts)
#     for text, emotion in zip(test_texts, predictions):
#         logger.info(f"Text: '{text}' -> Emotion: {emotion}")

# def retrain_model():
#     """Retrain with current dataset."""
#     try:
#         train_and_save_model()
#         logger.info("Retrained successfully.")
#         return True
#     except Exception as e:
#         logger.error(f"Retrain error: {e}")
#         return False

# if __name__ == "__main__":
#     train_and_save_model()
"""Train the emotion classification model."""
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def create_default_dataset() -> pd.DataFrame:
    """Create default emotion dataset if needed."""
    data = {
        'text': [
            "I feel great today! 😄", 
            "This makes me so angry 😠",
            "I'm devastated by the news 😢",
            "Such a peaceful evening 🌙",
            "I'm deeply in love ❤️",
            "Feeling excited about the trip 🎉",
            "This situation is frustrating 😣",
            "I feel anxious and worried 😟",
            "What a beautiful day 🌞",
            "Heartbroken after the breakup 💔",
            "Smile :) 😊",
            "Pleading Face 🥺",
            "Rolling on the Floor Laughing 🤣",
            "Smiling Face with Heart-Eyes 😍",
            "Folded Hands 🙏",
            "I love listening to thunder by Imagine Dragons 🎵",
            "Sad love songs 😢",
            "Golden ages old songs 🎻",
            "Feel like I'm about to get explosive emotions 💥",
            "So much inner healing 🌿"
        ],
        'emotion': [
            'happy', 'angry', 'sad', 'calm', 'love', 
            'excited', 'frustrated', 'anxious', 'happy', 'sad',
            'happy', 'sad', 'happy', 'love', 'calm',
            'happy', 'sad', 'calm', 'angry', 'calm'
        ]
    }
    df = pd.DataFrame(data)
    df.drop_duplicates(subset=['text'], inplace=True)
    return df

def load_dataset() -> pd.DataFrame:
    """Load dataset from CSV file in data folder."""
    csv_paths = [
        'data/emotion_dataset.csv',  # Ye correct path hai
        'emotion_dataset.csv',
        'dataset/emotion_dataset.csv'
    ]
    
    for path in csv_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                logger.info(f"Loaded dataset from {path} with {len(df)} rows")
                
                # Check required columns
                if 'text' in df.columns and 'emotion' in df.columns:
                    return df
                else:
                    logger.warning(f"CSV file {path} missing required columns 'text' or 'emotion'")
                    # Try to use first two columns if standard names not found
                    if len(df.columns) >= 2:
                        df = df.rename(columns={df.columns[0]: 'text', df.columns[1]: 'emotion'})
                        return df
                    
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    
    # If no CSV found, create default dataset
    logger.info("No CSV file found. Creating default dataset.")
    df = create_default_dataset()
    return df

def load_music_dataset() -> pd.DataFrame:
    """Load music dataset from CSV file in data folder."""
    csv_paths = [
        'data/music_dataset.csv',  # Ye correct path hai
        'music_dataset.csv',
        'dataset/music_dataset.csv'
    ]
    
    for path in csv_paths:
        try:
            if os.path.exists(path):
                music_df = pd.read_csv(path)
                logger.info(f"Loaded music dataset from {path} with {len(music_df)} rows")
                return music_df
        except Exception as e:
            logger.warning(f"Failed to load music dataset from {path}: {e}")
    
    logger.info("No music dataset found. Continuing without it.")
    return pd.DataFrame()

# def train_and_save_model():
#     """Train and save the model with evaluation."""
#     # Load emotion dataset
#     df = load_dataset()
    
#     # Load music dataset
#     music_df = load_music_dataset()
    
#     # Display dataset info
#     logger.info(f"Training dataset shape: {df.shape}")
#     logger.info(f"Dataset columns: {df.columns.tolist()}")
#     logger.info(f"Emotions: {df['emotion'].unique()}")
#     logger.info(f"Emotions distribution:\n{df['emotion'].value_counts()}")
    
#     if not music_df.empty:
#         logger.info(f"Music dataset shape: {music_df.shape}")
#         logger.info(f"Music dataset columns: {music_df.columns.tolist()}")
    
#     # Split for validation
#     X_train, X_val, y_train, y_val = train_test_split(
#         df['text'], df['emotion'], test_size=0.2, random_state=42, stratify=df['emotion']
#     )
    
#     model = Pipeline([
#         ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
#         ('clf', LinearSVC(dual='auto', C=1.0, random_state=42))
#     ])
    
#     # Cross-validation
#     scores = cross_val_score(model, X_train, y_train, cv=5)
#     logger.info(f"CV Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    
#     # Train the model
#     model.fit(X_train, y_train)
    
#     # Evaluate
#     y_pred = model.predict(X_val)
#     logger.info("Validation Report:\n" + classification_report(y_val, y_pred))
    
#     # Save model
#     os.makedirs('models', exist_ok=True)
#     joblib.dump(model, 'models/emotion_classifier.pkl', protocol=4)
#     logger.info("Model saved to models/emotion_classifier.pkl")
    
#     # Test the model with some examples
#     test_texts = [
#         "I feel amazing today!",
#         "This is so frustrating",
#         "I'm feeling peaceful",
#         "I love this song"
#     ]
    
#     predictions = model.predict(test_texts)
#     for text, emotion in zip(test_texts, predictions):
#         logger.info(f"Text: '{text}' -> Emotion: {emotion}")

def train_and_save_model():
    """Train and save the model with evaluation."""
    # Load emotion dataset
    df = load_dataset()
    
    # Load music dataset
    music_df = load_music_dataset()
    
    # Display dataset info
    logger.info(f"Training dataset shape: {df.shape}")
    logger.info(f"Dataset columns: {df.columns.tolist()}")
    logger.info(f"Emotions distribution:\n{df['emotion'].value_counts()}")
    
    if not music_df.empty:
        logger.info(f"Music dataset shape: {music_df.shape}")
        logger.info(f"Music dataset columns: {music_df.columns.tolist()}")
    
    # FIX: Filter out emotions with only 1 sample
    emotion_counts = df['emotion'].value_counts()
    valid_emotions = emotion_counts[emotion_counts >= 2].index
    df_filtered = df[df['emotion'].isin(valid_emotions)]
    
    logger.info(f"Original dataset: {len(df)} samples, {len(df['emotion'].unique())} emotions")
    logger.info(f"Filtered dataset: {len(df_filtered)} samples, {len(df_filtered['emotion'].unique())} emotions")
    logger.info(f"Removed {len(df) - len(df_filtered)} samples from rare emotions")
    
    # Use filtered dataset
    X = df_filtered['text']
    y = df_filtered['emotion']
    
    # Split for validation - now stratify will work
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('clf', LinearSVC(dual='auto', C=1.0, random_state=42))
    ])
    
    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5)
    logger.info(f"CV Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    logger.info("Validation Report:\n" + classification_report(y_val, y_pred))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/emotion_classifier.pkl', protocol=4)
    logger.info("Model saved to models/emotion_classifier.pkl")
    
    # Test the model with some examples
    test_texts = [
        "I feel amazing today!",
        "This is so frustrating",
        "I'm feeling peaceful",
        "I love this song"
    ]
    
    predictions = model.predict(test_texts)
    for text, emotion in zip(test_texts, predictions):
        logger.info(f"Text: '{text}' -> Emotion: {emotion}")
        
def retrain_model():
    """Retrain with current dataset."""
    try:
        train_and_save_model()
        logger.info("Retrained successfully.")
        return True
    except Exception as e:
        logger.error(f"Retrain error: {e}")
        return False

if __name__ == "__main__":
    train_and_save_model()