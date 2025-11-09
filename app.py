# # # import os
# # # import pandas as pd
# # # import logging
# # # import joblib
# # # import nltk
# # # import numpy as np
# # # import time
# # # import re
# # # import emoji
# # # from functools import lru_cache
# # # from nltk.tokenize import word_tokenize
# # # from nltk.corpus import stopwords
# # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # from sklearn.metrics.pairwise import cosine_similarity
# # # from requests.adapters import HTTPAdapter
# # # from urllib3.util.retry import Retry
# # # import requests
# # # import json
# # # from collections import Counter
# # # from datetime import datetime
# # # import streamlit as st
# # # import urllib.parse
# # # # app.py mein imports section
# # # from active_learning import active_learner
# # # from src.emotion_detection import get_confidence_score, suggest_emotion, preprocess_text
# # # # Setup logging
# # # logger = logging.getLogger(__name__)
# # # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
# # #     logging.FileHandler("app.log", encoding="utf-8"),
# # #     logging.StreamHandler()
# # # ])

# # # # Load config
# # # config = json.load(open('config.json'))

# # # # Cache for YouTube links
# # # CACHE_FILE = "youtube_cache.json"
# # # if os.path.exists(CACHE_FILE):
# # #     with open(CACHE_FILE, 'r', encoding='utf-8') as f:
# # #         youtube_cache = json.load(f)
# # # else:
# # #     youtube_cache = {}

# # # # Ensure NLTK data is loaded
# # # try:
# # #     nltk.data.find('corpora/stopwords')
# # # except LookupError:
# # #     nltk.download('stopwords')
# # # try:
# # #     nltk.data.find('tokenizers/punkt')
# # # except LookupError:
# # #     nltk.download('punkt')

# # # # EMOJI_EMOTION_MAP definition in app.py
# # # EMOJI_EMOTION_MAP = {
# # #     # Happy emojis
# # #     '😊': 'happy', '😂': 'happy', '🤣': 'happy', '😃': 'happy', '😄': 'happy',
# # #     '😁': 'happy', '😆': 'happy', '😎': 'happy', '🤠': 'happy', '🥳': 'happy',
# # #     '😇': 'happy', '🙂': 'happy', '😀': 'happy', '😺': 'happy', '😸': 'happy',
# # #     '😹': 'happy', '😻': 'happy', '💃': 'happy', '🕺': 'happy', '🎉': 'happy',
# # #     '🎊': 'happy', '✨': 'happy', '🎈': 'happy', '🥰': 'love',
    
# # #     # Love emojis
# # #     '😍': 'love', '❤️': 'love', '💕': 'love', '💖': 'love', '💞': 'love',
# # #     '💘': 'love', '💓': 'love', '💗': 'love', '💙': 'love', '💚': 'love',
# # #     '💛': 'love', '💜': 'love', '🧡': 'love', '🤍': 'love', '🤎': 'love',
# # #     '💑': 'love', '👩‍❤️‍👨': 'love', '👨‍❤️‍👨': 'love', '👩‍❤️‍👩': 'love', '💏': 'love',
# # #     '👩‍❤️‍💋‍👨': 'love', '👨‍❤️‍💋‍👨': 'love', '👩‍❤️‍💋‍👩': 'love', '🫶': 'love', '💋': 'love',
    
# # #     # Sad emojis
# # #     '😢': 'sad', '😭': 'sad', '😔': 'sad', '😞': 'sad', '😟': 'sad',
# # #     '😕': 'sad', '🙁': 'sad', '☹️': 'sad', '😣': 'sad', '😖': 'sad',
# # #     '😫': 'sad', '😩': 'sad', '🥺': 'sad', '😿': 'sad', '😾': 'sad',
# # #     '💔': 'sad',
    
# # #     # Angry emojis
# # #     '😡': 'angry', '🤬': 'angry', '😠': 'angry', '😤': 'angry', '👿': 'angry',
# # #     '😾': 'angry', '💢': 'angry',
    
# # #     # Calm emojis
# # #     '😴': 'calm', '😌': 'calm', '🙂': 'calm', '😊': 'calm', '🌙': 'calm',
# # #     '🧘': 'calm', '🌿': 'calm', '🍃': 'calm', '🌊': 'calm', '🏖️': 'calm',
# # #     '🎐': 'calm', '🕉️': 'calm', '☮️': 'calm', '☯️': 'calm', '🌅': 'calm',
# # #     '🌄': 'calm', '🌠': 'calm',
    
# # #     # Excited emojis
# # #     '😃': 'excited', '😄': 'excited', '😁': 'excited', '😆': 'excited',
# # #     '🤩': 'excited', '🥳': 'excited', '🎉': 'excited', '🎊': 'excited',
# # #     '🎁': 'excited', '🎂': 'excited', '🎈': 'excited', '✨': 'excited',
# # #     '⚡': 'excited', '🚀': 'excited', '🔥': 'excited', '💫': 'excited',
    
# # #     # Anxious emojis
# # #     '😨': 'anxious', '😰': 'anxious', '😥': 'anxious', '😓': 'anxious',
# # #     '😬': 'anxious', '😳': 'anxious', '🤯': 'anxious', '🥶': 'anxious',
# # #     '😵': 'anxious', '😵‍💫': 'anxious',
    
# # #     # Other emotions
# # #     '😐': 'neutral', '😶': 'neutral', '😑': 'neutral', '🙄': 'neutral',
# # #     '😯': 'surprised', '😲': 'surprised', '🥴': 'confused', '😕': 'confused',
# # #     '🤔': 'confused', '😷': 'sick', '🤒': 'sick', '🤕': 'sick', '🤢': 'disgusted',
# # #     '🤮': 'disgusted', '😈': 'mischievous', '👻': 'playful', '🤡': 'playful',
# # #     '💩': 'playful', '👏': 'proud', '🤝': 'friendly', '🙌': 'excited',
# # #     '👍': 'friendly', '👎': 'angry', '❤️‍🔥': 'love', '❤️‍🩹': 'sad',
# # #     '🤗': 'friendly', '🤲': 'calm', '🙏': 'calm', '✌️': 'happy',
    
# # #     # Additional emojis from your search history
# # #     '💬': 'friendly',  # speech bubble
# # #     '💷': 'neutral',   # pound banknote
# # # }

# # # class EmotionDetector:

# # #     def rule_based_detection(self, text):
# # #         if not text or not isinstance(text, str):
# # #             return "neutral"
    
# # #         text_lower = text.lower()
    
# # #     # DEBUG: Print the input text to see what's being processed
# # #         print(f"DEBUG: Processing text: '{text}'")
    
# # #     # Check if input contains only emojis (with possible whitespace)
# # #         emoji_only = True
# # #         for char in text:
# # #             if char not in EMOJI_EMOTION_MAP and not char.isspace():
# # #                 emoji_only = False
# # #                 break
    
# # #     # DEBUG: Print if it's emoji only
# # #         print(f"DEBUG: Emoji only: {emoji_only}")
    
# # #     # If input contains only emojis (with possible whitespace)
# # #         if emoji_only:
# # #         # Count emojis by emotion
# # #             emotion_counts = {}
# # #             for char in text:
# # #                 if char in EMOJI_EMOTION_MAP:
# # #                     emotion = EMOJI_EMOTION_MAP[char]
# # #                     emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
# # #                     # DEBUG: Print each emoji and its mapped emotion
# # #                     print(f"DEBUG: Emoji '{char}' -> {emotion}")
        
# # #         # DEBUG: Print emotion counts
# # #             print(f"DEBUG: Emotion counts: {emotion_counts}")
        
# # #             # Return emotion with highest count
# # #             if emotion_counts:
# # #                 result = max(emotion_counts.items(), key=lambda x: x[1])[0]
# # #                 print(f"DEBUG: Selected emotion: {result}")
# # #                 return result
# # #             else:
# # #                 print("DEBUG: No emotions found, returning neutral")
# # #                 return "neutral"
    
# # #     # For mixed text and emoji inputs, prioritize emojis
# # #         emoji_emotions = []
# # #         for char in text:
# # #             if char in EMOJI_EMOTION_MAP:
# # #                 emotion = EMOJI_EMOTION_MAP[char]
# # #                 emoji_emotions.append(emotion)
# # #                 # DEBUG: Print each emoji and its mapped emotion
# # #                 print(f"DEBUG: Emoji '{char}' -> {emotion}")

# # #     # If we have emojis in the text, use the most frequent emoji emotion
# # #         if emoji_emotions:
# # #             emotion_counter = Counter(emoji_emotions)
# # #             result = emotion_counter.most_common(1)[0][0]
# # #             print(f"DEBUG: Most common emoji emotion: {result}")
# # #             return result
    
# # #     # If no emojis, use keyword matching
# # #         keywords = {
# # #             'happy': ['happy', 'joy', 'joyful', 'excited', 'good', 'great', 'fun', 'smile', 'laugh', 'enjoy', 'friends', 'celebrate', 'celebration', 'dance', 'party'],
# # #             'sad': ['sad', 'unhappy', 'cry', 'tears', 'depressed', 'lonely', 'miss', 'alone', 'heartbroken', 'broken', 'upset', 'miserable', 'unfortunate', 'grief', 'sorrow'],
# # #             'angry': ['angry', 'mad', 'hate', 'furious', 'annoyed', 'frustrated', 'irritated', 'outraged', 'rage', 'fury', 'temper', 'hostile'],
# # #             'calm': ['calm', 'peaceful', 'relax', 'chill', 'quiet', 'serene', 'tranquil', 'peace', 'still', 'composed', 'collected'],
# # #             'love': ['love', 'romantic', 'heart', 'crush', 'adore', 'affection', 'lovely', 'beloved', 'darling', 'sweetheart', 'passion', 'romance', 'cherish'],
# # #             'excited': ['excited', 'thrilled', 'pumped', 'energetic', 'anticipate', 'looking forward', 'eager', 'enthusiastic', 'keen', 'avid'],
# # #             'anxious': ['anxious', 'nervous', 'worried', 'stressed', 'scared', 'afraid', 'tense', 'apprehensive', 'concerned', 'uneasy', 'panic'],
# # #             'neutral': ['ok', 'fine', 'alright', 'normal', 'regular', 'usual', 'ordinary', 'average', 'moderate'],
# # #             'friendly': ['friendly', 'friendship', 'companion', 'buddy', 'pal', 'mate', 'amicable', 'sociable', 'outgoing'],
# # #             'sick': ['sick', 'unwell', 'not well', 'ill', 'fever', 'headache', 'nauseous', 'vomit', 'dizzy', 'pain', 'ache']
# # #         }
    
# # #         for emotion, words in keywords.items():
# # #             for word in words:
# # #                 if word in text_lower:
# # #                     print(f"DEBUG: Keyword '{word}' found -> {emotion}")
# # #                     return emotion
    
# # #         print("DEBUG: No keywords found, returning neutral")
# # #         return "neutral"
    
# # #     def __init__(self):
# # #         model_path = os.path.join('models', 'emotion_classifier.pkl')
# # #         vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
# # #         self.model = joblib.load(model_path)
# # #         self.vectorizer = joblib.load(vectorizer_path)
# # #         self.stop_words = set(stopwords.words('english'))
    
# # #     def preprocess(self, text):
# # #         if not text or not isinstance(text, str):
# # #             return ""
    
# # #     # Convert to lowercase
# # #         text = text.lower()
    
# # #     # Remove special characters but keep spaces
# # #         text = re.sub(r'[^\w\s]', ' ', text)
    
# # #     # Handle common contractions and synonyms
# # #         contractions = {
# # #             "im": "i am", "dont": "do not", "cant": "cannot", "wont": "will not",
# # #             "shouldnt": "should not", "wouldnt": "would not", "couldnt": "could not",
# # #             "isnt": "is not", "arent": "are not", "wasnt": "was not", "werent": "were not",
# # #             "havent": "have not", "hasnt": "has not", "hadnt": "had not", "doesnt": "does not",
# # #             "didnt": "did not", "ill": "i will", "youll": "you will", "theyll": "they will",
# # #             "weve": "we have", "ive": "i have", "youve": "you have", "theyve": "they have",
# # #             "im": "i am", "youre": "you are", "theyre": "they are", "were": "we are",
# # #             "thats": "that is", "whats": "what is", "wheres": "where is", "whos": "who is",
# # #             "hows": "how is", "shes": "she is", "hes": "he is", "its": "it is",
# # #             "feel": "feeling", "feels": "feeling", "felt": "feeling"
# # #         }
    
# # #         words = text.split()
# # #         processed_words = []
# # #         for word in words:
# # #             if word in contractions:
# # #                 processed_words.extend(contractions[word].split())
# # #             else:
# # #                 processed_words.append(word)

# # #         return ' '.join(processed_words)
    

# # #     def detect(self, text):
# # #     # First try rule-based detection for speed and reliability
# # #         rule_based_result = self.rule_based_detection(text)
    
# # #     # TEMPORARY: Skip model prediction to test rule-based
# # #         return rule_based_result
    
# # #     # If we have a model, try to use it
# # #         if self.model is not None and self.vectorizer is not None:
# # #             try:
# # #                 processed = self.preprocess(text)
# # #                 if isinstance(processed, str):  # Make sure it's text, not a vector
# # #                     processed_vector = self.vectorizer.transform([processed])
# # #                     model_result = self.model.predict(processed_vector)[0]
                
# # #                     # You can add confidence checking here if needed
# # #                     return model_result
# # #                 else:
# # #                     # If somehow we got a vector, fall back to rule-based
# # #                     return rule_based_result
# # #             except Exception as e:
# # #                 print(f"Model prediction error: {e}")
# # #                 # Fall back to rule-based detection
# # #                 return rule_based_result
# # #         else:
# # #         # No model available, use rule-based
# # #             return rule_based_result


# # # detector = EmotionDetector()

# # # class MusicRecommender:
# # #     def __init__(self):
# # #         os.makedirs('data', exist_ok=True)
# # #         file_path = os.path.join('data', 'music_dataset.csv')
# # #         if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
# # #             try:
# # #                 self.df = pd.read_csv(file_path, encoding='utf-8')
# # #                 if self.df.empty:
# # #                     raise pd.errors.EmptyDataError("CSV file is empty")
# # #                 logger.info(f"Loaded DataFrame from {file_path}:\n{self.df.head().to_string()}")
# # #                 if self.df[['title', 'artist', 'emotion', 'category', 'lyrics']].isna().any().any():
# # #                     logger.warning("NaN values detected in DataFrame. Filling with empty strings.")
# # #                     self.df = self.df.fillna("")
# # #                 raw_cleaned_text = self.df['title'] + ' ' + self.df['artist'] + ' ' + self.df['lyrics'].fillna('')
# # #                 logger.info(f"Raw cleaned text sample: {raw_cleaned_text.head().to_string()}")
# # #                 self.df['cleaned_text'] = raw_cleaned_text.apply(detector.preprocess)
# # #                 logger.info(f"Processed cleaned text sample: {self.df['cleaned_text'].head().to_string()}")
# # #                 unique_emotions = self.df['emotion'].unique()
# # #                 logger.info(f"Unique emotions in dataset: {unique_emotions}")
# # #             except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
# # #                 logger.error(f"Failed to load {file_path}: {str(e)}. Using fallback dataset.")
# # #                 self.df = pd.DataFrame({
# # #                     'title': ["Tum Hi Ho", "Channa Mereya", "Kal Ho Naa Ho", "Rolling in the Deep", "Happy"],
# # #                     'artist': ["Arijit Singh", "Arijit Singh", "Shankar-Ehsaan-Loy", "Adele", "Pharrell Williams"],
# # #                     'emotion': ["love", "sad", "sad", "angry", "happy"],
# # #                     'category': ["Hindi", "Hindi", "Hindi", "English", "English"],
# # #                     'lyrics': ["Tum hi ho, ab tum hi ho...", "Channa mereya, mereya...", "Har pal yahan, je bhar jiyo...", "There's a fire starting in my heart...", "Cause I'm happy clap along..."]
# # #                 })
# # #                 raw_cleaned_text = self.df['title'] + ' ' + self.df['artist'] + ' ' + self.df['lyrics'].fillna('')
# # #                 self.df['cleaned_text'] = raw_cleaned_text.apply(detector.preprocess)
# # #                 logger.info(f"Fallback DataFrame:\n{self.df.head().to_string()}")
# # #         else:
# # #             logger.error(f"{file_path} not found or empty. Using fallback dataset.")
# # #             self.df = pd.DataFrame({
# # #                 'title': ["Tum Hi Ho", "Channa Mereya", "Kal Ho Naa Ho", "Rolling in the Deep", "Happy"],
# # #                 'artist': ["Arijit Singh", "Arijit Singh", "Shankar-Ehsaan-Loy", "Adele", "Pharrell Williams"],
# # #                 'emotion': ["love", "sad", "sad", "angry", "happy'"],
# # #                 'category': ["Hindi", "Hindi", "Hindi", "English", "English"],
# # #                 'lyrics': ["Tum hi ho, ab tum hi ho...", "Channa mereya, mereya...", "Har pal yahan, je bhar jiyo...", "There's a fire starting in my heart...", "Cause I'm happy clap along..."]
# # #             })
# # #             raw_cleaned_text = self.df['title'] + ' ' + self.df['artist'] + ' ' + self.df['lyrics'].fillna('')
# # #             self.df['cleaned_text'] = raw_cleaned_text.apply(detector.preprocess)
# # #             logger.info(f"Fallback DataFrame:\n{self.df.head().to_string()}")
        
# # #         if not self.df.empty:
# # #             self.tfidf = TfidfVectorizer(max_features=5000)
# # #             self.tfidf_matrix = self.tfidf.fit_transform(self.df['cleaned_text'])
# # #             self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
# # #             logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
# # #         else:
# # #             self.tfidf_matrix = np.empty((0, 0))
# # #             self.cosine_sim = np.empty((0, 0))
# # #             logger.warning("Empty DataFrame, initialized empty matrices.")

# # #         # PRELOAD RECOMMENDATIONS - YEH ADD KARO
# # #         self.preload_recommendations()

# # #     # PRELOAD METHOD - YEH ADD KARO
# # #     def preload_recommendations(self):
# # #         """Pre-load recommendations for common emotions to make responses faster"""
# # #         self.common_emotions = ['happy', 'sad', 'angry', 'calm', 'love', 'excited', 'neutral']
# # #         self.preloaded_songs = {}
        
# # #         logger.info("Starting to pre-load recommendations...")
        
# # #         for emotion in self.common_emotions:
# # #             try:
# # #                 # Get 15 songs for each emotion (more than needed for filtering)
# # #                 songs = self.get_recommendations(emotion, "All")
# # #                 self.preloaded_songs[emotion] = songs
# # #                 logger.info(f"Pre-loaded {len(songs)} songs for emotion: {emotion}")
# # #             except Exception as e:
# # #                 logger.error(f"Error pre-loading songs for {emotion}: {e}")
# # #                 # Add empty list as fallback
# # #                 self.preloaded_songs[emotion] = []
        
# # #         logger.info("Pre-loading completed!")

# # #     # GET RECOMMENDATIONS METHOD UPDATE KARO
# # #     def get_recommendations(self, emotion, category="All", n=20):
# # #         try:
# # #             emotion = emotion.lower()
# # #             logger.info(f"Detecting recommendations for emotion: {emotion}, category: {category}")
            
            
# # #             # FIRST CHECK PRELOADED SONGS - YEH ADD KARO
# # #             if emotion in self.preloaded_songs and self.preloaded_songs[emotion]:
# # #                 preloaded = self.preloaded_songs[emotion]
                
# # #                 # Filter by category if needed
# # #                 if category != "All":
# # #                     category_filter = "hindi" if category.lower() == "bollywood" else "english"
# # #                     filtered_songs = [song for song in preloaded if song.get('category', '').lower() == category_filter]
# # #                 else:
# # #                     filtered_songs = preloaded
                
# # #                 # Return exactly n songs
# # #                 return filtered_songs[:n]
            
# # #             # If not preloaded, use original logic
# # #             selected_titles = set()
# # #             filtered = pd.DataFrame()
            
# # #             # Get all songs for the exact emotion
# # #             available = self.df[self.df['emotion'].str.lower() == emotion].dropna(subset=['title', 'artist', 'emotion', 'category', 'lyrics'])
# # #             logger.info(f"Available songs for {emotion}: {len(available)}")
            
# # #             for _, song in available.iterrows():
# # #                 song_category = song['category'].lower()
# # #                 if category == "All" or (category.lower() == "bollywood" and song_category == "hindi") or (category.lower() == "english" and song_category == "english"):
# # #                     if song['title'] not in selected_titles:
# # #                         filtered = pd.concat([filtered, pd.DataFrame([song])], ignore_index=True)
# # #                         selected_titles.add(song['title'])
# # #                         if len(filtered) >= n:  # Stop when we have enough
# # #                             break
                
            
# # #             # If less than desired, use similarity
# # #             if len(filtered) < n and len(self.df) > 0:
# # #                 emotion_index = self.df[self.df['emotion'].str.lower() == emotion].index
# # #                 if len(emotion_index) > 0:
# # #                     emotion_index = emotion_index[0]
# # #                     sim_scores = list(enumerate(self.cosine_sim[emotion_index]))
# # #                     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n*2]  # Get more similar songs
# # #                     song_indices = [i[0] for i in sim_scores]
# # #                     similar_songs = self.df.iloc[song_indices].dropna(subset=['title', 'artist', 'emotion', 'category', 'lyrics'])
                    
# # #                     for _, song in similar_songs.iterrows():
# # #                         song_category = song['category'].lower()
# # #                         if category == "All" or (category.lower() == "bollywood" and song_category == "hindi") or (category.lower() == "english" and song_category == "english"):
# # #                             if song['title'] not in selected_titles:
# # #                                 filtered = pd.concat([filtered, pd.DataFrame([song])], ignore_index=True)
# # #                                 selected_titles.add(song['title'])
# # #                                 if len(filtered) >= n:  # Stop when we have enough
# # #                                     break
            
# # #             # Fill with random songs if still not enough
# # #             if len(filtered) < n:
# # #                 remaining = n - len(filtered)
# # #                 available_other = self.df[~self.df['title'].isin(selected_titles)].dropna(subset=['title', 'artist', 'emotion', 'category', 'lyrics'])
# # #                 if len(available_other) > 0:
# # #                     sampled = available_other[
# # #                         (available_other['category'].str.lower() == "hindi" if category.lower() == "bollywood" else
# # #                          available_other['category'].str.lower() == "english" if category.lower() == "english" else True)
# # #                     ].sample(min(remaining, len(available_other)))
# # #                     filtered = pd.concat([filtered, sampled], ignore_index=True)
            
# # #             logger.info(f"Filtered recommendations: {len(filtered)} songs")
            
# # #             # Convert to list of dictionaries and add links
# # #             recommendations = []
# # #             for _, song in filtered.iterrows():
# # #                 cache_key = f"{song['title']}_{song['artist']}"
# # #                 youtube_link = youtube_cache.get(cache_key, "N/A")
# # #                 if youtube_link == "N/A":
# # #                     youtube_link = get_youtube_link(song['title'], song['artist'], config["YOUTUBE_API_KEY"])
# # #                     if youtube_link != "N/A":
# # #                         youtube_cache[cache_key] = youtube_link
# # #                         with open(CACHE_FILE, 'w', encoding='utf-8') as f:
# # #                             json.dump(youtube_cache, f)
                
# # #                 spotify_link = get_spotify_link(song['title'], song['artist'], config) or "N/A"
                
# # #                 recommendations.append({
# # #                     'title': song['title'],
# # #                     'artist': song['artist'],
# # #                     'emotion': song['emotion'],
# # #                     'category': song['category'],
# # #                     'lyrics': song['lyrics'],
# # #                     'youtube_link': youtube_link,
# # #                     'spotify_link': spotify_link
# # #                 })
            
# # #             logger.info(f"Final recommendations: {len(recommendations)} songs")
# # #             return recommendations[:n]  # Return exactly n songs
            
# # #         except Exception as e:
# # #             logger.error(f"Error in recommendations: {e}")
# # #             # Fallback to random songs
# # #             fallback = self.df.sample(min(n, len(self.df))).to_dict('records')
# # #             for song in fallback:
# # #                 song['youtube_link'] = get_youtube_link(song['title'], song['artist'], config["YOUTUBE_API_KEY"]) or "N/A"
# # #                 song['spotify_link'] = get_spotify_link(song['title'], song['artist'], config) or "N/A"
# # #             return fallback[:n]  # Return exactly n songs



# # # recommender = MusicRecommender()


# # # # Global search history (in-memory)
# # # if 'search_history' not in st.session_state:
# # #     st.session_state.search_history = []

# # # # Add these functions to your app.py file

# # # # ✅ CORRECTED FUNCTIONS - REMOVE ALL 'self' REFERENCES

# # # # app.py ke imports ke baad yeh functions add karo
# # # def get_spotify_link(title, artist, config):
# # #     """Get Spotify link for a song"""
# # #     try:
# # #         token = get_spotify_token(config["SPOTIFY_CLIENT_ID"], config["SPOTIFY_CLIENT_SECRET"])
# # #         if not token:
# # #             return "N/A"
        
# # #         query = urllib.parse.quote(f"{title} {artist}")
# # #         url = f"https://api.spotify.com/v1/search?q={query}&type=track&limit=5"
# # #         headers = {"Authorization": f"Bearer {token}"}
        
# # #         response = requests.get(url, headers=headers, timeout=10)
# # #         response.raise_for_status()
        
# # #         if response.json().get("tracks", {}).get("items"):
# # #             for track in response.json()["tracks"]["items"]:
# # #                 track_title = track["name"].lower()
# # #                 track_artists = [a["name"].lower() for a in track["artists"]]
                
# # #                 if title.lower() in track_title and any(artist.lower() in a for a in track_artists):
# # #                     return track["external_urls"]["spotify"]
        
# # #         return "N/A"
# # #     except Exception as e:
# # #         logger.error(f"Failed to fetch Spotify link for {title} by {artist}: {str(e)}")
# # #         return "N/A"

# # # def get_youtube_link(title, artist, api_key):
# # #     """Get YouTube link for a song"""
# # #     try:
# # #         query = urllib.parse.quote(f"{title} {artist} official")
# # #         url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&key={api_key}&type=video&maxResults=1"
        
# # #         response = requests.get(url, timeout=10)
# # #         response.raise_for_status()
        
# # #         if response.json().get("items"):
# # #             video_id = response.json()["items"][0]["id"]["videoId"]
# # #             return f"https://www.youtube.com/watch?v={video_id}"
        
# # #         return "N/A"
# # #     except Exception as e:
# # #         logger.error(f"Failed to fetch YouTube link for {title} by {artist}: {str(e)}")
# # #         return "N/A"

# # # def get_spotify_token(client_id, client_secret):
# # #     """Get Spotify access token"""
# # #     try:
# # #         auth_url = "https://accounts.spotify.com/api/token"
# # #         auth_data = {"grant_type": "client_credentials"}
# # #         auth = (client_id, client_secret)
        
# # #         response = requests.post(auth_url, auth=auth, data=auth_data, timeout=30)
# # #         response.raise_for_status()
        
# # #         return response.json().get("access_token")
# # #     except Exception as e:
# # #         logger.error(f"Spotify token request failed: {str(e)}")
# # #         return None

# # # # Temporary test code - add this to check your dataset
# # # print("Checking happy songs in dataset...")
# # # happy_songs = recommender.df[recommender.df['emotion'].str.lower() == 'happy']
# # # print(f"Total happy songs: {len(happy_songs)}")
# # # print("Happy songs sample:")
# # # print(happy_songs[['title', 'artist', 'category']].head(20))


# # # @lru_cache(maxsize=100)
# # # def get_cached_youtube_link(title, artist, api_key):
# # #     return get_youtube_link(title, artist, api_key)

# # # @lru_cache(maxsize=100)
# # # def get_cached_spotify_link(title, artist, config):
# # #     return get_spotify_link(title, artist, config)
# # # # Streamlit UI
# # # st.title("🎵 Music Recommendation Based on Mood")

# # # # Search history display in sidebar
# # # with st.sidebar:
# # #     st.subheader("Search History")
# # #     if st.session_state.search_history:
# # #         history_df = pd.DataFrame(st.session_state.search_history)
# # #         st.table(history_df)
# # #     else:
# # #         st.write("No search history yet.")

# # # # Custom text input
# # # user_input = st.text_input("How are you feeling today?", placeholder="Type your mood or add emojis...", key="mood_input")

# # # # Initialize session states
# # # if 'selected_category' not in st.session_state:
# # #     st.session_state.selected_category = "All"
# # # if 'detected_emotion' not in st.session_state:
# # #     st.session_state.detected_emotion = None

# # # # CSS for filter chips and divider
# # # st.markdown(
# # #     """
# # #     <style>
# # #     /* Base styling */
# # #     .main {
# # #         background-color: #0E1117;
# # #         color: #FAFAFA;
# # #     }
    
# # #     /* Filter chips - soft colors */
# # #     .filter-chips {
# # #         display: flex;
# # #         gap: 10px;
# # #         padding: 12px 0;
# # #         margin: 15px 0;
# # #     }
    
# # #     .filter-chip {
# # #         padding: 10px 22px;
# # #         background-color: #2A2F3B;
# # #         border: 1px solid #40444E;
# # #         border-radius: 20px;
# # #         cursor: pointer;
# # #         color: #E0E0E0;
# # #         font-weight: 500;
# # #         transition: all 0.2s ease;
# # #     }
    
# # #     .filter-chip:hover {
# # #         background-color: #3A3F4B;
# # #         color: white;
# # #     }
    
# # #     .filter-chip.active {
# # #         background-color: #FF6B6B;
# # #         border-color: #FF6B6B;
# # #         color: white;
# # #     }
    
# # #     /* Button styling - soft coral */
# # #     .stButton > button {
# # #         background-color: #FF6B6B;
# # #         color: white;
# # #         border: none;
# # #         border-radius: 20px;
# # #         padding: 12px 24px;
# # #         font-weight: 600;
# # #         transition: all 0.2s ease;
# # #     }
    
# # #     .stButton > button:hover {
# # #         background-color: #FF8E8E;
# # #         transform: scale(1.02);
# # #     }
    
# # #     /* Song cards - soft dark */
# # #     .song-card {
# # #         background-color: #1E222A;
# # #         border-radius: 12px;
# # #         padding: 18px;
# # #         margin: 15px 0;
# # #         border: 1px solid #343840;
# # #         transition: all 0.2s ease;
# # #     }
    
# # #     .song-card:hover {
# # #         border-color: #FF6B6B;
# # #     }
    
# # #     /* Emotion badges - pastel colors */
# # #     .emotion-badge {
# # #         display: inline-block;
# # #         padding: 5px 14px;
# # #         border-radius: 15px;
# # #         font-size: 12px;
# # #         font-weight: 600;
# # #         margin: 5px 0;
# # #     }
    
# # #     .happy { background: #FFD93D; color: #1E222A; }
# # #     .sad { background: #6BCB77; color: white; }
# # #     .love { background: #FF6B6B; color: white; }
# # #     .angry { background: #FF9E6B; color: white; }
# # #     .calm { background: #4D96FF; color: white; }
# # #     .excited { background: #FF78C4; color: white; }
# # #     .neutral { background: #9BA3AF; color: white; }
# # #     .friendly { background: #FFB26B; color: white; }
    
# # #     /* Input field styling */
# # #     .stTextInput>div>div>input {
# # #         background-color: #1E222A;
# # #         color: white;
# # #         border: 1px solid #343840;
# # #         border-radius: 15px;
# # #         padding: 12px;
# # #     }
    
# # #     .stTextInput>div>div>input:focus {
# # #         border-color: #FF6B6B;
# # #     }
    
# # #     /* Success message */
# # #     .stSuccess {
# # #         background-color: #6BCB77;
# # #         color: white;
# # #         border-radius: 10px;
# # #     }
# # #     </style>
# # #     """,
# # #     unsafe_allow_html=True
# # # )

# # # def detect_emotion_with_learning(text):
# # #     """Detect emotion with automatic active learning"""
# # #     # First try regular detection
# # #     emotion = detector.detect(text)
    
# # #     # For emoji-only inputs, skip active learning (high confidence)
# # #     emoji_only = True
# # #     for char in text:
# # #         if char not in EMOJI_EMOTION_MAP and not char.isspace():
# # #             emoji_only = False
# # #             break
    
# # #     if emoji_only:
# # #         return emotion  # Skip active learning for emoji-only inputs
    
# # #     confidence = get_confidence_score(text)
    
# # #     # If low confidence, use active learning
# # #     if confidence < 0.6:
# # #         # Try to find similar emotion
# # #         similar_emotion, similarity = active_learner.find_similar_emotion(text)
        
# # #         if similar_emotion:
# # #             if similarity >= 0.95:  # Exact match
# # #                 emotion = similar_emotion
# # #                 st.info(f"Exact match found: {emotion}")
# # #             else:  # Similar but not exact
# # #                 # Automatically learn this new pattern
# # #                 if active_learner.add_to_dataset(text, similar_emotion):
# # #                     st.success(f"✓ Automatically learned: '{text}' = {similar_emotion}")
# # #                 emotion = similar_emotion
# # #         else:
# # #             # Try keyword suggestion for completely new patterns
# # #             suggested = suggest_emotion(text)
# # #             if suggested:
# # #                 # Automatically learn this new pattern
# # #                 if active_learner.add_to_dataset(text, suggested):
# # #                     st.success(f"✓ Automatically learned new pattern: '{text}' = {suggested}")
# # #                 emotion = suggested
# # #             else:
# # #                 # Fallback to neutral for completely unknown patterns
# # #                 emotion = "neutral"
# # #                 st.info("Could not determine emotion, using neutral as default")
    
# # #     return emotion

# # # # Analyze button
# # # analyze_btn = st.button("Analyze Mood and Recommend")

# # # # Divider
# # # st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# # # # Create filter chips using buttons in a single container
# # # with st.container():
# # #     st.markdown('<div class="filter-chips">', unsafe_allow_html=True)
# # #     cols = st.columns([1, 1, 1])  # Equal columns for buttons
# # #     categories = ["All", "Bollywood", "English"]
# # #     for i, category in enumerate(categories):
# # #         with cols[i]:
# # #             # Apply active class via CSS
# # #             button_style = f"background-color: {'#1DB954' if st.session_state.selected_category == category else '#4a4a4a'}; color: white; font-weight: {'700' if st.session_state.selected_category == category else '500'}; text-decoration: {'underline' if st.session_state.selected_category == category else 'none'};"
# # #             if st.button(category, key=f"category_{category}", help=f"Select {category} category"):
# # #                 if st.session_state.selected_category != category:
# # #                     st.session_state.selected_category = category
# # #                     logger.info(f"Category changed to: {category}")
# # #     st.markdown('</div>', unsafe_allow_html=True)

# # # if analyze_btn and user_input:
# # #     # Record to search history
# # #     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# # #     emotion = detector.detect(user_input)
# # #     emotion = detect_emotion_with_learning(user_input)
# # #     st.session_state.search_history.append({"text": user_input, "emotion": emotion, "timestamp": current_time})
# # #     logger.info(f"Added to history: {user_input}, {emotion}, {current_time}")
    
# # #     st.session_state.detected_emotion = emotion
# # #     st.session_state.selected_category = "All"  # Set 'All' as default on button click
# # #     logger.info(f"Default category set to 'All' after analyze button click")
# # #     st.success(f"Detected Emotion: {emotion.capitalize()}")

# # # # Always show recommendations if emotion is detected (refreshes on category change)
# # # if st.session_state.detected_emotion:
# # #     with st.spinner("Finding the best songs for you..."):
# # #         time.sleep(1)  # Simulate processing time
# # #         recommendations = recommender.get_recommendations(st.session_state.detected_emotion, st.session_state.selected_category)
    
# # #     # Clear loader and display recommendations
# # #     st.success("Top Recommendations:")
# # #     if not recommendations:
# # #         st.warning("No recommendations found for the detected emotion. Please try a different mood (e.g., happy, sad, love, angry) or check the dataset.")
# # #     else:
# # #         st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">', unsafe_allow_html=True)
# # #         # Filter out songs with no Spotify link
# # #         valid_recommendations = [song for song in recommendations if song.get('spotify_link', "N/A") != "N/A"]
# # #         if not valid_recommendations:
# # #             st.warning("No songs with valid Spotify links found for this emotion.")
# # #         else:
# # #             for song in valid_recommendations:
# # #                 title = song['title']
# # #                 artist = song['artist']
# # #                 youtube_link = song.get('youtube_link', "N/A")
# # #                 spotify_link = song.get('spotify_link', "N/A")
                
# # #                 # Debug logs
# # #                 logger.info(f"YouTube link for {title} by {artist}: {youtube_link}")
# # #                 logger.info(f"Spotify link for {title} by {artist}: {spotify_link}")
                
# # #                 st.markdown(
# # #                     f"""
# # #                     <div style="margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #333; color: white; position: relative;">
# # #                         <div style="position: absolute; top: 15px; right: 15px;">
# # #                             <a href="{youtube_link}" target="_blank" style="text-decoration: none; margin-left: 10px;"><i class="fab fa-youtube" style="font-size: 24px; color: #FF0000;"></i></a>
# # #                             <a href="{spotify_link}" target="_blank" style="text-decoration: none;"><i class="fab fa-spotify" style="font-size: 24px; color: #1DB954;"></i></a>
# # #                         </div>
# # #                         <div style="margin-bottom: 15px;">
# # #                             <h3 style="margin: 0 0 5px 0; color: #fff;">{title}</h3>
# # #                             <p style="margin: 0 0 5px 0; color: #ccc;">by {artist}</p>
# # #                         </div>
# # #                         <div style="display: flex; gap: 15px; background-color: #333; padding: 15px; border-radius: 5px;">
# # #                             <div style="flex: 0 0 300px;">
# # #                                 <iframe width="300" height="200" src="https://www.youtube.com/embed/{youtube_link.split('=')[-1] if 'v=' in youtube_link else youtube_link}" frameborder="0" allowfullscreen style="border-radius: 5px;"></iframe>
# # #                             </div>
# # #                             <div>
# # #                                 <iframe src="https://open.spotify.com/embed/track/{spotify_link.split('/')[-1]}" width="330" height="85" style="border-radius: 5px;"></iframe>
# # #                             </div>
# # #                         </div>
# # #                     </div>
# # #                     """,
# # #                     unsafe_allow_html=True
# # #                 )


# # import os
# # import pandas as pd
# # import logging
# # import joblib
# # import nltk
# # import numpy as np
# # import time
# # import re
# # import emoji
# # from functools import lru_cache
# # from nltk.tokenize import word_tokenize
# # from nltk.corpus import stopwords
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import cosine_similarity
# # from requests.adapters import HTTPAdapter
# # from urllib3.util.retry import Retry
# # import requests
# # import json
# # from collections import Counter
# # from datetime import datetime
# # import streamlit as st
# # import urllib.parse

# # # Import our modules
# # from src.music_recommendation import search_directly
# # from src.emotion_detection import EmotionDetector, get_confidence_score, suggest_emotion, preprocess_text
# # from src.music_recommendation import initialize_recommender, get_recommendations, update_dataset , normalize_category
# # from active_learning import active_learner
# # from constants import EMOJI_EMOTION_MAP, EMOTION_CATEGORY_MAPPING

# # # Setup logging
# # logger = logging.getLogger(__name__)
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
# #     logging.FileHandler("app.log", encoding="utf-8"),
# #     logging.StreamHandler()
# # ])

# # # Load config
# # config = json.load(open('config.json'))

# # # Initialize the recommender with config
# # initialize_recommender(config)

# # # Cache for YouTube links
# # CACHE_FILE = "youtube_cache.json"
# # if os.path.exists(CACHE_FILE):
# #     with open(CACHE_FILE, 'r', encoding='utf-8') as f:
# #         youtube_cache = json.load(f)
# # else:
# #     youtube_cache = {}

# # # Ensure NLTK data is loaded
# # try:
# #     nltk.data.find('corpora/stopwords')
# # except LookupError:
# #     nltk.download('stopwords')
# # try:
# #     nltk.data.find('tokenizers/punkt')
# # except LookupError:
# #     nltk.download('punkt')

# # # Initialize components
# # detector = EmotionDetector()

# # # Initialize session states FIRST, before any other code
# # if 'search_history' not in st.session_state:
# #     st.session_state.search_history = []

# # if 'selected_category' not in st.session_state:
# #     st.session_state.selected_category = "All"

# # if 'detected_emotion' not in st.session_state:
# #     st.session_state.detected_emotion = None

# # if 'recommendation_count' not in st.session_state:
# #     st.session_state.recommendation_count = 0

# # if 'show_recommendations' not in st.session_state:
# #     st.session_state.show_recommendations = False

# # # Add these to session state initialization
# # if 'direct_search_results' not in st.session_state:
# #     st.session_state.direct_search_results = None

# # if 'search_type' not in st.session_state:
# #     st.session_state.search_type = None

# # # CSS for filter chips and divider
# # st.markdown(
# #     """
# #     <style>
# #     /* Base styling */
# #     .main {
# #         background-color: #0E1117;
# #         color: #FAFAFA;
# #     }
    
# #     /* Filter chips - soft colors */
# #     .filter-chips {
# #         display: flex;
# #         gap: 10px;
# #         padding: 12px 0;
# #         margin: 15px 0;
# #     }
    
# #     .filter-chip {
# #         padding: 10px 22px;
# #         background-color: #2A2F3B;
# #         border: 1px solid #40444E;
# #         border-radius: 20px;
# #         cursor: pointer;
# #         color: #E0E0E0;
# #         font-weight: 500;
# #         transition: all 0.2s ease;
# #     }
    
# #     .filter-chip:hover {
# #         background-color: #3A3F4B;
# #         color: white;
# #     }
    
# #     .filter-chip.active {
# #         background-color: #FF6B6B;
# #         border-color: #FF6B6B;
# #         color: white;
# #     }
    
# #     /* Button styling - soft coral */
# #     .stButton > button {
# #         background-color: #FF6B6B;
# #         color: white;
# #         border: none;
# #         border-radius: 20px;
# #         padding: 12px 24px;
# #         font-weight: 600;
# #         transition: all 0.2s ease;
# #     }
    
# #     .stButton > button:hover {
# #         background-color: #FF8E8E;
# #         transform: scale(1.02);
# #     }
    
# #     /* Song cards - soft dark */
# #     .song-card {
# #         background-color: #1E222A;
# #         border-radius: 12px;
# #         padding: 18px;
# #         margin: 15px 0;
# #         border: 1px solid #343840;
# #         transition: all 0.2s ease;
# #     }
    
# #     .song-card:hover {
# #         border-color: #FF6B6B;
# #     }
    
# #     /* Emotion badges - pastel colors */
# #     .emotion-badge {
# #         display: inline-block;
# #         padding: 5px 14px;
# #         border-radius: 15px;
# #         font-size: 12px;
# #         font-weight: 600;
# #         margin: 5px 0;
# #     }
    
# #     .happy { background: #FFD93D; color: #1E222A; }
# #     .sad { background: #6BCB77; color: white; }
# #     .love { background: #FF6B6B; color: white; }
# #     .angry { background: #FF9E6B; color: white; }
# #     .calm { background: #4D96FF; color: white; }
# #     .excited { background: #FF78C4; color: white; }
# #     .neutral { background: #9BA3AF; color: white; }
# #     .friendly { background: #FFB26B; color: white; }
    
# #     /* Input field styling */
# #     .stTextInput>div>div>input {
# #         background-color: #1E222A;
# #         color: white;
# #         border: 1px solid #343840;
# #         border-radius: 15px;
# #         padding: 12px;
# #     }
    
# #     .stTextInput>div>div>input:focus {
# #         border-color: #FF6B6B;
# #     }
    
# #     /* Success message */
# #     .stSuccess {
# #         background-color: #6BCB77;
# #         color: white;
# #         border-radius: 10px;
# #     }
# #     </style>
# #     """,
# #     unsafe_allow_html=True
# # )

# # def detect_search_type(text):
# #     """Detect if user wants emotion-based or direct search"""
# #     text_lower = text.lower()
    
# #     # Keywords that indicate direct song/artist search
# #     direct_search_keywords = [
# #         'play', 'song', 'artist', 'music', 'by', 'search', 'find',
# #         'gaan', 'gāna', 'गाना', 'संगीत', 'play song', 'play music'
# #     ]
    
# #     # Check if text contains direct search keywords
# #     for keyword in direct_search_keywords:
# #         if keyword in text_lower:
# #             return 'direct'
    
# #     # Check if it's a known artist or song name pattern
# #     if any(word in text_lower for word in ['arjit', 'arijit', 'kishore', 'lata', 'taylor', 'weeknd']):
# #         return 'direct'
    
# #     # Default to emotion detection
# #     return 'emotion'

# # def detect_emotion_with_learning(text):
# #     """Detect emotion with automatic active learning"""
# #     # First try regular detection
# #     emotion = detector.detect(text)
    
# #     # For emoji-only inputs, skip active learning (high confidence)
# #     emoji_only = True
# #     for char in text:
# #         if char not in EMOJI_EMOTION_MAP and not char.isspace():
# #             emoji_only = False
# #             break
    
# #     if emoji_only:
# #         return emotion  # Skip active learning for emoji-only inputs
    
# #     confidence = get_confidence_score(text)
    
# #     # If low confidence, use active learning
# #     if confidence < 0.6:
# #         # Try to find similar emotion
# #         similar_emotion, similarity = active_learner.find_similar_emotion(text)
        
# #         if similar_emotion:
# #             if similarity >= 0.95:  # Exact match
# #                 emotion = similar_emotion
# #                 st.info(f"Exact match found: {emotion}")
# #             else:  # Similar but not exact
# #                 # Automatically learn this new pattern
# #                 if active_learner.add_to_dataset(text, similar_emotion):
# #                     st.success(f"✓ Automatically learned: '{text}' = {similar_emotion}")
# #                 emotion = similar_emotion
# #         else:
# #             # Try keyword suggestion for completely new patterns
# #             suggested = suggest_emotion(text)
# #             if suggested:
# #                 # Automatically learn this new pattern
# #                 if active_learner.add_to_dataset(text, suggested):
# #                     st.success(f"✓ Automatically learned new pattern: '{text}' = {suggested}")
# #                 emotion = suggested
# #             else:
# #                 # Fallback to neutral for completely unknown patterns
# #                 emotion = "neutral"
# #                 st.info("Could not determine emotion, using neutral as default")
    
# #     return emotion

# # # Streamlit UI
# # st.title("🎵 Music Recommendation Based on Mood")

# # # Search history display in sidebar
# # with st.sidebar:
# #     st.subheader("Search History")
# #     if st.session_state.search_history:
# #         reversed_history = list(reversed(st.session_state.search_history))
# #         history_df = pd.DataFrame(st.session_state.search_history)
# #         # Format the timestamp for better display
# #         if not history_df.empty and 'timestamp' in history_df.columns:
# #             history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%H:%M:%S')
        
# #         st.dataframe(history_df[['text', 'emotion', 'timestamp']], use_container_width=True)
        
# #         # Add a clear history button
# #         if st.button("Clear History", key="clear_history"):
# #             st.session_state.search_history = []
# #             st.rerun()

# #     else:
# #         st.write("No search history yet.")

# # # Custom text input
# # user_input = st.text_input("How are you feeling today?", placeholder="Type your mood or add emojis...", key="mood_input")

# # # Analyze button
# # analyze_btn = st.button("Analyze Mood and Recommend")

# # # Divider
# # st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# # # Create filter chips using buttons in a single container
# # with st.container():
# #     st.markdown('<div class="filter-chips">', unsafe_allow_html=True)
# #     cols = st.columns([1, 1, 1])  # Equal columns for buttons
# #     categories = ["All", "Bollywood", "English"]
# #     for i, category in enumerate(categories):
# #         with cols[i]:
# #             # Apply active class via CSS
# #             button_style = f"background-color: {'#1DB954' if st.session_state.selected_category == category else '#4a4a4a'}; color: white; font-weight: {'700' if st.session_state.selected_category == category else '500'}; text-decoration: {'underline' if st.session_state.selected_category == category else 'none'};"
# #             if st.button(category, key=f"category_{category}", help=f"Select {category} category"):
# #                 if st.session_state.selected_category != category:
# #                     st.session_state.selected_category = category
# #                     logger.info(f"Category changed to: {category}")
# #     st.markdown('</div>', unsafe_allow_html=True)

# # # if analyze_btn and user_input:
# # #     # Record to search history
# # #     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# # #     emotion = detect_emotion_with_learning(user_input)
# # #     st.session_state.search_history.append({"text": user_input, "emotion": emotion, "timestamp": current_time})
# # #     logger.info(f"Added to history: {user_input}, {emotion}, {current_time}")
    
# # #     st.session_state.detected_emotion = emotion
# # #     st.session_state.selected_category = "All"  # Set 'All' as default on button click
# # #     logger.info(f"Default category set to 'All' after analyze button click")
# # #     st.success(f"Detected Emotion: {emotion.capitalize()}")
# # if analyze_btn and user_input:
# #     # Detect search type
# #     search_type = detect_search_type(user_input)
# #     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
# #     if search_type == 'direct':
# #         # Direct song/artist search
# #         st.info(f"🔍 Searching for: '{user_input}'")
# #         recommendations = search_directly(user_input, 10)
# #         st.session_state.direct_search_results = recommendations
# #         st.session_state.search_type = 'direct'
# #         st.session_state.search_history.append({
# #             "text": user_input, 
# #             "emotion": "direct_search", 
# #             "timestamp": current_time
# #         })
        
# #     else:
# #         # Emotion-based search
# #         emotion = detect_emotion_with_learning(user_input)
# #         st.session_state.search_history.append({
# #             "text": user_input, 
# #             "emotion": emotion, 
# #             "timestamp": current_time
# #         })
        
# #         st.session_state.detected_emotion = emotion
# #         st.session_state.selected_category = "All"
# #         st.session_state.search_type = 'emotion'
# #         st.success(f"Detected Emotion: {emotion.capitalize()}")
# # # Always show recommendations if emotion is detected (refreshes on category change)
# # # if st.session_state.detected_emotion:
# # #     with st.spinner("Finding the best songs for you..."):
# # #         time.sleep(1)  # Simulate processing time
# # #         recommendations = get_recommendations(st.session_state.detected_emotion, st.session_state.selected_category, 10)
# #   # Show recommendations based on search type
# # if st.session_state.get('search_type') == 'direct' and st.session_state.get('direct_search_results'):
# #     recommendations = st.session_state.direct_search_results
# #     st.success("🎵 Direct Search Results:")
    
# # elif st.session_state.get('detected_emotion'):
# #     with st.spinner("Finding the best songs for you..."):
# #         time.sleep(1)
# #         recommendations = get_recommendations(
# #             st.session_state.detected_emotion, 
# #             st.session_state.selected_category, 
# #             10
# #         )
# #     # st.success("Top Recommendations:")  
# #     # Clear loader and display recommendations
# #     st.success("Top Recommendations:")
# #     if not recommendations:
# #         st.warning("No recommendations found for the detected emotion. Please try a different mood (e.g., happy, sad, love, angry) or check the dataset.")
# #     else:
# #         st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">', unsafe_allow_html=True)
# #         # Filter out songs with no Spotify link
# #         valid_recommendations = [song for song in recommendations if song.get('spotify_link', "N/A") != "N/A"]
# #         if not valid_recommendations:
# #             st.warning("No songs with valid Spotify links found for this emotion.")
# #         else:
# #             for song in valid_recommendations:
# #                 title = song['title']
# #                 artist = song['artist']
# #                 youtube_link = song.get('youtube_link', "N/A")
# #                 spotify_link = song.get('spotify_link', "N/A")
                
# #                 # Debug logs
# #                 logger.info(f"YouTube link for {title} by {artist}: {youtube_link}")
# #                 logger.info(f"Spotify link for {title} by {artist}: {spotify_link}")
                
# #                 st.markdown(
# #                     f"""
# #                     <div style="margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #333; color: white; position: relative;">
# #                         <div style="position: absolute; top: 15px; right: 15px;">
# #                             <a href="{youtube_link}" target="_blank" style="text-decoration: none; margin-left: 10px;"><i class="fab fa-youtube" style="font-size: 24px; color: #FF0000;"></i></a>
# #                             <a href="{spotify_link}" target="_blank" style="text-decoration: none;"><i class="fab fa-spotify" style="font-size: 24px; color: #1DB954;"></i></a>
# #                         </div>
# #                         <div style="margin-bottom: 15px;">
# #                             <h3 style="margin: 0 0 5px 0; color: #fff;">{title}</h3>
# #                             <p style="margin: 0 0 5px 0; color: #ccc;">by {artist}</p>
# #                         </div>
# #                         <div style="display: flex; gap: 15px; background-color: #333; padding: 15px; border-radius: 5px;">
# #                             <div style="flex: 0 0 300px;">
# #                                 <iframe width="300" height="200" src="https://www.youtube.com/embed/{youtube_link.split('=')[-1] if 'v=' in youtube_link else youtube_link}" frameborder="0" allowfullscreen style="border-radius: 5px;"></iframe>
# #                             </div>
# #                             <div>
# #                                 <iframe src="https://open.spotify.com/embed/track/{spotify_link.split('/')[-1]}" width="330" height="85" style="border-radius: 5px;"></iframe>
# #                             </div>
# #                         </div>
# #                     </div>
# #                     """,
# #                     unsafe_allow_html=True
# #                 )
# #  complete spotify
import os
import pandas as pd
import logging
import joblib
import nltk
import numpy as np
import time
import re
import emoji
from functools import lru_cache
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import json
from collections import Counter
from datetime import datetime
import streamlit as st
import urllib.parse

# Import our modules
from src.music_recommendation import search_directly
from src.emotion_detection import EmotionDetector, get_confidence_score, suggest_emotion, preprocess_text
from src.music_recommendation import initialize_recommender, get_recommendations, update_dataset , normalize_category
from active_learning import active_learner
from constants import EMOJI_EMOTION_MAP, EMOTION_CATEGORY_MAPPING

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("app.log", encoding="utf-8"),
    logging.StreamHandler()
])

# Load config
config = json.load(open('config.json'))

# DEBUG: Check if API keys are loaded properly
logger.info(f"YouTube API Key loaded: {'Yes' if config.get('YOUTUBE_API_KEY') else 'No'}")
logger.info(f"Spotify Client ID loaded: {'Yes' if config.get('SPOTIFY_CLIENT_ID') else 'No'}")

# Initialize the recommender with config
initialize_recommender(config)

# Cache for YouTube links
CACHE_FILE = "youtube_cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        youtube_cache = json.load(f)
else:
    youtube_cache = {}

# Ensure NLTK data is loaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize components
detector = EmotionDetector()

# Initialize session states FIRST, before any other code
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

if 'selected_category' not in st.session_state:
    st.session_state.selected_category = "All"

if 'detected_emotion' not in st.session_state:
    st.session_state.detected_emotion = None

if 'recommendation_count' not in st.session_state:
    st.session_state.recommendation_count = 0

if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False

# Add these to session state initialization
if 'direct_search_results' not in st.session_state:
    st.session_state.direct_search_results = None

if 'search_type' not in st.session_state:
    st.session_state.search_type = None

# CSS for filter chips and divider
st.markdown(
    """
    <style>
    /* Base styling */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Filter chips - soft colors */
    .filter-chips {
        display: flex;
        gap: 10px;
        padding: 12px 0;
        margin: 15px 0;
    }
    
    .filter-chip {
        padding: 10px 22px;
        background-color: #2A2F3B;
        border: 1px solid #40444E;
        border-radius: 20px;
        cursor: pointer;
        color: #E0E0E0;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .filter-chip:hover {
        background-color: #3A3F4B;
        color: white;
    }
    
    .filter-chip.active {
        background-color: #FF6B6B;
        border-color: #FF6B6B;
        color: white;
    }
    
    /* Button styling - soft coral */
    .stButton > button {
        background-color: #FF6B6B;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #FF8E8E;
        transform: scale(1.02);
    }
    
    /* Song cards - soft dark */
    .song-card {
        background-color: #1E222A;
        border-radius: 12px;
        padding: 18px;
        margin: 15px 0;
        border: 1px solid #343840;
        transition: all 0.2s ease;
    }
    
    .song-card:hover {
        border-color: #FF6B6B;
    }
    
    /* Emotion badges - pastel colors */
    .emotion-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: 600;
        margin: 5px 0;
    }
    
    .happy { background: #FFD93D; color: #1E222A; }
    .sad { background: #6BCB77; color: white; }
    .love { background: #FF6B6B; color: white; }
    .angry { background: #FF9E6B; color: white; }
    .calm { background: #4D96FF; color: white; }
    .excited { background: #FF78C4; color: white; }
    .neutral { background: #9BA3AF; color: white; }
    .friendly { background: #FFB26B; color: white; }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        background-color: #1E222A;
        color: white;
        border: 1px solid #343840;
        border-radius: 15px;
        padding: 12px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #FF6B6B;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #6BCB77;
        color: white;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def detect_search_type(text):
    """Detect if user wants emotion-based or direct search"""
    text_lower = text.lower()
    
    # Keywords that indicate direct song/artist search
    direct_search_keywords = [
        'play', 'song', 'artist', 'music', 'by', 'search', 'find',
        'gaan', 'gāna', 'गाना', 'संगीत', 'play song', 'play music'
    ]
    
    # Check if text contains direct search keywords
    for keyword in direct_search_keywords:
        if keyword in text_lower:
            return 'direct'
    
    # Check if it's a known artist or song name pattern
    if any(word in text_lower for word in ['arjit', 'arijit', 'kishore', 'lata', 'taylor', 'weeknd']):
        return 'direct'
    
    # Default to emotion detection
    return 'emotion'

def detect_emotion_with_learning(text):
    """Detect emotion with automatic active learning"""
    # First try regular detection
    emotion = detector.detect(text)
    
    # For emoji-only inputs, skip active learning (high confidence)
    emoji_only = True
    for char in text:
        if char not in EMOJI_EMOTION_MAP and not char.isspace():
            emoji_only = False
            break
    
    if emoji_only:
        return emotion  # Skip active learning for emoji-only inputs
    
    confidence = get_confidence_score(text)
    
    # If low confidence, use active learning
    if confidence < 0.6:
        # Try to find similar emotion
        similar_emotion, similarity = active_learner.find_similar_emotion(text)
        
        if similar_emotion:
            if similarity >= 0.95:  # Exact match
                emotion = similar_emotion
                st.info(f"Exact match found: {emotion}")
            else:  # Similar but not exact
                # Automatically learn this new pattern
                if active_learner.add_to_dataset(text, similar_emotion):
                    st.success(f"✓ Automatically learned: '{text}' = {similar_emotion}")
                emotion = similar_emotion
        else:
            # Try keyword suggestion for completely new patterns
            suggested = suggest_emotion(text)
            if suggested:
                # Automatically learn this new pattern
                if active_learner.add_to_dataset(text, suggested):
                    st.success(f"✓ Automatically learned new pattern: '{text}' = {suggested}")
                emotion = suggested
            else:
                # Fallback to neutral for completely unknown patterns
                emotion = "neutral"
                st.info("Could not determine emotion, using neutral as default")
    
    return emotion

# Streamlit UI
st.title("🎵 Music Recommendation Based on Mood")

# Search history display in sidebar
with st.sidebar:
    st.subheader("Search History")
    if st.session_state.search_history:
        reversed_history = list(reversed(st.session_state.search_history))
        history_df = pd.DataFrame(st.session_state.search_history)
        # Format the timestamp for better display
        if not history_df.empty and 'timestamp' in history_df.columns:
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%H:%M:%S')
        
        st.dataframe(history_df[['text', 'emotion', 'timestamp']], use_container_width=True)
        
        # Add a clear history button
        if st.button("Clear History", key="clear_history"):
            st.session_state.search_history = []
            st.rerun()

    else:
        st.write("No search history yet.")

# Custom text input
user_input = st.text_input("How are you feeling today?", placeholder="Type your mood or add emojis...", key="mood_input")

# Analyze button
analyze_btn = st.button("Analyze Mood and Recommend")

# Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Create filter chips using buttons in a single container
with st.container():
    st.markdown('<div class="filter-chips">', unsafe_allow_html=True)
    cols = st.columns([1, 1, 1])  # Equal columns for buttons
    categories = ["All", "Bollywood", "English"]
    for i, category in enumerate(categories):
        with cols[i]:
            # Apply active class via CSS
            button_style = f"background-color: {'#1DB954' if st.session_state.selected_category == category else '#4a4a4a'}; color: white; font-weight: {'700' if st.session_state.selected_category == category else '500'}; text-decoration: {'underline' if st.session_state.selected_category == category else 'none'};"
            if st.button(category, key=f"category_{category}", help=f"Select {category} category"):
                if st.session_state.selected_category != category:
                    st.session_state.selected_category = category
                    logger.info(f"Category changed to: {category}")
    st.markdown('</div>', unsafe_allow_html=True)

if analyze_btn and user_input:
    # Detect search type
    search_type = detect_search_type(user_input)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if search_type == 'direct':
        # Direct song/artist search
        st.info(f"🔍 Searching for: '{user_input}'")
        recommendations = search_directly(user_input, 10)
        st.session_state.direct_search_results = recommendations
        st.session_state.search_type = 'direct'
        st.session_state.search_history.append({
            "text": user_input, 
            "emotion": "direct_search", 
            "timestamp": current_time
        })
        
    else:
        # Emotion-based search
        emotion = detect_emotion_with_learning(user_input)
        st.session_state.search_history.append({
            "text": user_input, 
            "emotion": emotion, 
            "timestamp": current_time
        })
        
        st.session_state.detected_emotion = emotion
        st.session_state.selected_category = "All"
        st.session_state.search_type = 'emotion'
        st.success(f"Detected Emotion: {emotion.capitalize()}")

# Show recommendations based on search type
if st.session_state.get('search_type') == 'direct' and st.session_state.get('direct_search_results'):
    recommendations = st.session_state.direct_search_results
    st.success("🎵 Direct Search Results:")
    
elif st.session_state.get('detected_emotion'):
    with st.spinner("Finding the best songs for you..."):
        time.sleep(1)
        recommendations = get_recommendations(
            st.session_state.detected_emotion, 
            st.session_state.selected_category, 
            10
        )
    st.success("Top Recommendations:")

# Display recommendations
if 'recommendations' in locals() and recommendations:
    # Filter out songs with no Spotify link
    valid_recommendations = [song for song in recommendations if song.get('spotify_link', "N/A") != "N/A"]
    if not valid_recommendations:
        st.warning("No songs with valid Spotify links found for this emotion.")
    else:
        for song in valid_recommendations:
            title = song['title']
            artist = song['artist']
            youtube_link = song.get('youtube_link', "N/A")
            spotify_link = song.get('spotify_link', "N/A")
            
            # Extract YouTube video ID
            youtube_id = None
            if youtube_link != "N/A" and 'v=' in youtube_link:
                youtube_id = youtube_link.split('v=')[1].split('&')[0]
            elif youtube_link != "N/A" and 'youtu.be/' in youtube_link:
                youtube_id = youtube_link.split('youtu.be/')[1].split('?')[0]
            
            # Extract Spotify track ID
            spotify_track_id = None
            if spotify_link != "N/A" and 'open.spotify.com/track/' in spotify_link:
                spotify_track_id = spotify_link.split('open.spotify.com/track/')[1].split('?')[0]
            
            # Create song card
            st.markdown(
                f"""
                <div style="margin: 10px; padding: 15px; border: 1px solid #444; border-radius: 10px; background-color: #1E222A; color: white; position: relative;">
                    <div style="position: absolute; top: 15px; right: 15px;">
                        <a href="{youtube_link}" target="_blank" style="text-decoration: none; margin-left: 10px;">
                            <i class="fab fa-youtube" style="font-size: 24px; color: #FF0000;"></i>
                        </a>
                        <a href="{spotify_link}" target="_blank" style="text-decoration: none;">
                            <i class="fab fa-spotify" style="font-size: 24px; color: #1DB954;"></i>
                        </a>
                    </div>
                    <div style="margin-bottom: 15px;">
                        <h3 style="margin: 0 0 5px 0; color: #fff;">{title}</h3>
                        <p style="margin: 0 0 5px 0; color: #ccc;">by {artist}</p>
                        <span class="emotion-badge {song.get('emotion', 'neutral')}">{song.get('emotion', 'neutral').capitalize()}</span>
                    </div>
                    <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 300px;">
                            {"<iframe width='100%' height='200' src='https://www.youtube.com/embed/" + youtube_id + "' frameborder='0' allowfullscreen style='border-radius: 5px;'></iframe>" if youtube_id else "<p style='color: #888;'>YouTube link not available</p>"}
                        </div>
                        <div style="flex: 1; min-width: 300px;">
                            {"<iframe src='https://open.spotify.com/embed/track/" + spotify_track_id + "' width='100%' height='85' frameborder='0' allowtransparency='true' allow='encrypted-media' style='border-radius: 5px;'></iframe>" if spotify_track_id else "<p style='color: #888;'>Spotify link not available</p>"}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
elif 'recommendations' in locals() and not recommendations:
    st.warning("No recommendations found. Please try a different mood or search term.")
# import os
# import pandas as pd
# import logging
# import joblib
# import nltk
# import numpy as np
# import time
# import re
# import emoji
# from functools import lru_cache
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry
# import requests
# import json
# from collections import Counter
# from datetime import datetime
# import streamlit as st
# import urllib.parse

# # Import our modules
# from src.music_recommendation import search_directly
# from src.emotion_detection import EmotionDetector, get_confidence_score, suggest_emotion, preprocess_text
# from src.music_recommendation import initialize_recommender, get_recommendations, update_dataset , normalize_category
# from active_learning import active_learner
# from constants import EMOJI_EMOTION_MAP, EMOTION_CATEGORY_MAPPING

# # Setup logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
#     logging.FileHandler("app.log", encoding="utf-8"),
#     logging.StreamHandler()
# ])

# # Load config
# config = json.load(open('config.json'))

# # DEBUG: Check if API keys are loaded properly
# logger.info(f"YouTube API Key loaded: {'Yes' if config.get('YOUTUBE_API_KEY') else 'No'}")
# logger.info(f"Spotify Client ID loaded: {'Yes' if config.get('SPOTIFY_CLIENT_ID') else 'No'}")

# # Initialize the recommender with config
# initialize_recommender(config)

# # Cache for YouTube links
# CACHE_FILE = "youtube_cache.json"
# if os.path.exists(CACHE_FILE):
#     with open(CACHE_FILE, 'r', encoding='utf-8') as f:
#         youtube_cache = json.load(f)
# else:
#     youtube_cache = {}

# # Ensure NLTK data is loaded
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# # Initialize components
# detector = EmotionDetector()

# # Initialize session states FIRST, before any other code
# if 'search_history' not in st.session_state:
#     st.session_state.search_history = []

# if 'selected_category' not in st.session_state:
#     st.session_state.selected_category = "All"

# if 'detected_emotion' not in st.session_state:
#     st.session_state.detected_emotion = None

# if 'recommendation_count' not in st.session_state:
#     st.session_state.recommendation_count = 0

# if 'show_recommendations' not in st.session_state:
#     st.session_state.show_recommendations = False

# # Add these to session state initialization
# if 'direct_search_results' not in st.session_state:
#     st.session_state.direct_search_results = None

# if 'search_type' not in st.session_state:
#     st.session_state.search_type = None

# # CSS for filter chips and divider
# st.markdown(
#     """
#     <style>
#     /* Base styling */
#     .main {
#         background-color: #0E1117;
#         color: #FAFAFA;
#     }
    
#     /* Filter chips - soft colors */
#     .filter-chips {
#         display: flex;
#         gap: 10px;
#         padding: 12px 0;
#         margin: 15px 0;
#     }
    
#     .filter-chip {
#         padding: 10px 22px;
#         background-color: #2A2F3B;
#         border: 1px solid #40444E;
#         border-radius: 20px;
#         cursor: pointer;
#         color: #E0E0E0;
#         font-weight: 500;
#         transition: all 0.2s ease;
#     }
    
#     .filter-chip:hover {
#         background-color: #3A3F4B;
#         color: white;
#     }
    
#     .filter-chip.active {
#         background-color: #FF6B6B;
#         border-color: #FF6B6B;
#         color: white;
#     }
    
#     /* Button styling - soft coral */
#     .stButton > button {
#         background-color: #FF6B6B;
#         color: white;
#         border: none;
#         border-radius: 20px;
#         padding: 12px 24px;
#         font-weight: 600;
#         transition: all 0.2s ease;
#     }
    
#     .stButton > button:hover {
#         background-color: #FF8E8E;
#         transform: scale(1.02);
#     }
    
#     /* Song cards - soft dark */
#     .song-card {
#         background-color: #1E222A;
#         border-radius: 12px;
#         padding: 18px;
#         margin: 15px 0;
#         border: 1px solid #343840;
#         transition: all 0.2s ease;
#     }
    
#     .song-card:hover {
#         border-color: #FF6B6B;
#     }
    
#     /* Emotion badges - pastel colors */
#     .emotion-badge {
#         display: inline-block;
#         padding: 5px 14px;
#         border-radius: 15px;
#         font-size: 12px;
#         font-weight: 600;
#         margin: 5px 0;
#     }
    
#     .happy { background: #FFD93D; color: #1E222A; }
#     .sad { background: #6BCB77; color: white; }
#     .love { background: #FF6B6B; color: white; }
#     .angry { background: #FF9E6B; color: white; }
#     .calm { background: #4D96FF; color: white; }
#     .excited { background: #FF78C4; color: white; }
#     .neutral { background: #9BA3AF; color: white; }
#     .friendly { background: #FFB26B; color: white; }
    
#     /* Input field styling */
#     .stTextInput>div>div>input {
#         background-color: #1E222A;
#         color: white;
#         border: 1px solid #343840;
#         border-radius: 15px;
#         padding: 12px;
#     }
    
#     .stTextInput>div>div>input:focus {
#         border-color: #FF6B6B;
#     }
    
#     /* Success message */
#     .stSuccess {
#         background-color: #6BCB77;
#         color: white;
#         border-radius: 10px;
#     }

#     /* YouTube button styling */
#     .youtube-button {
#         display: inline-block;
#         padding: 12px 24px;
#         background: #FF0000;
#         color: white;
#         text-decoration: none;
#         border-radius: 8px;
#         font-weight: bold;
#         font-size: 16px;
#         border: 2px solid #FF0000;
#         transition: all 0.3s;
#         text-align: center;
#         margin: 10px 0;
#     }
    
#     .youtube-button:hover {
#         background: #cc0000;
#         border-color: #cc0000;
#         color: white;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# def detect_search_type(text):
#     """Detect if user wants emotion-based or direct search"""
#     text_lower = text.lower()
    
#     # Keywords that indicate direct song/artist search
#     direct_search_keywords = [
#         'play', 'song', 'artist', 'music', 'by', 'search', 'find',
#         'gaan', 'gāna', 'गाना', 'संगीत', 'play song', 'play music'
#     ]
    
#     # Check if text contains direct search keywords
#     for keyword in direct_search_keywords:
#         if keyword in text_lower:
#             return 'direct'
    
#     # Check if it's a known artist or song name pattern
#     if any(word in text_lower for word in ['arjit', 'arijit', 'kishore', 'lata', 'taylor', 'weeknd']):
#         return 'direct'
    
#     # Default to emotion detection
#     return 'emotion'

# def detect_emotion_with_learning(text):
#     """Detect emotion with automatic active learning"""
#     # First try regular detection
#     emotion = detector.detect(text)
    
#     # For emoji-only inputs, skip active learning (high confidence)
#     emoji_only = True
#     for char in text:
#         if char not in EMOJI_EMOTION_MAP and not char.isspace():
#             emoji_only = False
#             break
    
#     if emoji_only:
#         return emotion  # Skip active learning for emoji-only inputs
    
#     confidence = get_confidence_score(text)
    
#     # If low confidence, use active learning
#     if confidence < 0.6:
#         # Try to find similar emotion
#         similar_emotion, similarity = active_learner.find_similar_emotion(text)
        
#         if similar_emotion:
#             if similarity >= 0.95:  # Exact match
#                 emotion = similar_emotion
#                 st.info(f"Exact match found: {emotion}")
#             else:  # Similar but not exact
#                 # Automatically learn this new pattern
#                 if active_learner.add_to_dataset(text, similar_emotion):
#                     st.success(f"✓ Automatically learned: '{text}' = {similar_emotion}")
#                 emotion = similar_emotion
#         else:
#             # Try keyword suggestion for completely new patterns
#             suggested = suggest_emotion(text)
#             if suggested:
#                 # Automatically learn this new pattern
#                 if active_learner.add_to_dataset(text, suggested):
#                     st.success(f"✓ Automatically learned new pattern: '{text}' = {suggested}")
#                 emotion = suggested
#             else:
#                 # Fallback to neutral for completely unknown patterns
#                 emotion = "neutral"
#                 st.info("Could not determine emotion, using neutral as default")
    
#     return emotion

# # Streamlit UI
# st.title("🎵 Music Recommendation Based on Mood")

# # Search history display in sidebar
# with st.sidebar:
#     st.subheader("Search History")
#     if st.session_state.search_history:
#         reversed_history = list(reversed(st.session_state.search_history))
#         history_df = pd.DataFrame(st.session_state.search_history)
#         # Format the timestamp for better display
#         if not history_df.empty and 'timestamp' in history_df.columns:
#             history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%H:%M:%S')
        
#         st.dataframe(history_df[['text', 'emotion', 'timestamp']], use_container_width=True)
        
#         # Add a clear history button
#         if st.button("Clear History", key="clear_history"):
#             st.session_state.search_history = []
#             st.rerun()

#     else:
#         st.write("No search history yet.")

# # Custom text input
# user_input = st.text_input("How are you feeling today?", placeholder="Type your mood or add emojis...", key="mood_input")

# # Analyze button
# analyze_btn = st.button("Analyze Mood and Recommend")

# # Divider
# st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# # Create filter chips using buttons in a single container
# with st.container():
#     st.markdown('<div class="filter-chips">', unsafe_allow_html=True)
#     cols = st.columns([1, 1, 1])  # Equal columns for buttons
#     categories = ["All", "Bollywood", "English"]
#     for i, category in enumerate(categories):
#         with cols[i]:
#             # Apply active class via CSS
#             button_style = f"background-color: {'#1DB954' if st.session_state.selected_category == category else '#4a4a4a'}; color: white; font-weight: {'700' if st.session_state.selected_category == category else '500'}; text-decoration: {'underline' if st.session_state.selected_category == category else 'none'};"
#             if st.button(category, key=f"category_{category}", help=f"Select {category} category"):
#                 if st.session_state.selected_category != category:
#                     st.session_state.selected_category = category
#                     logger.info(f"Category changed to: {category}")
#     st.markdown('</div>', unsafe_allow_html=True)

# if analyze_btn and user_input:
#     # Detect search type
#     search_type = detect_search_type(user_input)
#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
#     if search_type == 'direct':
#         # Direct song/artist search
#         st.info(f"🔍 Searching for: '{user_input}'")
#         recommendations = search_directly(user_input, 10)
#         st.session_state.direct_search_results = recommendations
#         st.session_state.search_type = 'direct'
#         st.session_state.search_history.append({
#             "text": user_input, 
#             "emotion": "direct_search", 
#             "timestamp": current_time
#         })
        
#     else:
#         # Emotion-based search
#         emotion = detect_emotion_with_learning(user_input)
#         st.session_state.search_history.append({
#             "text": user_input, 
#             "emotion": emotion, 
#             "timestamp": current_time
#         })
        
#         st.session_state.detected_emotion = emotion
#         st.session_state.selected_category = "All"
#         st.session_state.search_type = 'emotion'
#         st.success(f"Detected Emotion: {emotion.capitalize()}")

# # Show recommendations based on search type
# if st.session_state.get('search_type') == 'direct' and st.session_state.get('direct_search_results'):
#     recommendations = st.session_state.direct_search_results
#     st.success("🎵 Direct Search Results:")
    
# elif st.session_state.get('detected_emotion'):
#     with st.spinner("Finding the best songs for you..."):
#         time.sleep(1)
#         recommendations = get_recommendations(
#             st.session_state.detected_emotion, 
#             st.session_state.selected_category, 
#             10
#         )
#     st.success("Top Recommendations:")

# # Display recommendations
# if 'recommendations' in locals() and recommendations:
#     # Filter out songs with no Spotify link
#     valid_recommendations = [song for song in recommendations if song.get('spotify_link', "N/A") != "N/A"]
#     if not valid_recommendations:
#         st.warning("No songs with valid Spotify links found for this emotion.")
#     else:
#         for song in valid_recommendations:
#             title = song['title']
#             artist = song['artist']
#             youtube_link = song.get('youtube_link', "N/A")
#             spotify_link = song.get('spotify_link', "N/A")
            
#             # Extract Spotify track ID
#             spotify_track_id = None
#             if spotify_link != "N/A" and 'open.spotify.com/track/' in spotify_link:
#                 spotify_track_id = spotify_link.split('open.spotify.com/track/')[1].split('?')[0]
            
#             # YouTube display handling
#             if youtube_link != "N/A" and 'youtube.com/results' in youtube_link:
#                 youtube_display = f"""
#                 <div style="text-align: center;">
#                     <a href='{youtube_link}' target='_blank' class='youtube-button'>
#                     🔍 Search "{song['title']}" on YouTube
#                     </a>
#                 </div>
#                 """
#             else:
#                 youtube_display = """
#                 <div style="text-align: center; color: #888; font-style: italic;">
#                     YouTube search not available
#                 </div>
#                 """
            
#             # Create song card
#             st.markdown(
#                 f"""
#                 <div style="margin: 10px; padding: 15px; border: 1px solid #444; border-radius: 10px; background-color: #1E222A; color: white; position: relative;">
#                     <div style="position: absolute; top: 15px; right: 15px;">
#                         <a href="{youtube_link}" target="_blank" style="text-decoration: none; margin-left: 10px;">
#                             <i class="fab fa-youtube" style="font-size: 24px; color: #FF0000;"></i>
#                         </a>
#                         <a href="{spotify_link}" target="_blank" style="text-decoration: none;">
#                             <i class="fab fa-spotify" style="font-size: 24px; color: #1DB954;"></i>
#                         </a>
#                     </div>
#                     <div style="margin-bottom: 15px;">
#                         <h3 style="margin: 0 0 5px 0; color: #fff;">{title}</h3>
#                         <p style="margin: 0 0 5px 0; color: #ccc;">by {artist}</p>
#                         <span class="emotion-badge {song.get('emotion', 'neutral')}">{song.get('emotion', 'neutral').capitalize()}</span>
#                     </div>
#                     <div style="display: flex; gap: 15px; flex-wrap: wrap;">
#                         <div style="flex: 1; min-width: 300px; text-align: center;">
#                             {youtube_display}
#                         </div>
#                         <div style="flex: 1; min-width: 300px;">
#                             {"<iframe src='https://open.spotify.com/embed/track/" + spotify_track_id + "' width='100%' height='85' frameborder='0' allowtransparency='true' allow='encrypted-media' style='border-radius: 5px;'></iframe>" if spotify_track_id else "<p style='color: #888; text-align: center;'>Spotify link not available</p>"}
#                         </div>
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
# elif 'recommendations' in locals() and not recommendations:
#     st.warning("No recommendations found. Please try a different mood or search term.")