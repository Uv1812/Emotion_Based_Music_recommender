# Music Recommendation Based on Mood

A smart **Streamlit web app** that detects your emotion from text (with emojis!) and recommends **perfect songs** from **Spotify** and **YouTube** — supporting **English** and **Bollywood** music.

Feeling in love? Sad? Groovy? Just type how you feel — get personalized songs instantly!

**Live Demo**: [https://your-app-link.streamlit.app](https://your-app-link.streamlit.app) *(replace with your deployed link)*

![App Preview](assets/preview.gif)  
*(Add a GIF or screenshot of your app in `assets/` folder)*

---

### Features

- **Emotion Detection** from natural language + emojis (44+ emotions supported)
- **Smart Music Recommendations** using Spotify audio features (valence, energy)
- **Three Tabs**:  
  - **All** → Mixed English + Bollywood songs (default)  
  - **Bollywood** → Only Hindi songs  
  - **English** → Only English songs  
- **Spotify Embed Player** + **30-sec Preview Audio**
- **YouTube Video Embed** with official music videos
- **Like/Dislike Feedback** system
- **Export Recommendations** to CSV
- **Search History** in sidebar
- **Dark Theme** with beautiful song cards
- **Persistent Cache** – faster repeats, fewer API calls
- **Fallback Songs** – works offline or without API keys

---

### Tech Stack

| Technology        | Purpose                          |
|-------------------|----------------------------------|
| Python            | Backend logic                    |
| Streamlit         | Web interface                    |
| Spotify API       | Song data + audio features       |
| YouTube Data API  | Video links                      |
| Scikit-learn      | Emotion classification           |
| NLTK              | Text preprocessing               |
| Pandas            | Data handling                    |
| Spotipy           | Spotify API wrapper              |
| Joblib            | Model persistence                |

---

### Project Structure
