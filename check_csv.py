# # check_csv.py
# import pandas as pd

# print("=== Emotion Dataset ===")
# emotion_df = pd.read_csv('emotion_dataset.csv')
# print(f"Shape: {emotion_df.shape}")
# print(f"Columns: {emotion_df.columns.tolist()}")
# print(f"First 5 rows:\n{emotion_df.head()}")

# print("\n=== Music Dataset ===")
# music_df = pd.read_csv('music_dataset.csv')
# print(f"Shape: {music_df.shape}")
# print(f"Columns: {music_df.columns.tolist()}")
# print(f"First 5 rows:\n{music_df.head()}")
# check_csv.py
import pandas as pd

print("=== Emotion Dataset ===")
try:
    emotion_df = pd.read_csv('data/emotion_dataset.csv')  # Correct path
    print(f"Shape: {emotion_df.shape}")
    print(f"Columns: {emotion_df.columns.tolist()}")
    print(f"First 5 rows:\n{emotion_df.head()}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== Music Dataset ===")
try:
    music_df = pd.read_csv('data/music_dataset.csv')  # Correct path
    print(f"Shape: {music_df.shape}")
    print(f"Columns: {music_df.columns.tolist()}")
    print(f"First 5 rows:\n{music_df.head()}")
except Exception as e:
    print(f"Error: {e}")
