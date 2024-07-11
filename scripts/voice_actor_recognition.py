import librosa
import numpy as np
import faiss
import sys
import time

# List to store voice actor names and corresponding MFCC vectors
voice_actors = []
faiss_index = None

# Function to add voice actor to the database
def add_voice_actor(name, audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    voice_actors.append((name, mfcc_mean))

# Function to initialize FAISS index
def initialize_faiss():
    global faiss_index
    dimension = voice_actors[0][1].shape[0]
    faiss_index = faiss.IndexFlatL2(dimension)
    for _, mfcc_mean in voice_actors:
        faiss_index.add(np.array([mfcc_mean]).astype('float32'))

# Function to calculate distance
def calculate_distance(sample_path):
    y, sr = librosa.load(sample_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    D, I = faiss_index.search(np.array([mfcc_mean]).astype('float32'), k=1)
    closest_index = I[0][0]
    closest_distance = D[0][0]
    closest_actor_name = voice_actors[closest_index][0]
    return closest_actor_name, closest_distance

def main(audio_file_path):
    start_time = time.time()

    # Add voice actors to the database
    add_voice_actor('Seth McFarlane', 'voice_actor_1.mp3')
    add_voice_actor('Matt Groening', 'voice_actor_2.wav')
    add_voice_actor('Mike Myers', 'voice_actor_3.wav')

    # Initialize FAISS index
    initialize_faiss()

    # Perform the search with the provided sample
    closest_actor_name, closest_distance = calculate_distance(audio_file_path)
    threshold = 100.0  # Example threshold value; adjust based on your data and requirements

    if closest_distance > threshold:
        result = f"No close match found. Closest actor: {closest_actor_name} with distance: {closest_distance}"
    else:
        result = f"Most similar voice actor: {closest_actor_name} with distance: {closest_distance}"

    end_time = time.time()
    elapsed_time = end_time - start_time
    result += f"\nElapsed time: {elapsed_time:.2f} seconds"

    with open("output.txt", "w") as output_file:
        output_file.write(result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python voice_actor_recognition.py <audio_file_path>")
        sys.exit(1)
    main(sys.argv[1])
