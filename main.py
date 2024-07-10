import librosa
import numpy as np
import faiss
import time

# List to store voice actor names and corresponding MFCC vectors
voice_actors = []

# Function to add voice actor to the database
def add_voice_actor(name, audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    voice_actors.append((name, mfcc_mean))

# Function to find the closest voice actor
def find_closest_voice_actor(new_audio_path):
    y_new, sr_new = librosa.load(new_audio_path, sr=None)
    mfcc_new = np.mean(librosa.feature.mfcc(y=y_new, sr=sr_new, n_mfcc=13), axis=1)
    D, I = index.search(np.array([mfcc_new]).astype('float32'), k=1)
    closest_index = I[0][0]
    closest_distance = D[0][0]
    closest_actor_name = voice_actors[closest_index][0]
    return closest_actor_name, closest_distance

def main():
    start_time = time.time()

    # Add voice actors to the database
    add_voice_actor('Seth McFarlane', 'voice_actor_1.mp3')
    add_voice_actor('Matt Groening', 'voice_actor_2.wav')
    add_voice_actor('Mike Myers', 'voice_actor_3.wav')

    # Initialize FAISS index
    dimension = voice_actors[0][1].shape[0]
    global index
    index = faiss.IndexFlatL2(dimension)

    # Add MFCC vectors to the FAISS index
    for _, mfcc_mean in voice_actors:
        index.add(np.array([mfcc_mean]).astype('float32'))

    # Perform the search with a new sample
    closest_actor_name, closest_distance = find_closest_voice_actor('seth_mcfarlane_test.mp3')
    print(f"Most similar voice actor: {closest_actor_name} with distance: {closest_distance}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
