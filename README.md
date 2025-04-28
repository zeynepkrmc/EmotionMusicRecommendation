# 🎵 Moodify: Music Recommendation Based on Face Emotion Recognition

Moodify is a fun and innovative project that combines facial emotion recognition with music recommendation. Using a Convolutional Neural Network (CNN) for emotion detection and integrating Spotify Web API, Moodify analyzes real-time facial expressions through a webcam and generates personalized music playlists that match the user's emotional state.

# 📌 Project Overview

- Goal: Enhance user experience by providing personalized music suggestions based on detected emotions.
- Key Features:
  - Real-time emotion detection via webcam.
  - Automatic playlist generation based on detected emotions.
  - Seamless integration with Spotify's vast music database.
- Target Emotions: Happy, Sad, Angry, Fear, Disgust, Neutral, Surprised.

# 🧠 How It Works ?

- 1. Facial Emotion Recognition:
  - The user’s facial expression is captured through a webcam.
  - A CNN model trained on the FER2013 dataset classifies the detected emotion.
  - 
- 2. Music Recommendation:
  - The recognized emotion is used to fetch playlists from Spotify.
  - Music is recommended according to the user's emotional profile, using the Spotify Web API (Spotipy library).

- 3. User Interface:
  - A web-based interface (HTML, CSS, Bootstrap) allows users to see real-time emotion detection and receive instant music recommendations.
 
# 🛠️ Technologies Used

- Python (Core programming)
- TensorFlow / Keras (Model development)
- OpenCV (Real-time webcam input)
- Spotipy (Spotify Web API integration)
- HTML / CSS / Bootstrap (Frontend)
- FER2013 Dataset (Emotion recognition training)

# 📊 Model Architecture

- 4 Convolutional Layers with ReLU Activation
- 3 MaxPooling Layers
- 3 Dropout Layers (to prevent overfitting)
- 2 Dense Layers:
  - First Dense: 1024 neurons
  - Output Dense: 7 neurons (Softmax activation for emotion classification)
    
-> Training Highlights:

- Optimizer: Adam
- Loss Function: Categorical Cross-Entropy
- Batch Size: 64
- Epochs: 100
- Data Augmentation: Applied (panning, zooming, scaling)

  # 📈 Model Training & Performance
  
- Multiple training sessions were conducted to tune the model.
- The 4th Training Session was selected due to balanced training and validation accuracies:
- Training Accuracy: 55%
- Validation Accuracy: 57%
- Successfully avoided overfitting with dropout and data augmentation strategies.

  # 🔗 Spotify API Integration
  
* Authentication: SpotifyClientCredentials
* Main Usage: Spotify's Search API to find playlists/songs related to the detected emotion.
  * https://developer.spotify.com/documentation/web-api
* Playlist Generation: Emotion → Search appropriate playlist → Shuffle and recommend tracks.

# 🧩 Challenges Encountered

- Overfitting: Solved by applying multiple Dropout layers and data augmentation.
- Spotify API Authorization Errors: Worked around by shifting from the /recommendations endpoint to the /search endpoint.
- Emotion-Matching Songs: Since Spotify does not provide direct emotion-based recommendations, custom logic was developed to match moods to playlists.

# 🚀 Future Improvements

- Improve the emotion detection model with larger, diverse datasets.
- Include user feedback to refine and personalize music recommendations.
- Deploy Moodify as a mobile application.
- Collaborate directly with Spotify for deeper integration and better playlist matching.
- Enhance real-time processing speed for a smoother user experience.

# 📚 References

- Spotify Web API Documentation

- FER2013 Dataset (Kaggle)

- Keras Layers API

# 👥 Authors

- Gizem EROL
- Selvinaz Zeynep KIYIKCI
- Zeynep Sude KIRMACI

* CEN435 – Artificial Intelligence Project
Autumn 2024

Moodify brings technology and emotions together — listen to your feelings! 🎶
