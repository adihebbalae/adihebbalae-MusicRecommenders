# adihebbalae-MusicRecommenders
# Music Recommender System

## Overview
This project implements a **Music Recommender System** that uses machine learning techniques to classify and recommend music based on song metadata and musical timbre. The system leverages **Convolutional Neural Networks (CNNs)**, **K-Nearest Neighbors (KNN)**, and **Recurrent Neural Networks (RNNs)** to analyze and categorize songs, providing tailored recommendations.

## Features
- **Song Metadata Analysis**: Utilizes song metadata (e.g., genre, artist, year, tempo) for classification.
- **Musical Timbre Analysis**: Processes audio features using CNNs to capture the timbre and texture of music.
- **Sequential Data Modeling**: RNNs are employed to model temporal dependencies in musical sequences.
- **Similarity-Based Recommendations**: Implements KNN to recommend songs based on user preferences and song similarities.

## Technologies
- **Programming Language**: Python
- **Libraries and Frameworks**:
  - TensorFlow/Keras: For deep learning models (CNN, RNN).
  - Scikit-learn: For KNN and preprocessing tasks.
  - Librosa: For audio feature extraction.
  - NumPy, Pandas: For data manipulation.
  - Matplotlib, Seaborn: For data visualization.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/username/music-recommender.git
   cd music-recommender
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and preprocess the dataset (see [Dataset](#dataset) section).

## Dataset
The system uses a dataset containing:
- Song metadata (e.g., artist, album, year, genre).
- Raw audio files or precomputed audio features (e.g., MFCCs, chroma).

Example datasets:
- [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/)
- [GTZAN Genre Dataset](http://marsyas.info/downloads/datasets.html)

Ensure the dataset is organized as follows:
```
./data/
  metadata.csv
  audio/
    song1.wav
    song2.wav
```

## How It Works
### 1. Data Preprocessing
- **Metadata**: Clean and normalize fields such as genre, tempo, and artist.
- **Audio Features**: Extract features (e.g., MFCCs, chroma, spectral contrast) using Librosa.
- **Train-Test Split**: Divide the data into training, validation, and test sets.

### 2. Models
#### **CNN**
- Input: Extracted audio features (e.g., spectrograms).
- Purpose: Classify music based on timbre and texture.

#### **KNN**
- Input: Combined feature vector (metadata + audio features).
- Purpose: Recommend similar songs by calculating distances in feature space.

#### **RNN**
- Input: Sequential features (e.g., time-series data from audio).
- Purpose: Capture temporal dependencies for music classification.

### 3. Recommendation Engine
- Combines the outputs of the models.
- Adjusts weights for personalized recommendations based on user preferences.

## Usage
1. Preprocess the data:
   ```bash
   python preprocess.py
   ```
2. Train the models:
   ```bash
   python train.py --model cnn
   python train.py --model rnn
   python train.py --model knn
   ```
3. Generate recommendations:
   ```bash
   python recommend.py --user_id <USER_ID>
   ```

## Results
- Evaluation metrics include accuracy, precision, recall, and F1-score for classification tasks.
- KNN-based recommendations are evaluated using precision-at-k and recall-at-k metrics.

## Future Work
- Incorporate transformer-based models for improved sequential data analysis.
- Enhance the user interface for real-time music recommendations.
- Expand metadata features with lyrics and user-generated tags.


