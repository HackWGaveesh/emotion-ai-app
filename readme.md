# <img src="Logo.png" alt="Emotion AI Logo" width="50"> Emotion AI: Real-time Emotion Recognition with Future Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://emotionai-hackwgaveesh.streamlit.app/)

## Overview

Emotion AI is an advanced Streamlit web application that performs real-time emotion recognition from images and webcam feeds, and uniquely offers future emotion prediction capabilities. Leveraging a fine-tuned Vision Transformer (ViT) for accurate emotion detection and a Long Short-Term Memory (LSTM) neural network for forecasting emotional trends, this app provides a comprehensive insight into emotional states.

## Features

  * **Image-based Emotion Analysis**: Upload an image and get instant emotion probabilities across 8 different categories.
  * **Real-time Webcam Analysis**: Utilize your device's camera for live emotion detection and continuous tracking of emotional shifts.
  * **Future Emotion Prediction**: An LSTM model learns from your emotional history to predict likely future emotional states.
  * **Interactive Analytics Dashboard**: Visualize your emotional journey over time with detailed charts, including:
      * Emotional Journey Timeline (Line Chart)
      * Emotion Intensity Heatmap
      * Dominant Emotions Distribution (Pie Chart)
      * Emotional State Radar Chart
  * **8 Emotion Categories**: Detects and predicts "amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", and "sadness". [cite: 1]
  * **Clearable History**: Option to clear all collected emotion data.
  * **Downloadable Data**: Export your emotional history as a CSV file for further analysis.

## Technologies Used

  * **Streamlit**: For creating the interactive web application. [cite: 1]
  * **PyTorch**: Deep learning framework for ViT and LSTM models. [cite: 1]
  * **Hugging Face Transformers**: For the Vision Transformer (ViT) model. [cite: 1]
  * **Pillow (PIL)**: For image processing. [cite: 1]
  * **OpenCV**: For webcam integration. [cite: 1]
  * **Pandas**: For data manipulation and history tracking. [cite: 1]
  * **Plotly**: For rich and interactive data visualizations. [cite: 1]
  * **NumPy**: For numerical operations. [cite: 1]

## How it Works

1.  **Emotion Detection (ViT)**: A pre-trained Vision Transformer model, fine-tuned for facial emotion classification, processes input images or webcam frames. It outputs probabilities for 8 distinct emotions. [cite: 1]
2.  **Emotional History Tracking**: The detected emotion probabilities are logged with timestamps, creating a continuous emotional history. [cite: 1]
3.  **Future Prediction (LSTM)**: An LSTM neural network is continuously updated with the emotional history. It learns sequences and patterns in your emotional data to predict the probabilities of future emotional states. [cite: 1]
4.  **Interactive Visualization**: Streamlit and Plotly are used to present the current emotions, predicted future emotions, and historical trends through various interactive charts and metrics. [cite: 1]

## Setup and Installation

To run this application locally, follow these steps:

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/HackWGaveesh/emotion-ai-app.git
    cd emotion-ai-app
    ```

2.  **Create a Virtual Environment (Recommended)**:

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Pre-trained Models**:

      * `emotion_vit_finetuned.pth`: This is the fine-tuned ViT model's weights.
      * `saved_model/`: This directory contains the ViT image processor.

    You can typically find these files within the GitHub repository if not already present. Ensure they are in the root directory and `saved_model` directory respectively.

5.  **Run the Streamlit App**:

    ```bash
    streamlit run app.py
    ```

    The application will open in your default web browser.

## Project Structure

```
.
├── .devcontainer/                # Development container configuration
├── saved_model/                  # Contains ViTImageProcessor (pre-trained processor) [cite: 1]
│   └── (processor files)
├── Default_image.png             # Default image for demo purposes
├── Logo.png                      # Application logo
├── app.py                        # Main Streamlit application file [cite: 1]
├── emotion_vit_finetuned.pth     # Fine-tuned Vision Transformer model weights [cite: 1]
├── readme.md                     # This README file
└── requirements.txt              # Python dependencies [cite: 1]
```

## Usage

### Image Analysis Tab

1.  Navigate to the "Image Analysis" tab.
2.  Click "Choose an image with a face..." to upload an image.
3.  The app will display the uploaded image and analyze the emotions present, showing a detailed breakdown and a radar chart.
4.  If enough history is accumulated (at least 3 entries), it will also display future emotion predictions.

### Real-time Webcam Tab

1.  Go to the "Real-time Webcam" tab.
2.  Click "▶️ Start Camera" to activate your webcam.
3.  The app will display your webcam feed and continuously analyze your emotions in real-time, showing current and predicted future emotional states.
4.  Click "⏹️ Stop Camera" to stop the webcam feed.
5.  Click "🧹 Clear History" to reset the webcam-specific emotion history.

### Analytics Dashboard Tab

1.  Visit the "Analytics Dashboard" tab.
2.  This section provides a comprehensive overview of all collected emotional data (from both image uploads and webcam).
3.  View interactive charts showing your emotional journey, intensity, and dominant emotions.
4.  You can view the raw data and download it as a CSV file.
5.  There's an option to "Clear All History Data" to reset the entire session's emotional data.

## Contributing

Contributions are welcome\! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.


-----

Feel free to reach out with any questions or feedback\!

Created with ❤️ by HackWGaveesh
