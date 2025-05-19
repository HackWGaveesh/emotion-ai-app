import streamlit as st
from streamlit.components.v1 import html  # Import for rendering HTML
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import time
import io
import cv2
from collections import deque
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration with custom theme
st.set_page_config(
    page_title="Emotion Recognition with Future Prediction",
    page_icon="😀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e7aff;
        color: white;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .emotion-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .future-prediction {
        border-left: 4px solid #4e7aff;
        padding-left: 20px;
    }
    .emotion-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        margin-right: 5px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
MODEL_PATH = "emotion_vit_finetuned.pth"
PROCESSOR_PATH = "saved_model"

# Define emotion classes
CLASSES = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
NUM_CLASSES = len(CLASSES)

# Emotion colors for consistent visualization
EMOTION_COLORS = {
    "amusement": "#FF9E00",    # Vibrant orange
    "anger": "#E53935",        # Deep red
    "awe": "#8E24AA",          # Purple
    "contentment": "#43A047",  # Green
    "disgust": "#795548",      # Brown
    "excitement": "#FFC107",   # Amber
    "fear": "#7E57C2",         # Deep purple
    "sadness": "#1E88E5"       # Blue
}

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM Model for emotion sequence prediction
class EmotionLSTM(nn.Module):
    def __init__(self, input_size=NUM_CLASSES, hidden_size=128, num_layers=2, output_size=NUM_CLASSES):
        super(EmotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, hidden_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(hidden_size//2, output_size)
        
        # Softmax for probabilities
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.output_layer(out)
        
        # Apply softmax for probabilities
        probs = self.softmax(out)
        
        return out, probs

# Function to load the ViT model
@st.cache_resource
def load_vit_model():
    try:
        processor = ViTImageProcessor.from_pretrained(PROCESSOR_PATH)
        model_name = "dima806/facial_emotions_image_detection"
        model = ViTForImageClassification.from_pretrained(model_name)
        
        if model.classifier.out_features != NUM_CLASSES:
            st.write(f"Adjusting classifier from {model.classifier.out_features} to {NUM_CLASSES} classes")
            original_weights = model.classifier.weight.data
            original_bias = model.classifier.bias.data
            model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
            with torch.no_grad():
                min_classes = min(original_weights.shape[0], NUM_CLASSES)
                model.classifier.weight.data[:min_classes] = original_weights[:min_classes]
                model.classifier.bias.data[:min_classes] = original_bias[:min_classes]
                
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        
        return model, processor
    except Exception as e:
        st.error(f"Error loading ViT model: {e}")
        return None, None

# Function to create and initialize LSTM model
@st.cache_resource
def create_lstm_model():
    try:
        lstm_model = EmotionLSTM().to(device)
        lstm_model.eval()
        return lstm_model
    except Exception as e:
        st.error(f"Error creating LSTM model: {e}")
        return None

# Function to process an image and predict emotion
def predict_emotion(image, model, processor):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs).logits
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        probabilities, indices = torch.topk(probs, k=len(CLASSES))
        results = {}
        for i, (prob, idx) in enumerate(zip(probabilities[0], indices[0])):
            results[CLASSES[idx.item()]] = prob.item()
            
        return results, outputs[0].cpu().numpy()
    except Exception as e:
        st.error(f"Error during emotion prediction: {e}")
        return None, None

# Function to predict future emotions using LSTM
def predict_future_emotions(emotion_history, lstm_model, num_predictions=5):
    try:
        emotion_tensor = torch.FloatTensor(emotion_history).unsqueeze(0).to(device)
        future_emotions = []
        current_input = emotion_tensor
        
        for _ in range(num_predictions):
            with torch.no_grad():
                _, probs = lstm_model(current_input)
            next_emotion = probs[0].cpu().numpy()
            future_emotions.append(next_emotion)
            next_emotion_tensor = torch.FloatTensor(next_emotion).unsqueeze(0).unsqueeze(0)
            current_input = torch.cat([current_input[:, 1:, :], next_emotion_tensor], dim=1)
            
        return future_emotions
    except Exception as e:
        st.error(f"Error during future emotion prediction: {e}")
        return None

# Function to convert webcam frame to PIL Image
def frame_to_pil(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)

# Visualize emotion predictions using plotly
def visualize_emotions_plotly(current_emotion):
    sorted_items = sorted(current_emotion.items(), key=lambda x: x[1], reverse=True)
    emotion_names = [item[0] for item in sorted_items]
    emotion_values = [item[1] for item in sorted_items]
    bar_colors = [EMOTION_COLORS[emotion] for emotion in emotion_names]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=emotion_names,
        y=emotion_values,
        marker_color=bar_colors,
        text=[f'{value:.2f}' for value in emotion_values],
        textposition='outside',
        hoverinfo='text',
        hovertext=[f'{name}: {value:.2f}' for name, value in zip(emotion_names, emotion_values)],
    ))
    
    fig.update_layout(
        title={'text': 'Current Emotion Analysis', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 22, 'color': '#2c3e50', 'family': 'Arial, sans-serif'}},
        xaxis_title='Emotion',
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_tickangle=-45,
        height=400,
        shapes=[dict(type="rect", xref="x", yref="paper", x0=-0.5 + emotion_names.index(emotion_names[0]) - 0.4, y0=0, x1=-0.5 + emotion_names.index(emotion_names[0]) + 0.4, y1=1, fillcolor="#f8f9fa", opacity=0.4, layer="below", line_width=0)]
    )
    
    return fig

# Visualize future emotion predictions using plotly
def visualize_future_emotions_plotly(future_emotions):
    if not future_emotions:
        return None
        
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter"}, {"type": "polar"}]], column_widths=[0.6, 0.4], subplot_titles=("Emotion Probability Trends", "Future Emotional State"))
    time_steps = [f"T+{i+1}" for i in range(len(future_emotions))]
    
    for i, emotion in enumerate(CLASSES):
        values = [pred[i] for pred in future_emotions]
        fig.add_trace(go.Scatter(x=time_steps, y=values, mode='lines+markers', name=emotion, line=dict(color=EMOTION_COLORS[emotion], width=3), marker=dict(size=8)), row=1, col=1)
    
    final_state = future_emotions[-1]
    fig.add_trace(go.Scatterpolar(r=final_state, theta=CLASSES, fill='toself', name='Final State', line=dict(color='rgba(32, 84, 147, 0.8)', width=2), fillcolor='rgba(32, 84, 147, 0.3)'), row=1, col=2)
    
    fig.update_layout(
        title={'text': 'Future Emotion Prediction', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 22, 'color': '#2c3e50', 'family': 'Arial, sans-serif'}},
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0.0, font=dict(size=10)),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=500,
        margin=dict(l=60, r=30, t=100, b=100),
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0)',
    )
    fig.update_xaxes(title_text="Future Time Steps", row=1, col=1)
    fig.update_yaxes(title_text="Probability", range=[0, 1], row=1, col=1)
    
    # Adjust subplot titles
    for annotation in fig['layout']['annotations']:
        if annotation['text'] == "Future Emotional State":
            annotation['y'] = 1.05  # Move higher
            annotation['font']['size'] = 14  # Larger font for boldness
    
    return fig

# Train LSTM with emotion history
def train_lstm_with_sample(lstm_model, emotion_history):
    if len(emotion_history) < 5:
        return lstm_model, None
    
    X, y = [], []
    seq_length = 3
    for i in range(len(emotion_history) - seq_length):
        X.append(emotion_history[i:i+seq_length])
        y.append(emotion_history[i+seq_length])
    
    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    lstm_model.train()
    losses = []
    
    batch_size = min(8, len(X))
    num_epochs = 10
    
    for epoch in range(num_epochs):
        indices = torch.randperm(len(X))
        total_loss = 0
        for start_idx in range(0, len(X), batch_size):
            end_idx = min(start_idx + batch_size, len(X))
            batch_indices = indices[start_idx:end_idx]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            optimizer.zero_grad()
            _, probs = lstm_model(batch_X)
            loss = criterion(probs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / ((len(X) + batch_size - 1) // batch_size)
        losses.append(avg_loss)
    
    lstm_model.eval()
    return lstm_model, losses

# Display current emotions
def display_current_emotion(emotion_probs):
    top_emotion = max(emotion_probs.items(), key=lambda x: x[1])
    emoji_map = {"amusement": "😄", "anger": "😡", "awe": "😲", "contentment": "😌", "disgust": "🤢", "excitement": "🤩", "fear": "😨", "sadness": "😢"}
    
    html_content = f"""
    <div style="background-color: white; border-radius: 15px; padding: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 48px; margin-right: 15px;">{emoji_map.get(top_emotion[0], '😐')}</span>
            <div>
                <h2 style="margin: 0; color: {EMOTION_COLORS[top_emotion[0]]};">{top_emotion[0].capitalize()}</h2>
                <p style="margin: 0; font-size: 1.2em; opacity: 0.8;">{top_emotion[1]:.2f} confidence</p>
            </div>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;">
    """
    for emotion, prob in sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True):
        if emotion != top_emotion[0]:
            html_content += f"""
            <span style="background-color: {EMOTION_COLORS[emotion]}; color: white; padding: 5px 10px; 
                  border-radius: 15px; font-size: 0.9em;">{emotion}: {prob:.2f}</span>
            """
    html_content += """
        </div>
    </div>
    """
    return html_content

# Display future emotion predictions
def display_future_emotions_table(future_emotions):
    if not future_emotions:
        return ""
    
    future_dominant = []
    for i, future in enumerate(future_emotions):
        dominant_idx = np.argmax(future)
        dominant_emotion = CLASSES[dominant_idx]
        confidence = future[dominant_idx]
        future_dominant.append((dominant_emotion, confidence))
    
    html_content = """
    <div style="background-color: white; border-radius: 15px; padding: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin-top: 20px;">
        <h3 style="margin-top: 0; color: #2c3e50;">Emotional Journey Forecast</h3>
        <div style="display: flex; justify-content: space-between; overflow-x: auto; padding-bottom: 10px;">
    """
    emoji_map = {"amusement": "😄", "anger": "😡", "awe": "😲", "contentment": "😌", "disgust": "🤢", "excitement": "🤩", "fear": "😨", "sadness": "😢"}
    
    for i, (emotion, confidence) in enumerate(future_dominant):
        html_content += f"""
        <div style="min-width: 100px; text-align: center; padding: 10px; margin-right: 10px; 
                  border-radius: 10px; background-color: rgba({int(EMOTION_COLORS[emotion][1:3], 16)}, 
                  {int(EMOTION_COLORS[emotion][3:5], 16)}, {int(EMOTION_COLORS[emotion][5:7], 16)}, 0.1);
                  border-bottom: 3px solid {EMOTION_COLORS[emotion]};">
            <div style="font-size: 12px; opacity: 0.8;">T+{i+1}</div>
            <div style="font-size: 24px; margin: 5px 0;">{emoji_map.get(emotion, '😐')}</div>
            <div style="font-weight: bold; color: {EMOTION_COLORS[emotion]};">{emotion.capitalize()}</div>
            <div style="font-size: 12px;">{confidence:.2f}</div>
        </div>
        """
    html_content += """
        </div>
    </div>
    """
    return html_content

# Visualize emotion trends over time using plotly
def visualize_emotion_trends_plotly(df):
    fig = go.Figure()
    for emotion in CLASSES:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df[emotion], mode='lines', name=emotion, line=dict(color=EMOTION_COLORS[emotion], width=2), hoverinfo='text', hovertext=[f"{ts.strftime('%H:%M:%S')}<br>{emotion}: {value:.2f}" for ts, value in zip(df['timestamp'], df[emotion])]))
    
    fig.update_layout(
        title={'text': 'Emotional Journey Timeline', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 22, 'color': '#2c3e50'}},
        xaxis_title='Time',
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5),
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0)',
        margin=dict(l=60, r=30, t=100, b=100),
        height=500
    )
    return fig

# Create heatmap of emotion history
def create_emotion_heatmap(df):
    data = df[CLASSES].values
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=df['timestamp'].dt.strftime('%H:%M:%S'),
        y=CLASSES,
        colorscale=[[0, 'rgb(247, 251, 255)'], [0.1, 'rgb(222, 235, 247)'], [0.2, 'rgb(198, 219, 239)'], [0.3, 'rgb(158, 202, 225)'], [0.4, 'rgb(107, 174, 214)'], [0.5, 'rgb(66, 146, 198)'], [0.6, 'rgb(33, 113, 181)'], [0.7, 'rgb(8, 81, 156)'], [0.8, 'rgb(8, 48, 107)'], [1, 'rgb(3, 19, 43)']],
        hoverongaps=False,
        hoverinfo='text',
        hovertext=[[f"{emotion}: {value:.2f}<br>{ts.strftime('%H:%M:%S')}" for ts, value in zip(df['timestamp'], df[emotion])] for emotion in CLASSES]
    ))
    fig.update_layout(
        title={'text': 'Emotion Intensity Over Time', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 22, 'color': '#2c3e50'}},
        xaxis_title='Time',
        yaxis_title='Emotion',
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0)',
        margin=dict(l=60, r=30, t=100, b=50),
        height=400
    )
    return fig

# Create emotion radar chart
def create_emotion_radar_chart(emotion_data):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(emotion_data.values()), theta=list(emotion_data.keys()), fill='toself', name='Current Emotional State', line=dict(color='rgba(32, 84, 147, 0.8)', width=2), fillcolor='rgba(32, 84, 147, 0.3)'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title={'text': 'Emotional State Radar', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 18, 'color': '#2c3e50'}},
        margin=dict(l=40, r=40, t=60, b=40),
        height=350,
        paper_bgcolor='rgba(255,255,255,0)'
    )
    return fig

# Create dominant emotion pie chart
def create_dominant_emotion_chart(df):
    dominant_counts = {}
    for _, row in df.iterrows():
        emotion_data = {emotion: row[emotion] for emotion in CLASSES}
        dominant_emotion = max(emotion_data.items(), key=lambda x: x[1])[0]
        dominant_counts[dominant_emotion] = dominant_counts.get(dominant_emotion, 0) + 1
    
    emotions = list(dominant_counts.keys())
    counts = list(dominant_counts.values())
    colors = [EMOTION_COLORS[emotion] for emotion in emotions]
    
    fig = go.Figure(data=[go.Pie(labels=emotions, values=counts, marker=dict(colors=colors), textinfo='label+percent', insidetextorientation='radial', hole=0.4)])
    fig.update_layout(
        title={'text': 'Dominant Emotions Distribution', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 18, 'color': '#2c3e50'}},
        margin=dict(l=20, r=20, t=60, b=20),
        height=350,
        paper_bgcolor='rgba(255,255,255,0)'
    )
    return fig

# Main Streamlit app function
def main():
    with st.sidebar:
        st.image("Logo.png", width=150)
        st.title("Emotion AI")
        st.markdown("---")
        st.write("Welcome to Emotion AI - an advanced emotion recognition system with future prediction capabilities.")
        st.write("📊 Analyze emotions from images or webcam")
        st.write("🔮 Predict future emotional states")
        st.write("📈 Track emotion history and patterns")
        st.markdown("---")
        st.write("Built with: Vision Transformer & LSTM")
        with st.expander("About this App"):
            st.write("""
            This application uses a Vision Transformer (ViT) model to detect emotions from facial expressions.
            The app analyzes 8 different emotions:
            - Amusement 😄
            - Anger 😡
            - Awe 😲
            - Contentment 😌
            - Disgust 🤢
            - Excitement 🤩
            - Fear 😨
            - Sadness 😢
            Future emotion predictions are powered by LSTM neural networks that learn from your emotional patterns.
            """)
    
    st.title("📊 Emotion Recognition with Future Prediction")
    st.markdown("""
    <p style="font-size: 1.2em; color: #555;">
        Upload an image or use your webcam to detect emotions and predict your future emotional journey.
    </p>
    """, unsafe_allow_html=True)
    
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    if 'history_df' not in st.session_state:
        st.session_state.history_df = pd.DataFrame(columns=['timestamp'] + CLASSES)
    
    with st.spinner("Loading emotion recognition models..."):
        vit_model, processor = load_vit_model()
        lstm_model = create_lstm_model()
        if not vit_model or not processor or not lstm_model:
            st.error("Failed to load models. Please check the paths and try again.")
            return
    
    tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Real-time Webcam", "📈 Analytics Dashboard"])
    
    with tab1:
        st.header("Upload an Image for Emotion Analysis")
        uploaded_file = st.file_uploader("Choose an image with a face...", type=["jpg", "jpeg", "png"], help="Upload a clear image of a face for best results")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing emotions..."):
                emotion_probs, emotion_features = predict_emotion(image, vit_model, processor)
                if emotion_probs:
                    timestamp = pd.Timestamp.now()
                    history_entry = {'timestamp': timestamp}
                    history_entry.update(emotion_probs)
                    new_df = pd.DataFrame([history_entry])
                    if not new_df.empty:
                        st.session_state.history_df = pd.concat([st.session_state.history_df, new_df], ignore_index=True)
                    st.session_state.emotion_history.append(list(emotion_probs.values()))
                    
                    with col2:
                        html(display_current_emotion(emotion_probs), height=300, scrolling=True)
                        radar_fig = create_emotion_radar_chart(emotion_probs)
                        st.plotly_chart(radar_fig, use_container_width=True)
                    
                    st.markdown("### Detailed Emotion Analysis")
                    fig = visualize_emotions_plotly(emotion_probs)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if len(st.session_state.emotion_history) >= 3:
                        st.markdown("### 🔮 Future Emotion Prediction")
                        if len(st.session_state.emotion_history) >= 5:
                            with st.spinner("Adapting prediction model to your emotional patterns..."):
                                lstm_model, losses = train_lstm_with_sample(lstm_model, st.session_state.emotion_history)
                                if losses:
                                    st.success(f"Model adapted to your emotional patterns (Final loss: {losses[-1]:.4f})")
                        
                        lookback_window = min(5, len(st.session_state.emotion_history))
                        future_emotions = predict_future_emotions(st.session_state.emotion_history[-lookback_window:], lstm_model, num_predictions=5)
                        if future_emotions:
                            html(display_future_emotions_table(future_emotions), height=200, scrolling=True)
                            future_fig = visualize_future_emotions_plotly(future_emotions)
                            st.plotly_chart(future_fig, use_container_width=True)
                            final_state = future_emotions[-1]
                            dominant_idx = np.argmax(final_state)
                            dominant_final = CLASSES[dominant_idx]
                            interpretation_html = f"""
                            <div class="future-prediction">
                                <h4>Emotional Trajectory Interpretation</h4>
                                <p>Based on your current emotional state, you're likely to experience a shift toward 
                                <strong style="color: {EMOTION_COLORS[dominant_final]};">{dominant_final}</strong> 
                                in the near future.</p>
                            </div>
                            """
                            html(interpretation_html, height=100, scrolling=True)
                else:
                    st.error("Failed to analyze emotions. Please try with another image.")
        else:
            image_path = "Default_image.png"
            image = Image.open(image_path)
            st.markdown(
                """<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">""",
                unsafe_allow_html=True
            )
            st.image(image, width=250, use_container_width=False)


    
    
    with tab2:
        st.header("Real-time Emotion Analysis")
        st.write("Use your webcam to analyze emotions and predict emotional trends in real-time.")
        webcam_placeholder = st.empty()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            start_webcam = st.button("▶️ Start Camera", use_container_width=True)
        with col2:
            stop_webcam = st.button("⏹️ Stop Camera", use_container_width=True)
        with col3:
            clear_webcam_history = st.button("🧹 Clear History", use_container_width=True)
        
        emotion_display_container = st.container()
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False
        if start_webcam:
            st.session_state.webcam_running = True
        if stop_webcam:
            st.session_state.webcam_running = False
        if clear_webcam_history:
            if 'webcam_history_df' not in st.session_state:
                st.session_state.webcam_history_df = pd.DataFrame(columns=['timestamp'] + CLASSES)
            else:
                st.session_state.webcam_history_df = pd.DataFrame(columns=['timestamp'] + CLASSES)
            st.success("Webcam emotion history cleared!")
        if 'webcam_history_df' not in st.session_state:
            st.session_state.webcam_history_df = pd.DataFrame(columns=['timestamp'] + CLASSES)
        
        if st.session_state.webcam_running:
            try:
                video_capture = cv2.VideoCapture(0)
                if not video_capture.isOpened():
                    st.error("Error: Could not open webcam. Please check your camera permissions.")
                    st.session_state.webcam_running = False
                else:
                    st.markdown("""
                    <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 20px;">
                        📸 Camera is active! Press 'Stop Camera' when you're done.
                    </div>
                    """, unsafe_allow_html=True)
                    emotion_buffer = deque(maxlen=5)
                    webcam_col1, webcam_col2 = st.columns([1, 1])
                    with webcam_col1:
                        current_emotion_chart = st.empty()
                        webcam_emotion_history = st.empty()
                    with webcam_col2:
                        future_emotion_chart = st.empty()
                        radar_chart = st.empty()
                    
                    while st.session_state.webcam_running:
                        ret, frame = video_capture.read()
                        if not ret:
                            st.error("Failed to get frame from webcam.")
                            break
                        webcam_placeholder.image(frame, channels="BGR", caption="Webcam Feed", use_container_width=True)
                        if time.time() % 0.5 < 0.1:
                            pil_image = frame_to_pil(frame)
                            emotion_probs, emotion_features = predict_emotion(pil_image, vit_model, processor)
                            if emotion_probs:
                                emotion_buffer.append(emotion_probs)
                                if len(emotion_buffer) > 0:
                                    avg_probs = {key: sum(item.get(key, 0) for item in emotion_buffer) / len(emotion_buffer) for key in CLASSES}
                                    timestamp = pd.Timestamp.now()
                                    history_entry = {'timestamp': timestamp}
                                    history_entry.update(avg_probs)
                                    new_df = pd.DataFrame([history_entry])
                                    if not new_df.empty:
                                        st.session_state.history_df = pd.concat([st.session_state.history_df, new_df], ignore_index=True)
                                        st.session_state.webcam_history_df = pd.concat([st.session_state.webcam_history_df, new_df], ignore_index=True)
                                    st.session_state.emotion_history.append(list(avg_probs.values()))
                                    
                                    with webcam_col1:
                                        fig = visualize_emotions_plotly(avg_probs)
                                        current_emotion_chart.plotly_chart(fig, use_container_width=True)
                                        if len(st.session_state.webcam_history_df) > 1:
                                            recent_df = st.session_state.webcam_history_df.tail(10)
                                            history_fig = visualize_emotion_trends_plotly(recent_df)
                                            webcam_emotion_history.plotly_chart(history_fig, use_container_width=True)
                                    
                                    if len(st.session_state.emotion_history) >= 3:
                                        with webcam_col2:
                                            lookback = min(5, len(st.session_state.emotion_history))
                                            future_emotions = predict_future_emotions(st.session_state.emotion_history[-lookback:], lstm_model, num_predictions=3)
                                            if future_emotions:
                                                future_fig = visualize_future_emotions_plotly(future_emotions)
                                                future_emotion_chart.plotly_chart(future_fig, use_container_width=True)
                                                radar_fig = create_emotion_radar_chart(avg_probs)
                                                radar_chart.plotly_chart(radar_fig, use_container_width=True)
                        if not st.session_state.webcam_running:
                            break
                        time.sleep(0.03)
                    video_capture.release()
            except Exception as e:
                st.error(f"Webcam error: {e}")
                st.session_state.webcam_running = False
        else:
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
                <h3>Start the Camera</h3>
                <p>Press the 'Start Camera' button to begin real-time emotion analysis.</p>
                <p style="font-size: 0.9em; opacity: 0.7;">Make sure your face is clearly visible for best results.</p>
            </div>
            """, unsafe_allow_html=True)
   
# <img src="https://via.placeholder.com/150x150.png?text=📹" style="width: 100px; margin-bottom: 15px;">

    with tab3:
        st.header("Emotion Analytics Dashboard")
        if not st.session_state.history_df.empty and len(st.session_state.history_df) > 1:
            st.markdown("""
            <div style="background-color: #eef2ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="margin-top: 0;">Emotional Intelligence Dashboard</h4>
                <p>Analyze your emotional patterns and understand your emotional journey over time.</p>
            </div>
            """, unsafe_allow_html=True)
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                total_entries = len(st.session_state.history_df)
                st.metric("Total Analyses", total_entries)
            with metric_col2:
                dominant_emotions = [max({emotion: row[emotion] for emotion in CLASSES}.items(), key=lambda x: x[1])[0] for _, row in st.session_state.history_df.iterrows()]
                from collections import Counter
                most_common = Counter(dominant_emotions).most_common(1)[0][0]
                st.metric("Most Common Emotion", most_common)
            with metric_col3:
                time_span = st.session_state.history_df['timestamp'].max() - st.session_state.history_df['timestamp'].min()
                minutes = time_span.total_seconds() / 60
                time_display = f"{int(time_span.total_seconds())} seconds" if minutes < 1 else f"{int(minutes)} minutes"
                st.metric("Session Duration", time_display)
            with metric_col4:
                stability_scores = [max({emotion: row[emotion] for emotion in CLASSES}.values()) for _, row in st.session_state.history_df.iterrows()]
                avg_stability = np.mean(stability_scores)
                stability_percentage = int(avg_stability * 100)
                st.metric("Emotional Clarity", f"{stability_percentage}%")
            
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("### Emotional Journey Timeline")
                trend_fig = visualize_emotion_trends_plotly(st.session_state.history_df)
                st.plotly_chart(trend_fig, use_container_width=True)
                st.markdown("### Emotion Intensity Heatmap")
                heatmap_fig = create_emotion_heatmap(st.session_state.history_df)
                st.plotly_chart(heatmap_fig, use_container_width=True)
            with chart_col2:
                st.markdown("### Dominant Emotions Distribution")
                pie_fig = create_dominant_emotion_chart(st.session_state.history_df)
                st.plotly_chart(pie_fig, use_container_width=True)
                st.markdown("### Most Recent Emotional State")
                latest_data = st.session_state.history_df.iloc[-1]
                latest_emotions = {emotion: latest_data[emotion] for emotion in CLASSES}
                latest_radar = create_emotion_radar_chart(latest_emotions)
                st.plotly_chart(latest_radar, use_container_width=True)
            
            with st.expander("View Complete Emotion History Data"):
                display_df = st.session_state.history_df.copy()
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(display_df, use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    csv = display_df.to_csv(index=False)
                    st.download_button(label="Download as CSV", data=csv, file_name=f"emotion_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
                with col2:
                    if st.button("Clear All History Data"):
                        st.session_state.history_df = pd.DataFrame(columns=['timestamp'] + CLASSES)
                        st.session_state.emotion_history = []
                        st.session_state.webcam_history_df = pd.DataFrame(columns=['timestamp'] + CLASSES)
                        st.success("All history data cleared!")
                        st.experimental_rerun()
        else:
            st.info("Not enough emotion data available yet. Upload images or use the webcam to build your emotional profile.")
            if st.button("Generate Sample Data for Demonstration"):
                now = pd.Timestamp.now()
                timestamps = [now - pd.Timedelta(minutes=i*2) for i in range(30)]
                np.random.seed(42)
                sample_data = []
                base_emotions = np.random.dirichlet(np.ones(NUM_CLASSES), size=1)[0]
                for ts in timestamps:
                    noise = np.random.normal(0, 0.05, NUM_CLASSES)
                    probs = base_emotions + noise
                    probs = np.maximum(probs, 0)
                    probs = probs / np.sum(probs)
                    base_emotions = 0.8 * base_emotions + 0.2 * probs
                    entry = {'timestamp': ts}
                    for i, emotion in enumerate(CLASSES):
                        entry[emotion] = probs[i]
                    sample_data.append(entry)
                st.session_state.history_df = pd.DataFrame(sample_data)
                for entry in sample_data:
                    st.session_state.emotion_history.append([entry[emotion] for emotion in CLASSES])
                st.success("Sample data generated! Refresh to see the dashboard.")
                st.experimental_rerun()

if __name__ == "__main__":
    main()