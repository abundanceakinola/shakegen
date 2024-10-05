import streamlit as st
import gdown
import os
import tensorflow as tf
import numpy as np

# Function to download file from Google Drive
def download_file_from_google_drive(file_id, output_file):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)

# Download model file from Google Drive
model_file_id = "1lRbDGMGP5ETCtfToZ_Ea9-xqtTt1mbX2"
model_file = "best_simple_LSTM.keras"
if not os.path.exists(model_file):
    with st.spinner('Downloading the model from Google Drive...'):
        download_file_from_google_drive(model_file_id, model_file)

# Download Shakespeare text from Google Drive
shakespeare_file_id = "1DIMeFhb40tE03Lay2gOXN40ytz1f3ptP"
shakespeare_file = "shakespeare.txt"
if not os.path.exists(shakespeare_file):
    with st.spinner('Downloading Shakespeare text from Google Drive...'):
        download_file_from_google_drive(shakespeare_file_id, shakespeare_file)

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data(file_path, max_length=97000):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()[:max_length]
    
    unique_chars = sorted(set(raw_text))
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
    
    return raw_text, char_to_idx, idx_to_char

# Load the model
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Text generation function
def generate_text(model, start_text, char_to_idx, idx_to_char, length=300, temperature=0.5):
    generated_text = start_text

    for _ in range(length):
        x_pred = np.zeros((1, len(start_text), len(char_to_idx)))
        for t, char in enumerate(start_text):
            x_pred[0, t, char_to_idx[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_with_temperature(preds, temperature)
        next_char = idx_to_char[next_index]

        generated_text += next_char
        start_text = start_text[1:] + next_char

    return generated_text

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Load data and model
raw_text, char_to_idx, idx_to_char = load_and_preprocess_data(shakespeare_file)
model = load_model(model_file)

# Streamlit UI title
st.title('ShakeGen: AI Sonnet Generator')
st.info('AI-powered Shakespearean Sonnet Generator using Simple LSTM')

# Sidebar content
st.sidebar.header("ShakeGen Information")
st.sidebar.write("""
**ShakeGen** is an AI-powered text generator designed to create poetry in the style of Shakespeare using an LSTM model. 
You can experiment with **seed text** and adjust the **temperature** slider for creative variability.
- **Temperature**: Controls randomness. Lower values (e.g. 0.5) generate more predictable text, while higher values (e.g. 1.5) increase creativity.
- **Example Seed Text**: "Shall I compare thee"
""")

# Add temperature slider
temperature = st.sidebar.slider('Select Temperature', 0.1, 2.0, value=0.5)

# Initialize chat history if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter a seed text"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate the response using the LSTM model
    response = generate_text(model, prompt, char_to_idx, idx_to_char, length=300, temperature=temperature)
    
    # Display robot response
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add robot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
