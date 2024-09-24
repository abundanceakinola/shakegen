import streamlit as st
import gdown
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import random

# Function to download file from Google Drive
def download_file_from_google_drive(file_id, output_file):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)

# Download and load the model
model_file_id = "1-2_TX-rMJYxhvAFRMYuGG38uvVksUd-7"  # Replace with your Google Drive file ID
model_file = "best_simple_LSTM.keras"

if not os.path.exists(model_file):
    with st.spinner('Downloading the model from Google Drive...'):
        download_file_from_google_drive(model_file_id, model_file)

model = load_model(model_file)

# Download Shakespeare text file
shakespeare_file_id = "1DIMeFhb40tE03Lay2gOXN40ytz1f3ptP"  # Replace with your Google Drive file ID
shakespeare_file = "shakespeare.txt"

if not os.path.exists(shakespeare_file):
    with st.spinner('Downloading Shakespeare text from Google Drive...'):
        download_file_from_google_drive(shakespeare_file_id, shakespeare_file)

# Load character mappings
# Read and process the text
with open(shakespeare_file, 'r', encoding='utf-8') as file:
    text = file.read()

# Only use the first 97,000 characters
text = text[1:97000]  # Cutting down to 97,000 characters

# Clean the text to remove unwanted characters like 'æ'
unwanted_chars = 'æ'  # You can add more characters to this string if needed
text = ''.join([char for char in text if char not in unwanted_chars])

characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

SEQ_LENGTH = 40  # This should match your model's input sequence length

# Define the sample function
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Define the generate_text function
def generate_text(seed_text, length, temperature):
    generated = seed_text
    sentence = seed_text[-SEQ_LENGTH:]
    
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            if char in char_to_index:
                x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character  # Shift the sentence window
    
    return generated

# Create a list to store past chats
if 'past_chats' not in st.session_state:
    st.session_state.past_chats = []

# Streamlit UI
st.title('ShakeGen: Simple LSTM Edition')

st.info('AI-powered Shakespearean Sonnet Generator using Simple LSTM')

# Sidebar content
st.sidebar.header("ShakeGen Information")
st.sidebar.write("""
**ShakeGen** is an AI-powered text generator designed to create poetry in the style of Shakespeare using an LSTM model. 
You can experiment with **seed text** and adjust the **temperature** slider for creative variability.
- **Temperature**: Controls randomness. Lower values (e.g. 0.5) generate more predictable text, while higher values (e.g. 1.5) increase creativity.
- **Example Seed Text**: "Shall I compare thee"
""")

# Temperature slider in the sidebar
temperature = st.sidebar.slider('Select Temperature', 0.1, 2.0, value=0.5)

# Display past chats (if any)
st.subheader("Chat History")
if st.session_state.past_chats:
    for chat in st.session_state.past_chats:
        st.markdown(f"**User:** {chat['seed']}\n\n**ShakeGen:** {chat['generated_text']}\n---")

# Add a div around the input and button for layout control
st.markdown("""
<div style="display: flex; justify-content: space-between; position: fixed; bottom: 0; width: 100%; background-color: #fff; padding: 10px;">
    <input type="text" id="seed_input" style="width: 80%; padding: 10px;" placeholder="Enter seed text for generation" />
    <button id="generate_button" style="width: 18%; padding: 10px;">Generate</button>
</div>
<script>
    const generateButton = document.getElementById("generate_button");
    generateButton.addEventListener("click", function() {
        const seedText = document.getElementById("seed_input").value;
        window.location.href = "/?seed_text=" + seedText;
    });
</script>
""", unsafe_allow_html=True)

# Extract the seed text from URL parameters
seed_text = st.experimental_get_query_params().get("seed_text", [""])[0]

# Generate button functionality
if seed_text:
    if len(seed_text) < SEQ_LENGTH:
        st.warning(f"Please provide at least {SEQ_LENGTH} characters for the seed text.")
    else:
        generated_text = generate_text(seed_text, 600, temperature)  # Generate 600 characters of text
        if generated_text:
            st.session_state.past_chats.append({
                'seed': seed_text,
                'generated_text': generated_text
            })
            st.experimental_rerun()
