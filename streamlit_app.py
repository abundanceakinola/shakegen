import streamlit as st
import gdown
import os
import re
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the latest model version from Google Drive
version_file_url = "https://drive.google.com/uc?id=1zaNKBM8-RAfrQF9FfkcvoKBIfM6fcfou"  # Replace with your version.txt file ID

# Download the version file to check the latest model
version_file = "version.txt"
gdown.download(version_file_url, version_file, quiet=False)

# Read the latest model version from version.txt
with open(version_file, 'r') as f:
    latest_model_filename = f.read().strip()

# Construct the URL to the model file
model_file_url = f"https://drive.google.com/uc?id=1bv1XhAQY_-73DKalRLpDhukGf9h8QAok"  # Replace with actual Google Drive model file ID for the version

# Local filenames
model_file = latest_model_filename

# Only download if the model file doesn't exist
if not os.path.exists(model_file):
    with st.spinner('Downloading the latest model from Google Drive...'):
        gdown.download(model_file_url, model_file, quiet=False)

# Load the model
model = load_model(model_file)

char_mappings_url = f"https://drive.google.com/uc?id=1YurcSu0Xnnr966aVMqhX_PpjpqU-QHJX"  # Replace with actual Google Drive file ID for the version
char_mappings_file = "char_mappings.json"

# Only download the character mappings if it doesn't exist locally
if not os.path.exists(char_mappings_file):
    with st.spinner('Downloading character mappings from Google Drive...'):
        gdown.download(char_mappings_url, char_mappings_file, quiet=False)


# Load the character mappings from the JSON file
with open(char_mappings_file, 'r') as f:
    char_mappings = json.load(f)
    char_to_index = char_mappings["char_to_index"]
    index_to_char = char_mappings["index_to_char"]

vocab_size = len(char_to_index)  # Ensure this matches your model's vocabulary size
seq_length = 50  # Define your sequence length

# Function to generate text using the model
# Function to generate sonnet
import random

def generate_sonnet_with_structure(seed_text, model, seq_length, vocab_size, char_to_index, index_to_char, temperature=1.0):
    generated_text = seed_text
    current_line = 0
    required_lines = 14  # 14 lines in a sonnet (12 quatrain lines, 2 couplet lines)
    max_line_length = 1000  # Safeguard for infinite loops
    line_word_count = 0  # Track words per line
    words_in_line = random.randint(8, 10)  # Randomize words per line

    while current_line < required_lines:
        x = np.zeros((1, seq_length, vocab_size))
        for t, char in enumerate(generated_text[-seq_length:]):
            if char in char_to_index:
                x[0, t, char_to_index[char]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)

        # Ensure the next index is valid
        if str(next_index) in index_to_char:
            next_char = index_to_char[str(next_index)]
        else:
            print(f"Warning: Invalid index {next_index}. Using fallback character.")
            next_char = ' '  # Fallback to space if index is out of range

        generated_text += next_char

        # Track words by spaces (' ') and handle line breaks when word count is reached
        if next_char == ' ':
            line_word_count += 1

        if line_word_count >= words_in_line:
            generated_text += '\n'
            current_line += 1
            line_word_count = 0  # Reset word count for next line
            words_in_line = random.randint(8, 10)  # Randomize words for the next line

            # Add indentation to couplet (lines 13 and 14)
            if current_line == 12:
                generated_text += "    <COUPLET_START>\n"
            elif current_line == 14:
                generated_text += "    <COUPLET_END>\n<SONNET_END>"
                break

        # Safeguard against infinite loops
        if len(generated_text) > max_line_length:
            break

    return generated_text


# Function to post-process the sonnet (like your Colab code)
def post_process_sonnet(sonnet):
    # Remove tags
    sonnet = sonnet.replace("<SONNET_START>\n", "")
    sonnet = sonnet.replace("<SONNET_END>", "")
    sonnet = sonnet.replace("<LINE>", "")

    # Split the sonnet into lines
    lines = sonnet.split('\n')

    # Process each line
    processed_lines = []
    for i, line in enumerate(lines):
        # Remove any remaining tags
        line = re.sub(r'<.*?>', '', line).strip()

        # Indent the couplet (last two lines)
        if i >= len(lines) - 2:
            line = "    " + line

        processed_lines.append(line)

    # Join the lines back together
    processed_sonnet = '\n'.join(processed_lines).strip()

    return processed_sonnet

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature  # Avoid log(0) errors
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def format_sonnet(text):
    # If no <LINE> tags, treat the entire text as a single block
    if '<LINE>' not in text:
        # Format the text without assuming <LINE> tags
        formatted_text = text.strip()
    else:
        # Split the text by <LINE> tags if they exist
        lines = re.split(r'<LINE>', text)
        # Remove empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]
        # Join the lines with newline characters
        formatted_text = '\n'.join(lines)
    
    # Remove any remaining tags
    formatted_text = re.sub(r'<[^>]+>', '', formatted_text)
    return formatted_text

def simple_text_cleanup(text):
    # Capitalize the first letter of each line
    lines = text.split('\n')
    cleaned_lines = [line.capitalize() for line in lines]
    
    # Join the lines back together
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Remove any non-alphabetic characters except spaces, periods, and commas within the words
    cleaned_text = re.sub(r'[^a-zA-Z\s.,]', '', cleaned_text)  # Keeps letters, spaces, commas, and periods
    
    # Ensure there's a space after each comma and period if it's missing
    cleaned_text = re.sub(r',(\S)', r', \1', cleaned_text)
    cleaned_text = re.sub(r'\.(\S)', r'. \1', cleaned_text)
    
    return cleaned_text


    
# Streamlit UI
st.title('ShakeGen')

st.info('AI-powered Shakespearean Sonnet Generator')

# Temperature slider for users to choose the temperature level
temperature = st.slider('Select Temperature', 0.1, 2.0, value=0.5)

# Initialize chat history
if "chats" not in st.session_state:
    st.session_state.chats = []

# Display chat messages from history on app rerun
for chat in st.session_state.chats:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Get user input as the seed text for the model
if prompt := st.chat_input("Enter the first line of the sonnet:"):
    st.session_state.chats.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate sonnet with the model
    with st.chat_message("assistant"):
        generated_sonnet = generate_sonnet_with_structure(
            seed_text=prompt,
            model=model,
            seq_length=seq_length,
            vocab_size=vocab_size,
            char_to_index=char_to_index,
            index_to_char=index_to_char,
            temperature=0.5  # You can allow the user to change this if needed
        )
        
        # Format the generated sonnet
        formatted_sonnet = format_sonnet(generated_sonnet)
        
        # Clean up the text
        cleaned_sonnet = simple_text_cleanup(formatted_sonnet)
        
        st.markdown("**Generated Sonnet:**")
        st.markdown(formatted_sonnet)
        
        # Add an expander to hide or show the cleaned-up sonnet
        with st.expander("Show Cleaned-up Sonnet"):
            st.markdown(cleaned_sonnet)
    
    # Append the generated and cleaned-up text to the chat history
    st.session_state.chats.append({"role": "assistant", "content": f"**Generated Sonnet:**\n\n{formatted_sonnet}\n\n**Cleaned-up Sonnet:**\n\n{cleaned_sonnet}"})

# Sidebar content
st.sidebar.header("Model and Generation Information")

# Show model details
st.sidebar.subheader("Model Details")
st.sidebar.write(f"Vocabulary size: {vocab_size}")
st.sidebar.write(f"Sequence length: {seq_length}")
# Add more model details if available (e.g., number of layers, model size, etc.)

# Add a guide on temperature
st.sidebar.subheader("Temperature Guide")
st.sidebar.write("""
- **Low (0.1 - 0.5):** Generates more predictable, structured text.
- **Medium (0.5 - 1.0):** Balanced creativity and coherence.
- **High (1.0 - 2.0):** More creative but potentially less coherent.
""")

# Display user input seed text
st.sidebar.subheader("Seed Text")
if prompt:
    st.sidebar.write(f"'{prompt}'")

