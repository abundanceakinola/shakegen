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
def generate_sonnet_with_structure(seed_text, model, seq_length, vocab_size, char_to_index, index_to_char, temperature=1.0):
    generated_text = seed_text
    required_length = 14  # 14 lines in a sonnet (12 lines of quatrains, 2 lines of couplet)
    current_line = 0

    while current_line < required_length:
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

        # Add structure for couplets
        if next_char == '\n':
            current_line += 1
            if current_line == 12:
                generated_text += "<COUPLET_START>\n"
            elif current_line == 14:
                generated_text += "<COUPLET_END>\n<SONNET_END>"
                break

        # Safeguard against infinite loops
        if len(generated_text) > 1000:
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

    
# Streamlit UI
st.title('ShakeGen')

st.info('AI-powered Shakespearean Sonnet Generator')

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
        generated_sonnet = generate_sonnet(
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
        
        st.markdown("**Cleaned-up Sonnet:**")
        st.markdown(cleaned_sonnet)
    
    # Append the generated and cleaned-up text to the chat history
    st.session_state.chats.append({"role": "assistant", "content": f"**Generated Sonnet:**\n\n{formatted_sonnet}\n\n**Cleaned-up Sonnet:**\n\n{cleaned_sonnet}"})

# Add this debugging information
st.sidebar.write("Debugging Information:")
st.sidebar.write(f"Vocabulary size: {vocab_size}")
st.sidebar.write(f"Number of characters in mapping: {len(index_to_char)}")
st.sidebar.write("First 10 character mappings:")
for i in range(10):
    if str(i) in index_to_char:
        st.sidebar.write(f"{i}: {repr(index_to_char[str(i)])}")
    else:
        st.sidebar.write(f"{i}: Not found in mapping")
