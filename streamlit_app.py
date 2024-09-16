import streamlit as st
import gdown
import os
import re
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from language_tool_python import LanguageTool

# Model file URL and character mappings URL from Google Drive
model_file_url = "https://drive.google.com/uc?id=1xtCyuXNKeyY_iRz0FVI_2Ov7q5YOPIS5"  # Model file link
char_mappings_url = "https://drive.google.com/uc?id=1YurcSu0Xnnr966aVMqhX_PpjpqU-QHJX"  # Replace with char_mappings.json file ID

# Local filenames
model_file = "best_model.keras"

# Only download if the model file doesn't exist
if not os.path.exists(model_file):
    with st.spinner('Downloading the model from Google Drive...'):
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
# Modify the generate_sonnet function
def generate_sonnet(seed_text, model, seq_length, vocab_size, char_to_index, index_to_char, temperature=1.0):
    generated_text = seed_text
    required_length = 14  # 14 lines in a sonnet
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

        if next_char == '\n':
            current_line += 1

        # Safeguard against infinite loops
        if len(generated_text) > 1000:
            break

        print(f"Predictions shape: {predictions.shape}")
        print(f"Next index: {next_index}")
        print(f"Generated character: {repr(next_char)}")

    return generated_text



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

def correct_grammar(text):
    tool = LanguageTool('en-US')
    corrected_text = tool.correct(text)
    return corrected_text
    
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
            temperature=1.0  # You can allow the user to change this if needed
        )
        
        # Format the generated sonnet
        formatted_sonnet = format_sonnet(generated_sonnet)
        
        # Correct grammar
        corrected_sonnet = correct_grammar(formatted_sonnet)
        
        st.markdown("**Generated Sonnet:**")
        st.markdown(formatted_sonnet)
        
        st.markdown("**Grammar-Corrected Sonnet:**")
        st.markdown(corrected_sonnet)
    
    # Append the generated and corrected text to the chat history
    st.session_state.chats.append({"role": "assistant", "content": f"**Generated Sonnet:**\n\n{formatted_sonnet}\n\n**Grammar-Corrected Sonnet:**\n\n{corrected_sonnet}"})

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
