import streamlit as st
import gdown
import os

# Function to download file from Google Drive
def download_file_from_google_drive(file_id, output_file):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)

# Download model file from Google Drive (dummy file fetch, no model loading)
model_file_id = "1-2_TX-rMJYxhvAFRMYuGG38uvVksUd-7"  # Replace with your actual file ID
model_file = "best_simple_LSTM.keras"

if not os.path.exists(model_file):
    with st.spinner('Downloading the model from Google Drive...'):
        download_file_from_google_drive(model_file_id, model_file)

# Download Shakespeare text from Google Drive (dummy file fetch)
shakespeare_file_id = "1DIMeFhb40tE03Lay2gOXN40ytz1f3ptP"  # Replace with your actual file ID
shakespeare_file = "shakespeare.txt"

if not os.path.exists(shakespeare_file):
    with st.spinner('Downloading Shakespeare text from Google Drive...'):
        download_file_from_google_drive(shakespeare_file_id, shakespeare_file)

# Dummy chat response generator
def dummy_generate_text(seed_text):
    # Simply repeat the seed text with a "robotic" gibberish
    return f"{seed_text}... blip bloop... I am a robot generating gibberish... blip blop."

# Create a list to store past chats
if 'past_chats' not in st.session_state:
    st.session_state.past_chats = []

# Streamlit UI
st.title('ShakeGen: Simple LSTM Edition (Demo)')

st.info('Dummy chat interface for ShakeGen LSTM model (just a demo)')

# Sidebar content remains unchanged
st.sidebar.header("ShakeGen Information")
st.sidebar.write("""
**ShakeGen** is an AI-powered text generator designed to create poetry in the style of Shakespeare using an LSTM model. 
You can experiment with **seed text** and adjust the **temperature** slider for creative variability.
- **Temperature**: Controls randomness. Lower values (e.g. 0.5) generate more predictable text, while higher values (e.g. 1.5) increase creativity.
- **Example Seed Text**: "Shall I compare thee"
""")

# Chat History Display
st.subheader("Chat History")
chat_history = "\n\n".join([f"**User:** {chat['seed']}\n\n**ShakeGen:** {chat['generated_text']}\n---" for chat in st.session_state.past_chats])
st.markdown(f"""
<div style="height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;">
{chat_history}
</div>
""", unsafe_allow_html=True)

# Input field and button for the chat interface
seed_text = st.text_input("Enter seed text for generation", placeholder="Type your seed text here...")
generate_button = st.button("Generate")

# Generate dummy response if button is clicked
if generate_button and seed_text:
    generated_text = dummy_generate_text(seed_text)
    st.session_state.past_chats.append({
        'seed': seed_text,
        'generated_text': generated_text
    })
    st.experimental_rerun()  # Reload to display new chat
