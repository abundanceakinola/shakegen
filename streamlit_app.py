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
    return f"{seed_text}... fair winds blow and the earth's glow... bleep bloop... thou art as brilliant as the sun!"

# Create a list to store past chats
if 'past_chats' not in st.session_state:
    st.session_state.past_chats = [{"seed": "Shall I compare thee to a summer's day", 
                                    "generated_text": "Shall I compare thee to a summer's day... bleep bloop... thou art as bright as the sun!"}]

# Streamlit UI
st.title('ShakeGen: Simple LSTM Edition')

st.info('AI-powered Shakespearean Sonnet Generator using Simple LSTM')

# Sidebar content remains unchanged
st.sidebar.header("ShakeGen Information")
st.sidebar.write("""
**ShakeGen** is an AI-powered text generator designed to create poetry in the style of Shakespeare using an LSTM model. 
You can experiment with **seed text** and adjust the **temperature** slider for creative variability.
- **Temperature**: Controls randomness. Lower values (e.g. 0.5) generate more predictable text, while higher values (e.g. 1.5) increase creativity.
- **Example Seed Text**: "Shall I compare thee"
""")

# Add temperature slider (not connected to model yet)
temperature = st.sidebar.slider('Select Temperature', 0.1, 2.0, value=0.5)

# Modernized Chat History Display
st.subheader("Chat History")
chat_history = ""
for chat in st.session_state.past_chats:
    # User's message (grey background)
    chat_history += f"""
    <div style="background-color: #e0e0e0; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
        <strong>User:</strong> {chat['seed']}
    </div>
    """
    # Robot's response (white background)
    chat_history += f"""
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
        <strong>ShakeGen:</strong> {chat['generated_text']}
    </div>
    """
st.markdown(f"""
<div style="height: 400px; overflow-y: scroll; padding: 10px;">
    {chat_history}
</div>
""", unsafe_allow_html=True)

# Text input and button at the bottom of the page
st.markdown("""
<div style="position: fixed; bottom: 0; width: 100%; background-color: white; padding: 10px; box-shadow: 0px -2px 10px rgba(0, 0, 0, 0.1);">
    <input type="text" id="seed_input" placeholder="Enter seed text for generation" style="width: 70%; padding: 10px; border-radius: 5px; border: 1px solid #ccc;" />
    <button id="generate_button" style="width: 20%; padding: 10px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;">Generate</button>
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

# Generate dummy response if seed text is provided
if seed_text:
    generated_text = dummy_generate_text(seed_text)
    st.session_state.past_chats.append({
        'seed': seed_text,
        'generated_text': generated_text
    })
    st.experimental_rerun()  # Reload to display new chat
