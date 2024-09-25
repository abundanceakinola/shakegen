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
    return f"{seed_text}\nAnd to trumm thy hays bry making mankso’r,\nOn she wailot my slope eass of love, what watuay sooW;\ncannate\nAs many goes fur maduth manck ante his trile\nAnd of thou and muchy verang the readd,\nSpetterest of penous day so thee thee rece."


# Initialize chat history if not already done
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "user", "content": "User's seed text goes here"},
                                 {"role": "assistant", "content": "Generated text with seed text goes here"}]

# Streamlit UI title
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

    # Generate the robot response with some gibberish
    response = dummy_generate_text(prompt)
    
    # Display robot response
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add robot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
