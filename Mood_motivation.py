import streamlit as st
import random
from groq import Groq
import re
from gtts import gTTS
import base64
from transformers import pipeline
import os
import tempfile
import warnings
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(page_title="Mood-to-Motivation Bot", layout="wide")

# Load environment variables from .env file (for local development)
load_dotenv()

# --- Configure API ---
# Handle Streamlit Cloud Secrets and Local Environment appropriately
try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError, Exception):
    API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Please check your .env file or Streamlit Cloud Secrets.")
    st.stop()

client = Groq(api_key=API_KEY)

# Load emotion detection pipeline with caching to prevent memory issues on reload
@st.cache_resource
def load_emotion_classifier():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", framework="pt")

emotion_classifier = load_emotion_classifier()

# --- Helper Functions ---
def generate_dynamic_response(user_message, mood_name=None, language='en'):
    """Use Groq to generate dynamic conversational responses"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"If user have any choice like motivation, joke, suggestion give him the reply based on {user_message}. "
                        f"Communicate with the user based on the {mood_name}. If user is negative or neutral make them happy. "
                        "If user not intialized the mood ask user to share their feelings. "
                        f"Communicate humble and smooth with the user {user_message}. Reply in 1-2 sentences only."
                    )
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Sorry, I couldn't generate a response: {str(e)}"

def text_to_audio(text, lang="en"):
    """Convert text to audio and return Streamlit audio component"""
    try:
        # Use a temporary file to prevent accumulating output.mp3 on cloud instances
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', mode='wb') as tmp_file:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        st.error(f"🎵 Audio generation failed: {str(e)}")
        return None

def cleanup_audio_files():
    """Clean up temporary audio files"""
    import glob
    for f in glob.glob('*.mp3'):
        try:
            os.remove(f)
        except:
            pass

# --- Initialize session_state ---
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "mood_detected" not in st.session_state:
    st.session_state.mood_detected = None

# Cleanup old audio files
cleanup_audio_files()

st.title("🌈 Mood-to-Motivation Bot 🎭💡")

# --- Layout ---
col1, col2 = st.columns([1, 2])  # Left narrow, right wide

with col1:
    st.subheader("🔍 Mood Input")
    communication_mode = st.radio("Choose Communication Mode:", ["Text", "Audio"], key="comm_mode")
    
    # We use a form to prevent Streamlit from creating infinite loops when modifying text + radio choices
    with st.form("input_form"):
        user_input = st.text_input("Hey! There", key="input")
        st.subheader("✨ Quick Options")
        choice = st.radio("Choose one:", ["Text", "Motivational Quote", "Suggestion", "Change Action", "Joke"])
        submitted = st.form_submit_button("Send")
    
    if submitted and user_input:
        with st.spinner("Analyzing your mood..."):
            result = emotion_classifier(user_input)[0]
            mood = result['label']
            st.session_state.mood_detected = mood
            
            # Adjust input if the user has a specific choice
            formatted_input = user_input
            if choice != "Text":
                formatted_input = "User asks for " + choice + " based on " + user_input
            
            st.session_state.conversation.append(("User", formatted_input))
            response = generate_dynamic_response(formatted_input, mood)
            st.session_state.conversation.append(("Bot", response))
            st.rerun()


with col2:
    st.subheader("💬 Conversation")
    st.write("Hello there! I am here to help you with your emotions")
    
    for sender, message in st.session_state.conversation:
        if sender == "User":
            st.markdown(f"**You:** {message}")
        else:
            if communication_mode == "Audio":
                st.markdown(f"**Bot:** {message}") # It's good practice to print text even in audio mode!
                audio_path = text_to_audio(message)
                if audio_path:
                    st.audio(audio_path, format="audio/mp3")
            else:
                st.markdown(f"**Bot:** {message}")
