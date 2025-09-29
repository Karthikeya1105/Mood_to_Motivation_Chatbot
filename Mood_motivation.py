import streamlit as st
import random
import google.generativeai as genai
import re
from gtts import gTTS
import base64
from transformers import pipeline

# --- Configure API ---
genai.configure(api_key="AIzaSyAdI_CE2EXs49yUziAF7n-oIgIW6pNBEe0")

# Load emotion detection pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# --- Helper Functions ---
def detect_mood(user_input: str):
    user_input = user_input.lower()
    for intent, data in MOODS.items():
        if intent in user_input:
            return intent
        for word in data['suggestions'] + data["expressions"]:
            if word.lower() in user_input:
                return intent
    return None

def generate_dynamic_response(user_message, mood_name=None, language='en',):
    """Use Gemini to generate dynamic conversational responses"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            [
                f"If user have any choice like motivation, joke, suggestion give him the reply based on {user_message}."
                f"Communicate with the user based on the {mood_name}. If user is negative or neutral make them happy. "
                "If user not intialized the mood ask user to share their feelings."
                f"Communicate humble and smooth with the user {user_message}. Reply in 1-2 sentences only.",
                user_message,
            ]
        )
        return response.text
    except Exception as e:
        return f"Sorry, I couldn't generate a response: {str(e)}"

def text_to_audio(text, lang="en"):
    """Convert text to audio and return Streamlit audio component"""
    tts = gTTS(text=text, lang=lang)
    audio_file = "output.mp3"
    tts.save(audio_file)
    return audio_file

# --- Initialize session_state ---
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "mood_detected" not in st.session_state:
    st.session_state.mood_detected = None

# --- Page Config ---
st.set_page_config(page_title="Mood-to-Motivation Bot", layout="wide")

st.title("🌈 Mood-to-Motivation Bot 🎭💡")
# --- Layout ---
col1, col2 = st.columns([1, 2])  # Left narrow, right wide
communication_mode="Text"

with col1:
    st.subheader("🔍 Mood Input")
    communication_mode = st.radio("Choose Communication Mode:", ["Text", "Audio"], key="comm_mode")
    user_input = st.text_input("Hey! There", key="input")
    
    if user_input:
        result = emotion_classifier(user_input)[0]
        mood = result['label']
        st.session_state.mood_detected = mood
        st.subheader("✨ Quick Options")
        choice = st.radio("Choose one:", ["Text","Motivational Quote", "Suggestion", "Change Action", "Joke"])
        if choice!="Text":
            user_input="User asks for "+choice+" based on "+user_input
        st.session_state.conversation.append(("User", user_input))
        response = generate_dynamic_response(user_input, mood)
        st.session_state.conversation.append(("Bot", response))

with col2:
    st.subheader("💬 Conversation")
    st.subheader("Hello there! I am here to help you with your emotions")
    for sender, message in st.session_state.conversation:
        if sender == "User":
            st.markdown(f"*You:* {message}")
        else:
            if communication_mode == "Audio":
                audio_path = text_to_audio(message)
                st.audio(audio_path, format="audio/mp3")
            else :
                st.markdown(f"*Bot:* {message}")
