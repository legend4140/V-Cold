import speech_recognition as sr
import nltk
from flask import Flask, render_template, request
import pyttsx3

app = Flask(__name__)
recognizer = sr.Recognizer()

nltk.download('punkt')

def tokenize_text(text):
    return nltk.word_tokenize(text)

def check_for_cold(symptoms, audio_text_tokens):
    for symptom in symptoms:
        if symptom in audio_text_tokens:
            return True
    return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect_cold', methods=['POST'])
def detect_cold():
    audio_text_en = ""

    try:
        if 'audio' in request.files:
            audio_file = request.files['audio']
            audio = sr.AudioFile(audio_file)
            with audio as source:
                audio_data = recognizer.record(source)
                audio_text_en = recognizer.recognize_google(audio_data, language="en-US")
    except sr.UnknownValueError:
        pass

    audio_text_tokens_en = tokenize_text(audio_text_en.lower())
    cold_symptoms = ['sneezing', 'coughing', 'congestion', 'runny nose', 'sore throat']

    if check_for_cold(cold_symptoms, audio_text_tokens_en):
        result = "You may have cold symptoms."
    else:
        result = "You do not seem to have cold symptoms."

    return result

if __name__ == '__main__':
    app.run()