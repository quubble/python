#Run Successful

import speech_recognition as sr
import pyaudio
import os, sys

# Initialize the recognizer
recognizer = sr.Recognizer()

# Capture audio from the microphone
with sr.Microphone() as source:
    print("Speak something:")
    audio = recognizer.listen(source)

# Convert speech to text
try:
    text = recognizer.recognize_google(audio)
    print("You said: " + text)
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print("Error occurred; {0}".format(e))
    
#https://www.perplexity.ai/search/Creating-a-Speech-afBeEGNDSbS5zfpZUuQHeg