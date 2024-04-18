# Import the required library
import pyttsx3

# Define a function to speak text
def speak_text(text):
    # Create an instance of the Text-to-Speech engine
    engine = pyttsx3.init()
    
    # Set the voice of the Text-to-Speech engine
    engine.setProperty("voice", engine.getProperty("voices")[0].id)
    
    # Speak the text
    engine.say(text)
    engine.runAndWait()
    
    # Mute the speech output to simulate stopping speech
    #engine.setProperty('volume', 0.0)

# Get input from the user about what to speak
input_text = input("Enter the text to speak: ")

# Call the speak_text function to speak the input text
speak_text(input_text)