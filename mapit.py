import sys, webbrowser, pyperclip

sys.argv  #[]

#check if cmd arguments were passed.

if len(sys.argv) > 1:
    address =' '.join(sys.argv)

else:
    address = pyperclip.paste()

#https://www.google.com/maps/@18.5996681,73.7527273,17z

webbrowser.open('https://www.google.com/maps/place/' +address)
