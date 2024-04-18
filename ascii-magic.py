
#program to generate Ascci Art
from ascii_magic import AsciiArt #,Back
#from PIL import ImageEnhance


my_art = AsciiArt.from_image("nf/moon.jpg")
#my_art.image = ImageEnhance.Brightness(my_art.image).enhance(0.2)
#my_art.to_terminal(columns=200, back=Back.BLUE)
my_art.to_terminal()

#image required