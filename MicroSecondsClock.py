#clock displays time in microseconds

import sys
from tkinter import *
#import time
from datetime import datetime

def tick():
	#for microsecond use datetime
    time_string = datetime.now().strftime("%H:%M:%S.%f")
    clock.config(text=time_string)
    clock.after(200, tick)

root = Tk()
clock=Label(root, font=("times", 100, "bold"), bg= "white")
clock.grid(row=0, column=1)
tick()

root.mainloop()