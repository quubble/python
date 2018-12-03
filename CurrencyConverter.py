import sys, os



print(" ")
print("Converting Pound into Euro")

print("\n") # This just to print empty place for neat looking

pound = float(input("Please enter amount of money in GBP you would like to convert into Euro €: "))
print("\n")

exchangeRate = float(input("Please enter the current Pound-Euro exchange rate: "))
print("\n")
 
euro = round((pound/exchangeRate),2) #This for caluclation.

print("You will receive €{0} Euros for your holiday.".format(euro))
