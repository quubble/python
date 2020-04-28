#Function to get the exponential form of any number : num_to_the_power

def num_to_the_power(num1,num2):
    return num1 ** num2

n1 = float(input("Enter the first number:"))
n2 = float(input("Enter the second number:"))

result = num_to_the_power(n1,n2)
print("\nThe result-exponent is: " + str(result))