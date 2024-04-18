import requests
import json

def get_weather_data(city, api_key):
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    response = requests.get(url)
    data = json.loads(response.text)
    return data

def format_output(data):
    temp_c = data['current']['temp_c']
    condition = data['current']['condition']['text']
    wind_mph = data['current']['wind_mph']
    return f"The current temperature in {data['location']['name']} is {temp_c}°C with {condition} and wind speed of {wind_mph} mph."

api_key = "74192b7449ab481594f95147241604"
city = input("Enter the name of the city: ")
data = get_weather_data(city, api_key)
output = format_output(data)
print(output)