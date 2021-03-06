"""
Simple Program to help you get started with Google's APIs
"""
import urllib.request, json
#Google MapsDdirections API endpoint
endpoint = 'https://maps.googleapis.com/maps/api/directions/json?'
api_key = 'AIzaSyBzpAlmXrhziNKtqq3gn237e4zPP5Zw_sM'
#Asks the user to input Where they are and where they want to go.
origin = input('Where are you?: ').replace(' ','+')
print()
destination = input('Where do you want to go?: ').replace(' ','+')
print()
#Building the URL for the request
nav_request = 'origin={}&destination={}&key={}'.format(origin,destination,api_key)
request = endpoint + nav_request
#Sends the request and reads the response.
response = urllib.request.urlopen(request).read()
#Loads response as JSON
directions = json.loads(response)
print()
print(directions)


