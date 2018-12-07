#Get Latitude & Longitude with python
import requests


response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address=1600+Amphitheatre+Parkway,+Mountain+View,+CA&key=AIzaSyBzpAlmXrhziNKtqq3gn237e4zPP5Zw_sM')

resp_json_payload = response.json()

print(resp_json_payload['results'][0]['geometry']['location'])