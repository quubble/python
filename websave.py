#from urllib.request import urlopen
#html = urlopen("https://github.com/rg3/youtube-dl/blob/master/README.md").read().decode('utf-8')
#print(html)



#os.chdir("G:\Chanakya-TV-Series\YouTube-Windows-Powershell-Tutorials")


import requests, os, sys

os.chdir("G:\Chanakya-TV-Series\YouTube-Windows-Powershell-Tutorials")

url = 'https://github.com/rg3/youtube-dl/blob/master/README.md'
r = requests.get(url, allow_redirects=True)
open('Youtube-dl-readme.html', 'wb').write(r.content)



