import bs4, requests

def AmazonPriceInfo(ProductUrl):
    res = requests.get(ProductUrl) # create HTTP response object to download page
    res.raise_for_status()

    soup = bs4.BeautifulSoup(res.text, 'html.parser')
    element = soup.select('#comic > img')
    return element[0].text.strip()
    






price = AmazonPriceInfo('https://xkcd.com/')
print('The Price is ' + price)
