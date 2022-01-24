import requests
from bs4 import BeautifulSoup

url = 'https://coinmarketcap.com/es/currencies/bitcoin/'

req = requests.get(url)
soup = BeautifulSoup(req.text, 'html.parser')

btc_price = soup.find('div', class_="priceValue").find('span').text

print(f'BTC price: {btc_price}')