from bs4 import BeautifulSoup
import requests
import pandas as pd

'''
The day of the week (Tuesday)
The date (01/06/2021)
A short description of the conditions (Clear early then increasing cloudiness after midnight. Low 41F. Winds light and variable)
The temperature low and high, with a function of your own to convert into Celsius
For each element you scrape, The name of the item you targetted (ex: DailyContent--daypartDate--3MM0J)
'''

page = requests.get('https://forecast.weather.gov/MapClick.php?x=276&y=148&site=lox&zmx=&zmy=&map_x=276&map_y=148#.YVLOoLgzaUk')
print(page)
soup = BeautifulSoup(page.content, 'html.parser')
div = soup.findAll('div', id="")
