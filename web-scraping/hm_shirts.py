import requests
from bs4 import BeautifulSoup
import urllib.request as urllib2
import os

# URL for hm.com with all the shirts in product form (no model)
url = "http://www2.hm.com/en_in/men/shop-by-product/shirts.html?product-type=men_shirts&image=stillLife&sort=stock&offset=0&page-size=219"

# Getting the webpage from the url
html = requests.get(url)

# Using BeautifulSoup to parse the html of the page into a soup object
soup = BeautifulSoup(html.content,"html.parser")

# Finding all the hyperlinks which have images attributed to them
# they all have a common identifier in class which is always "product-item-link"
links = soup.findAll("a", {"class" : "product-item-link"})

print(len(links))

count = 0

# Looping over all the links to get images
for link in links:
    print(count)
    # finding the img tag in each hyperlink
    img = link.find("img")
    # getting the source of the image
    imgsrc = img.get("src")

    # using the urlretrieve method to download the image from the internet
    urllib2.urlretrieve("https:" + imgsrc, "drive/ZML/FashioNet/train-images/shirt" + str(count) + ".jpg")
    count += 1
