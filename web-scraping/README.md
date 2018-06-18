# Web Scraping
This folder contains two codes written in python that help extract images of shirts from the web.\

## Modules used
- BeautifulSoup
- urllib.request
- requests

## Overview
1. The two websites from which images were scraped are [Flipkart](www.flipkart.com) and [hm.com](www.hm.com).
2. Images were scraped by finding their source location from their tags.
The tags were located from the webpage after observation and finding a identifier unique to them, in this case their **class** or **id**.
3. Using BeautifulSoup, these tags are obtained and their **src** attribute is used to get the link for the image
4. Then the urlretrieve function is used to download the image and store it on the system.
