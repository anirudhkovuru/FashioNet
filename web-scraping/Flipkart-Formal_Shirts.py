import requests
from bs4 import BeautifulSoup
import urllib.request as urllib2
import os

page=35
count=4080
while(page<45):
	try:
		print("\n\n\tpage is : "+str(page))
		url1 = "https://www.flipkart.com/men/shirts/formal-shirts/pr?otracker=categorytree&page="+str(page)+"&sid=2oq%2Cs9b%2Cmg4%2Cfh5&viewType=grid"
		html = requests.get(url1)
		soup = BeautifulSoup(html.content,"html.parser")
		divTag=soup.find("div",{"class":"_2SxMvQ"})
		links=set(divTag.findAll("a"))
		link=list(links)
		page+=1
	except Exception as e:
		print("error:"+e)
		pass
	i=0
	while(i<len(link)):
		try:
			url=requests.get("https://www.flipkart.com"+link[i].get("href"))
			print("the link:"+str(url))
			soup1=BeautifulSoup(url.content,"html.parser")
			title=soup1.find("h1")
			print(title)
			image=soup1.find("img",{"alt":title.content})
			imagesrc=image.get("src")
			if not os.path.exists("images/shirt"+str(count)):
				os.mkdir("images/shirt"+str(count))
				urllib2.urlretrieve(imagesrc, "images/shirt"+str(count)+"/"+str(count)+".jpg")

				f = open("images/shirt"+str(count)+"/"+str(count)+".txt", "a+")
				listElements = soup1.findAll("li", {"class":"_1KuY3T row"})
				for element in listElements:
					key = element.find("div").text
					value = element.find("li").text
					f.write(key+" : "+value+"\n")

				f.close()
			i+=1
			count+=1
		except Exception as e:
			print(e)
print("final page:"+str(page))