import requests
from bs4 import BeautifulSoup
import urllib.request as urllib2
import os

# Setting page number from which to scrape
page=1

# For naming the images
count=4080

# Looping through all pages
while(page<45):

	# checking if the catalogue page can be opened successfully
	try:
		print("\n\n\tpage is : "+str(page))
		# creating the link for the page
		url1 = "https://www.flipkart.com/men/shirts/formal-shirts/pr?otracker=categorytree&page="+str(page)+"&sid=2oq%2Cs9b%2Cmg4%2Cfh5&viewType=grid"
		html = requests.get(url1)
		soup = BeautifulSoup(html.content,"html.parser")

		# finding all links for the standalone pages of each shirt
		divTag=soup.find("div",{"class":"_2SxMvQ"})
		links=set(divTag.findAll("a"))
		link=list(links)
		page+=1
	except Exception as e:
		print("error:"+e)
		pass
	i=0
	while(i<len(link)):

		# checking if the webpage can be opened successfully
		try:
			# creating and requesting the link for the page
			url=requests.get("https://www.flipkart.com"+link[i].get("href"))
			print("the link:"+str(url))
			soup1=BeautifulSoup(url.content,"html.parser")
			title=soup1.find("h1")
			print(title)

			# finding the image tag whose alt is the same as the title
			image=soup1.find("img",{"alt":title.content})

			# getting the image src
			imagesrc=image.get("src")

			# Creating the directory for the shirt and its specifications
			if not os.path.exists("images/shirt"+str(count)):
				os.mkdir("images/shirt"+str(count))
				urllib2.urlretrieve(imagesrc, "images/shirt"+str(count)+"/"+str(count)+".jpg")

				# placing the shirt details into the folder
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
