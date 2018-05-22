import requests
import matplotlib.pyplot as plt
import json
import numpy as np

num = input("enter the number of pics needed : ")
req = requests.get("http://192.168.1.80:5000/?num="+str(num))
img = req.text
# loading the data as a JSON
img = json.loads(img)
for i in range(int(num)):
    image = img["image"+str(i+1)]
    plt.imshow(image[0])
    plt.show()
