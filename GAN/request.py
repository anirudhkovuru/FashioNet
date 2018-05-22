import requests
import matplotlib.pyplot as plt
import json
import numpy as np

req = requests.get("http://127.0.0.1:5000/")
img = req.text
# loading the data as a JSON
img = json.loads(img)
print(type(img["image"]))
# converting into a numpy array to show the image
img = np.asarray(img["image"])
plt.imshow(img[0])
plt.show()
