# FashioNet
This project focuses on using different types of **Generative Adversarial Networks** or GAN to create new shirt designs. The GAN helps create new designs by generating new shirts from noise signals after being trained over a dataset of shirts. Two main types of GANs are given focus in this project namely :
- DCGAN
- SRGAN

**Note: All of the above code (except the frontend) has been written keeping in mind that these would run in google colaboratory's python environment.**

## Steps of the project
This project involves 4 steps -

### 1. Web Scraping
In this step, the **BeautifulSoup** module of python is used to access the position of the required image within the webpage. Using the **urlretrieve** function of the **urllib2** module, the image is extracted and then stored on our local machine. This step helps us obtain our training and testing datasets.\
[More info](./web-scraping)

### 2. Data Preprocessing
In this step, the data is manually run through checks to remove any discrepancies. This is to make sure the dataset is homogenous and all the features of the images remain the same.\
**For example** :- An image with multiple shirts would have to be removed from a dataset where every images has only a single shirt.

### 3. Training and using the GANs
In this step, we use the GANs to generate new images of shirts from noise. Two types of GANs have been used as stated above.\
These have been implemented in python using the **keras** library with a **tensorflow** backend.\
The GANs together form a Stacked GAN where the output of the DCGAN becomes the input to the SRGAN.\
[More info](./app/app/server)

### 4. Integrating with the web for display
In this step, the web interface for the images is provided. Images are generated at the server and sent to the user for viewing.
The user can request the generation of more images.
The user can also download images which he likes and wishes to use for brainstorming new designs.\
The web based framework of this application is built in python using the **flask** library along with html and javascript.\
[More info](./app)
