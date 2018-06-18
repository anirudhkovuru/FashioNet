# FashioNet
This project focuses on using different types of **General Adversarial Networks** or GAN to create new shirt designs. The GAN helps create new designs by generating new shirts from noise signals after being trained over a dataset of shirts. Two main types of GANs are given focus in this project namely :
- DCGAN
- SRGAN

## Steps of the project
This project involves 4 steps -

### 1. Web Scraping
In this step, the **BeautifulSoup** module of python is used to access the position of the required image within the webpage. Using the **urlretrieve** function of the **urllib2** module, the image is extracted and then stored on our local machine. This step helps us obtain our training and testing datasets.\
[More info](./web-scraping)

### 2. Data Preprocessing
In this step, the data is manually run through checks to remove any discrepancies. This is to make sure the dataset is homogenous and all the features of the images remain the same. For example :- An image with multiple shirts would have to be removed from a dataset where every images has only a single shirt.

### 3. Training and using the GANs
In this step, we use the GANs to generate new images of shirts from noise. Two types of GANs have been used as stated above.\
These have been implemented in python using the **keras** library with a **tensorflow** backend.\
The GANs together form a Stacked GAN where the output of the DCGAN becomes the input to the SRGAN.\
[More info](./Backend)

### 4. Integrating with the web for display
