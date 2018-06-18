# FashioNet
This project focuses on using different types of **General Adversarial Networks** or GAN to create new shirt designs. The GAN helps create new designs by generating new shirts from noise signals after being trained over a dataset of shirts. Two main types of GANs are given focus in this project namely :
- DCGAN
- SRGAN

## Steps of the project
This project involves 4 steps -

### 1. Web Scraping
In this step, the **BeautifulSoup** module of python is used to access the position of the required image within the webpage. Using the *urlretrieve* function of the **urllib2** module, the image is extracted and then stored on our local machine. This step helps us obtain our training and testing datasets.

### 2. Data Preprocessing
