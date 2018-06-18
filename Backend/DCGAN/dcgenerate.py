from drive_dcgan import DCGAN
from keras.models import load_model
import scipy
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

gan = DCGAN()

print("Loading model...")
gan.generator = load_model('drive/ZML/FashioNet/saved-models/dcgenerator-model(1).h5')
print("Model loaded.")

for i in range(30):
    print(i)
    noise = np.random.normal(0, 1, (1, gan.latent_dim))
    gen_imgs = gan.generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    scipy.misc.imsave('drive/ZML/FashioNet/dc-images/' + str(i) + '.jpg', gen_imgs[0])
