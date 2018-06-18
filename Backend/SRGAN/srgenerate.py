from drive_srgan import SRGAN
from keras.models import load_model
import scipy
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

gan = SRGAN()

print("Loading model...")
gan.generator = load_model('drive/ZML/FashioNet/saved-models/srgenerator-model.h5')
print("Model loaded.")

path = glob('drive/ZML/FashioNet/dc-images/*')

name = 0
for img_path in path:
    print(img_path)
    input = scipy.misc.imread(img_path, mode='RGB').astype(np.float)
    input = input / 127.5 - 1

    gen_imgs = gan.generator.predict(input.reshape(1,64,64,3))

    gen_imgs = 0.5 * gen_imgs + 0.5
    scipy.misc.imsave('drive/ZML/FashioNet/final-images/' + str(name) + '.jpg', gen_imgs[0])
    name += 1
