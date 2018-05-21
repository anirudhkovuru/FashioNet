from dcgan import DCGAN
from keras.models import load_model
import numoy as np
import matplotlib.pyplot as plt

gan = DCGAN()
r, c = 3, 3
gan.combied = load_model("saved-models/combined-model.h5")
latent_dim = 200

def get_images(num):
    for i in range(num):
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = gan.generator.predict(noise)
        plt.imshow(gen_imgs)
        plt.show
