from dcgan import DCGAN
from keras.models import load_model

def get_imgs():
    dg = DCGAN()
    dg.combined = load_model("saved-models/combined-model.h5")
     = 
    for i in range(50):
