from dcgan import DCGAN
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from flask import Blueprint, Flask, jsonify
import tensorflow as tf

app = Flask(__name__)
gan = None


def load_gan():
    global gan
    # calling the constructor of the DCGAN
    gan = DCGAN()
    # Loading the saved trained model
    gan.generator = load_model("saved-models/generator-model.h5")
    global graph
    # initializing a graph to help with generating the images
    graph = tf.get_default_graph()

@app.route("/", methods = ['GET'])
def get_images():
    # input noise for the generator
    noise = np.random.normal(0, 1, (1, 200))
    # using the initialized graph to generate new images from the input noise
    with graph.as_default():
        gen_imgs = gan.generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    # translating into a list and converting into a JSON to send as response
    gen_imgs = gen_imgs.tolist()
    return jsonify({"image" :  gen_imgs})

if __name__ == '__main__':
    load_gan()
    app.run(debug = True, host = '0.0.0.0')
