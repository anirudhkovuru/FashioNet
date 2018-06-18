from dcgan import DCGAN
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
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

@app.route("/", methods = ['GET','POST'])
def get_images():
    json_image = {}
    num = request.args.get('num')
    for i in range(int(num)):
        # input noise for the generator
        noise = np.random.normal(0, 1, (1, 200))
        # using the initialized graph to generate new images from the input noise
        with graph.as_default():
            gen_imgs = gan.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        # translating into a list and converting into a JSON to send as response
        gen_imgs = gen_imgs.tolist()
        json_image["image"+str(i+1)] = gen_imgs

    return jsonify(json_image)


if __name__ == '__main__':
    load_gan()
    app.run(debug = True, host = '0.0.0.0')
