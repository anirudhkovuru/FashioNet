from flask import Blueprint, request, session, jsonify, render_template, redirect
from sqlalchemy.exc import IntegrityError

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from app import db
from app.backend.dcgan import DCGAN
from app.backend.srgan import SRGAN

mod_models = Blueprint('models', __name__, url_prefix='/api')

def load_gan():
    global dcgan
    global srgan

    # calling the constructor of the DCGAN
    dcgan = DCGAN()
    # calling the constructor of the SRGAN
    srgan = SRGAN()

    # Loading the saved trained models
    dcgan.generator = load_model("../saved-models/dcgenerator-model.h5")
    srgan.generator = load_model("../saved-models/srgenerator-model.h5")

    # initializing a graph to help with generating the images
    global graph
    graph = tf.get_default_graph()

@mod_models.route("/", methods = ['GET', 'POST'])
def display_images():
    if 


@mod_models.route("/display", methods = ['GET','POST'])
def get_images():

    # Loading the GAN models
    load_gan()
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
