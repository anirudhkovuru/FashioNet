from flask import Blueprint, request, session, jsonify, render_template, redirect, url_for
from sqlalchemy.exc import IntegrityError

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import scipy
import glob

from app import db
from app import app
from app.server.dcgan.drive_dcgan import DCGAN
from app.server.srgan.drive_srgan import SRGAN

mod_models = Blueprint('models', __name__, url_prefix='/api')

def load_gan():
    global dcgan
    global srgan

    # calling the constructor of the DCGAN
    dcgan = DCGAN()
    # calling the constructor of the SRGAN
    srgan = SRGAN()

    # Loading the saved trained models
    dcgan.generator = load_model(os.path.join(app.static_folder, 'saved-models/') + "dcgenerator-model.h5")
    srgan.generator = load_model(os.path.join(app.static_folder, 'saved-models/') + "srgenerator-model.h5")

    # initializing a graph to help with generating the images
    global graph
    graph = tf.get_default_graph()

@mod_models.route("/", methods = ['GET'])
def display():
    return render_template("index.html")

@mod_models.route("/display", methods = ['GET'])
def get_image_display():
    filenames = glob.glob1(os.path.join(app.static_folder, 'final-images/'), "*.jpg")
    dirLen = len(filenames)
    if dirLen % 2 == 0:
        dirLen1 = dirLen // 2
    else:
        dirLen1 = dirLen // 2 + 1
    return render_template("display.html", number1 = dirLen1, number2 = dirLen, filenames = filenames)

@mod_models.route("/get_images", methods = ['POST'])
def get_images():
    # Loading the GAN models
    num = request.form["number"]
    try:
        unique = request.form["unique"]
    except:
        unique = "off"

    #print(num)
    #print(unique)

    num = int(num)

    if unique == "on":
        print("Unique images needed...")
        filenames = glob.glob1(os.path.join(app.static_folder, 'final-images/'), "*.jpg")

        for file in filenames:
            os.remove(os.path.join(os.path.join(app.static_folder, 'final-images/'), file))

        load_gan()
        for i in range(num):
            # input noise for the generator
            noise = np.random.normal(0, 1, (1, 200))

            # using the initialized graph to generate new images from the input noise
            with graph.as_default():
                gen_imgs = dcgan.generator.predict(noise)
                final_imgs = srgan.generator.predict(gen_imgs)

            final_imgs = 0.5 * final_imgs + 0.5
            scipy.misc.imsave(os.path.join(app.static_folder, 'final-images/') + str(i) + '.jpg', final_imgs[0])
            print(str(i) + ".jpg created and saved.")

    elif unique == "off":
        actualNum = len(glob.glob1(os.path.join(app.static_folder, 'final-images/'), "*.jpg"))
        print("Actual number of images = " + str(actualNum))

        if actualNum >= num:
            print("Required number already there...")
            return render_template("display.html")

        else:
            print("Extra images needed...")
            load_gan()
            for i in range(actualNum, num):
                # input noise for the generator
                noise = np.random.normal(0, 1, (1, 200))

                # using the initialized graph to generate new images from the input noise
                with graph.as_default():
                    gen_imgs = dcgan.generator.predict(noise)
                    final_imgs = srgan.generator.predict(gen_imgs)

                final_imgs = 0.5 * final_imgs + 0.5
                scipy.misc.imsave(os.path.join(app.static_folder, 'final-images/') + str(i) + '.jpg', final_imgs[0])
                print(str(i) + ".jpg created and saved.")

    return redirect(url_for('models.get_image_display'))
