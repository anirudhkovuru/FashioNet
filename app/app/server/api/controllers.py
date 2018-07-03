# Import flask and its related functions
from flask import Blueprint, request, session, jsonify, render_template, redirect, url_for
# Import exception for integrity error from sqlalchemy
from sqlalchemy.exc import IntegrityError

# Import the load_model from keras
from keras.models import load_model

# Import numpy, matplotlib, tensorflow and other utility libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import scipy
import glob

# Import the app and database from app folder
from app import db
from app import app

# Import DCGAN and SRGAN frameworks from the server folders
from app.server.dcgan.drive_dcgan import DCGAN
from app.server.srgan.drive_srgan import SRGAN

# Defining the models blueprint with url prefix /api
mod_models = Blueprint('models', __name__, url_prefix='/api')

# Defining the global variables and initializing to None
global dcgan
dcgan = None
global srgan
srgan = None
global graph
graph = None

# Function to load the saved models
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

# Initializing the display function at url /api/
@mod_models.route("/", methods = ['GET'])
def display():
    # Returning the index template (home page)
    return render_template("index.html")

# Initializing the get_image_display function at url /api/display
@mod_models.route("/display", methods = ['GET'])
def get_image_display():
    # Obtaining all filenames to be passed
    filenames = glob.glob1(os.path.join(app.static_folder, 'final-images/'), "*.jpg")
    # Processing the number of images to be placed on each side
    dirLen = len(filenames)
    if dirLen % 2 == 0:
        dirLen1 = dirLen // 2
        if dirLen1 % 2 != 0:
            dirLen1 += 1
    else:
        dirLen1 = dirLen // 2 + 1

    # Returning the display template (gallery page)
    return render_template("display.html", number1 = dirLen1, number2 = dirLen, filenames = filenames)

# Initializing the get_images function at url /api/get_images
@mod_models.route("/get_images", methods = ['POST'])
def get_images():
    # Obtaining the form data
    num = request.form["number"]
    try:
        unique = request.form["unique"]
    except:
        unique = "off"
    num = int(num)

    # If unique images were asked
    if unique == "on":
        print("Unique images needed...")

        # Obtaining all the currently generated images
        filenames = glob.glob1(os.path.join(app.static_folder, 'final-images/'), "*.jpg")
        # Removing all of the current images
        for file in filenames:
            os.remove(os.path.join(os.path.join(app.static_folder, 'final-images/'), file))

        # If the frameworks and models are not loaded then call load_gan()
        if dcgan == None or srgan == None or graph == None:
            load_gan()

        # Loop to generate required number of images
        for i in range(num):
            # input noise for the generator
            noise = np.random.normal(0, 1, (1, 200))

            # using the initialized graph to generate new images from the input noise
            with graph.as_default():
                gen_imgs = dcgan.generator.predict(noise)
                final_imgs = srgan.generator.predict(gen_imgs)

            # Bringing the pixel values to 0 to 1 range
            final_imgs = 0.5 * final_imgs + 0.5
            # Saving the image
            scipy.misc.imsave(os.path.join(app.static_folder, 'final-images/') + str(i) + '.jpg', final_imgs[0])
            print(str(i) + ".jpg created and saved.")

    # If more images aside from existing ones are asked
    elif unique == "off":
        # Finding out the number of images actually present
        actualNum = len(glob.glob1(os.path.join(app.static_folder, 'final-images/'), "*.jpg"))
        print("Actual number of images = " + str(actualNum))

        # If required number already exists
        if actualNum >= num:
            print("Required number already there...")

        # More images are needed
        else:
            print("Extra images needed...")

            # If the frameworks and models are not loaded then call load_gan()
            if dcgan == None or srgan == None or graph == None:
                load_gan()

            # Only generate the extra amount of images
            for i in range(actualNum, num):
                # input noise for the generator
                noise = np.random.normal(0, 1, (1, 200))

                # using the initialized graph to generate new images from the input noise
                with graph.as_default():
                    gen_imgs = dcgan.generator.predict(noise)
                    final_imgs = srgan.generator.predict(gen_imgs)

                # Bringing the pixel values to 0 to 1 range
                final_imgs = 0.5 * final_imgs + 0.5
                # Saving the image
                scipy.misc.imsave(os.path.join(app.static_folder, 'final-images/') + str(i) + '.jpg', final_imgs[0])
                print(str(i) + ".jpg created and saved.")

    # Redirecting to get_image_display on url /api/display (gallery page)
    return redirect(url_for('models.get_image_display'))
