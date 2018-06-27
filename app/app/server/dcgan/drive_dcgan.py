from __future__ import print_function, division

# Importing keras and its various parts
from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

# Importing matplotlib, sys, numpy, cv2, PIL
import matplotlib.pyplot as plt
import sys
import numpy as np
import cv2
from PIL import Image
from os import listdir
from os.path import join, isfile

# DCGAN class
class DCGAN():

    # Initialization function
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 200

        # Adam optimizer for generator with learning rate as 0.0002 and bet_01 as 0.5
        optimizer_gen = Adam(0.0002, 0.5)

        # Adam optimizer for discriminator with learning rate as 0.0002 and bet_01 as 0.5
        optimizer_dis = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer_dis,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_gen)

    # Method to build generator
    def build_generator(self):

        # Initiate the generator model as Sequential
        model = Sequential()
        # Adding a dense layer with input as noise dimensions and output as
        # 128 x (output_img_side / 4) x (output_img_side / 4)
        model.add(Dense(128 * 16 * 16, activation="relu", input_dim=self.latent_dim))
        # Adding a reshape layer to convert to 12x12x128
        model.add(Reshape((16, 16, 128)))
        # Adding an upsampling layer to upscale the image
        model.add(UpSampling2D())
        # Adding a convolutional layer with 128 filters and filter of size 3x3
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        # Adding a batch normalization layer, to maintain the mean and stddev
        # at 0 and 1 respectively and 0.8 to control the momentum of this change
        model.add(BatchNormalization(momentum=0.8))
        # Adding the activation function for the layer, in this case ReLU
        model.add(Activation("relu"))
        # Adding an upsampling layer to upscale the image
        model.add(UpSampling2D())
        # Adding a convolutional layer with 64 filters and filter of size 3x3
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        # Adding a batch normalization layer, to maintain the mean and stddev
        # at 0 and 1 respectively and 0.8 to control the momentum of this change
        model.add(BatchNormalization(momentum=0.8))
        # Adding the activation function for the layer, in this case ReLU
        model.add(Activation("relu"))
        # Adding a convolutional layer with 3 filters and filter of size 3x3
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        # Adding the activation function for the layer, in this case tanh
        model.add(Activation("tanh"))

        # Prints the details of the generator net
        model.summary()

        # noise is initialized with shape as latent_dim
        noise = Input(shape=(self.latent_dim,))

        # Image output of the generator model after giving the noise input
        img = model(noise)

        # Return the generator model
        return Model(noise, img)

    # Method to build discriminator
    def build_discriminator(self):

        # Initiate the discriminator model as Sequential
        model = Sequential()

        # The layers for the discriminator
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        # Print the details of the discriminator net
        model.summary()

        # img is initialized with the image shape
        img = Input(shape=self.img_shape)

        # Validity as output of the discriminator after giving img as input
        validity = model(img)

        # Return discriminator model
        return Model(img, validity)

    # Method to obtain all shirt names
    def load_shirts(self):
        names = [f for f in listdir("drive/ZML/FashioNet/train-images") if isfile(join("drive/ZML/FashioNet/train-images", f))]
        return names

    # Method to load the shirt images
    def load_batch(self, idx, names):
        arr = []
        for i in idx:
            try:
                img = Image.open("drive/ZML/FashioNet/train-images/"+names[i])
                img = img.resize((self.img_rows,self.img_cols),Image.ANTIALIAS)
                temp = np.array(img)
                arr.append(temp)
                #print(names[i])
            except Exception as e:
                print(e)
                continue
        arr = np.array(arr)
        print(arr.shape)
        print(type(arr))
        return arr

    # Method to train the DCGAN
    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        print("loading data..")
        names = self.load_shirts()
        print("data loaded.")

        # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, len(names), batch_size)
            imgs = self.load_batch(idx, names)

            imgs = imgs / 127.5 - 1.

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    # Method to sample at intervals during training
    def save_imgs(self, epoch):
        r, c = 3, 3
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("drive/ZML/FashioNet/DCGAN/images/testshirt_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    # Create DCGAN object
    dcgan = DCGAN()
    dcgan.generator = load_model('drive/ZML/FashioNet/saved-models/dcgenerator-model.h5')
    dcgan.discriminator = load_model('drive/ZML/FashioNet/saved-models/dcdiscriminator-model.h5')
    # Train the DCGAN
    dcgan.train(epochs=9000, batch_size=32, save_interval=20)
    # Save the generator-discriminator combined model
    dcgan.generator.save("drive/ZML/FashioNet/saved-models/dcgenerator-model.h5")
    dcgan.discriminator.save("drive/ZML/FashioNet/saved-models/dcdiscriminator-model.h5")
