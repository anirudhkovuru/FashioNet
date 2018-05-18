from __future__ import print_function, division

# Import keras and its various parts
from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

# Import matplotlib for storing the images
import matplotlib.pyplot as plt

import sys

# Import numpy for array operations
import numpy as np

# Import for reading images
import cv2

# GAN class
class GAN():

    # Initializer for the GAN class
    def __init__(self):

        # Image properties
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3

        # Image shape
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Size of the noise vector
        self.latent_dim = 200

        # Initialize the optimizer using Adam with learning rate 0.0002
        # and beta_01 is 0.5
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Compile discriminator with binary cross entropy for loss
        # calculation, adam as optimizer metrics with accuracy
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise of size 100 as input and generates images
        z = Input(shape=(200,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines
        # validity
        validity = self.discriminator(img)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Method to build the generator
    def build_generator(self):

        # Typical layer by layer neural net. Initialized using Sequential()
        model = Sequential()

        # Adding a Dense layer to the network with input as 100 dim and output
        # as 256 dim
        model.add(Dense(256, input_dim=self.latent_dim))
        # Adding the activation function for the layer, in this case LeakyReLU
        model.add(LeakyReLU(alpha=0.2))
        # Adding a batch normalization layer, to maintain the mean and stddev
        # at 0 and 1 respectively and 0.8 to control the momentum of this change
        model.add(BatchNormalization(momentum=0.8))

        # Repeat for adding every new dense layer
        # Don't need to specify the input size of the next layer after the
        # first layer is initialized
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # Final layer with activation as tanh and using reshape to obtain the
        # image
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        # Prints the details of the generator net
        model.summary()

        # noise is initialized with shape as latent_dim
        noise = Input(shape=(self.latent_dim,))

        # Image output of the generator model after giving the noise input
        img = model(noise)

        # Return the generator model
        return Model(noise, img)

    # Method to build the discriminator
    def build_discriminator(self):

        # Typical layer by layer neural net. Initialized using Sequential()
        model = Sequential()

        # Layer to flatten the input image
        model.add(Flatten(input_shape=self.img_shape))
        # Adding a dense layer with output size 512
        model.add(Dense(512))
        # Apply the LeakyReLU function over the output of the dense layer
        model.add(LeakyReLU(alpha=0.2))

        # Repeat for adding every new dense layer
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))

        # Adding the final dense layer and then setting sigmoid as the
        # activation function
        model.add(Dense(1, activation='sigmoid'))

        # Prints the details of the discriminator net
        model.summary()

        # Setting the input in img
        img = Input(shape=self.img_shape)
        # Obtaining the validity from the model
        validity = model(img)

        # Return the discriminator model
        return Model(img, validity)

    # Method to train the GAN
    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = cifar10.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        # Expand dimensions to include channel data
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, 200))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 200))

            # Train the generator
            # (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # Method to obtain samples during training
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 200))
        gen_imgs = self.generator.predict(noise)

        # Rescale images to 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    # Create a GAN object
    gan = GAN()
    # Training the GAN
    gan.train(epochs=2000, batch_size=20, sample_interval=50)
    # Save the generator model
    gan.generator.save("saved-models/gen-model.h5")
