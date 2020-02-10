import os
import random
import cv2
# import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Reshape, Dense, Dropout, \
    Activation, LeakyReLU, Conv2D, Conv2DTranspose, \
    MaxPooling2D, UpSampling2D, Flatten, BatchNormalization
from keras.initializers import glorot_uniform, glorot_normal
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
# tf.random.set_seed(SEED)
weight_init = glorot_normal(seed=SEED)

LR = 0.0002
from keras.datasets.fashion_mnist import load_data


# real_100 = np.load('x_train.npy')
(real_100,_), (_,_) = load_data()
X = real_100.reshape((-1, 28, 28, 1))
# convert from ints to floats
real_100 = X.astype('float32')
# scale from [0,255] to [-1,1]
# real_100 = (X - 127.5) / 127.5

real = np.ndarray(shape=(real_100.shape[0], 64, 64, 1))
for i in range(real_100.shape[0]):
    real[i] = (cv2.resize(real_100[i], (64, 64))).reshape(64, 64, 1)


img_size = real[0].shape

# latent space of noise
z = (100,)
optimizer = Adam(lr=0.0002, beta_1=0.5)

# Build Generator
def generator_conv():
    noise = Input(shape=z)
    x = Dense(4*4*256)(noise)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((4, 4, 256))(x)
    x = Conv2DTranspose(filters=128,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        padding='same')(x)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    generated = Conv2D(1, (8, 8), padding='same', activation='tanh')(x)

    generator = Model(inputs=noise, outputs=generated)
    return generator

# gen = generator()
# gen.summary()
# fake = gen.predict(np.random.normal(0, 1, size=(100,)).reshape(1, -1))
# plt.imshow(fake[0])
# plt.show()

# Build Discriminator
def discriminator_conv():
    img = Input(img_size)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(img)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)

    discriminator = Model(inputs=img, outputs=out)
    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return discriminator

# discr = discriminator()
# discr.summary()


def generator_fc():
    noise = Input(shape=z)
    x = Dense(256, kernel_initializer=weight_init)(noise)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(512, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(np.prod(img_size), activation='tanh', kernel_initializer=weight_init)(x)
    generated = Reshape(img_size)(x)
    generator = Model(inputs=noise, outputs=generated)
    return generator

def discriminator_fc():
    img = Input(shape=img_size)
    x = Flatten()(img)
    x = Dense(512, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(256, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    out = Dense(1, activation='sigmoid', kernel_initializer=weight_init)(x)

    discriminator = Model(inputs=img, outputs=out)
    return discriminator



def generator_trainer(generator, discriminator):

    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    return model

# GAN model compiling
class GAN():
    def __init__(self, model='conv', img_shape=(64, 64, 1), latent_space=(100,)):
        self.img_size = img_shape  # channel_last
        self.z = latent_space
        self.optimizer = Adam(0.0002, 0.5)

        if model == 'conv':
            self.gen = generator_conv()
            self.discr = discriminator_conv()
        else:
            self.gen = generator_fc()
            self.discr = discriminator_fc()

        self.train_gen = generator_trainer(self.gen, self.discr)
        # self.gen.compile(self.optimizer, loss='binary_crossentropy')
        # self.discr.compile(self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # self.discr.trainable = False
        # noise = Input(self.z)
        # fake = self.gen(noise)
        # out = self.discr(fake)

        # self.train_gen = Model(inputs=noise, outputs=out)
        # self.train_gen.compile(self.optimizer, loss='binary_crossentropy')
        self.loss_D, self.loss_G = [], []

    def Generator(self):

        return

    def Descriminator(self):

        return

    def train(self, imgs, epochs=50, batch_size=128):
        # load data
        imgs = (imgs - 127.5)/127.5
        bs_half = batch_size//2

        for epoch in range(epochs):
            # Get a half batch of random real images
            idx = np.random.randint(0, imgs.shape[0], bs_half)
            real_img = imgs[idx]

            # Generate a half batch of new images
            noise = np.random.normal(0, 1, size=((bs_half,) + self.z))
            fake_img = self.gen.predict(noise)
            # Train the discriminator
            loss_fake = self.discr.train_on_batch(fake_img, np.zeros((bs_half, 1)))
            loss_real = self.discr.train_on_batch(real_img, np.ones((bs_half, 1)))
            self.loss_D.append(0.5 * np.add(loss_fake, loss_real))

            # Train the generator
            noise = np.random.normal(0, 1, size=((batch_size,) + self.z))
            loss_gen = self.train_gen.train_on_batch(noise, np.ones(batch_size))
            self.loss_G.append(loss_gen)

            if (epoch + 1) * 10 % epochs == 0:
                print('Epoch (%d / %d): [Loss_D_real: %f, Loss_D_fake: %f, acc: %.2f%%] [Loss_G: %f]' %
                  (epoch+1, epochs, loss_real[0], loss_fake[0], 100*self.loss_D[-1][1], loss_gen))

        return


def plt_img(gan):
    r, c = 2, 4
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = gan.gen.predict(noise)
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    plt.show()
    return
# train GAN
gan = GAN(model='conv')
LEARNING_STEPS = 52
for learning_step in range(LEARNING_STEPS):
    print('LEARNING STEP # ', learning_step+1, '-'*50)
    # iteration = [5, 5]
    # Adjust the learning times to balance G and D at a competitive level.
    # if gan.loss_D is not None:
    #     acc = gan.loss_D[1]
    #     iteration = [int(5 * (1-acc)) + 1, int(5 * acc) + 1]
    gan.train(real, epochs=100, batch_size=128)
    if (learning_step+1)%1 == 0:
        plt_img(gan)
