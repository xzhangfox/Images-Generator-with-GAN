import os
import random
import cv2
# import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Reshape, Dense,\
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



real_100 = np.load('x_train.npy')

real = np.ndarray(shape=(real_100.shape[0], 64, 64, 3))
for i in range(real_100.shape[0]):
    real[i] = cv2.resize(real_100[i], (64, 64))

img_size = real[0].shape

# latent space of noise
z = (100,)


def generator_vae():
    noise = Input(shape=z)
    x = Dense(4*4*512, activation='elu', kernel_initializer=weight_init)(noise)
    x = Reshape((4, 4, 512))(x)
    x = Conv2D(256, (3, 3), activation='elu', kernel_initializer=weight_init, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='elu', kernel_initializer=weight_init, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='elu', kernel_initializer=weight_init, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='elu', kernel_initializer=weight_init, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    generated = Conv2D(3, (3, 3), activation='tanh', kernel_initializer=weight_init, padding='same')(x)

    generator = Model(inputs=noise, outputs=generated)
    return generator

# Build Generator
def generator_conv():
    noise = Input(shape=z)
    x = Dense(4*4*512, activation='relu', kernel_initializer=weight_init)(noise)
    x = Reshape((4, 4, 512))(x)
    x = Conv2DTranspose(256,
                        kernel_size=(2, 2),
                        strides=(2, 2),
                        activation='relu',
                        data_format='channels_last',
                        kernel_initializer=weight_init)(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(128,kernel_size=(2, 2), strides=(2, 2), activation='relu',
                        data_format='channels_last', kernel_initializer=weight_init)(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), activation='relu',
                        data_format='channels_last', kernel_initializer=weight_init)(x)
    x = BatchNormalization()(x)
    generated = Conv2DTranspose(3, kernel_size=(2, 2), strides=(2, 2), activation='tanh',
                        data_format='channels_last', kernel_initializer=weight_init)(x)

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
    x = Conv2D(16, kernel_size=(3, 3), kernel_initializer=weight_init)(img)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(32, kernel_size=(3, 3), kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, kernel_size=(3, 3), kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=(3, 3), kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=(3, 3), kernel_initializer=weight_init)(x)
    x = Flatten()(x)
    out = Dense(1, activation='sigmoid', kernel_initializer=weight_init)(x)

    discriminator = Model(inputs=img, outputs=out)
    return discriminator

# discr = discriminator()
# discr.summary()

def generator_fc():
    noise = Input(shape=z)
    x = Dense(256, kernel_initializer=weight_init)(noise)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1024, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
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



# GAN model compiling
class GAN():
    def __init__(self, model='conv', img_shape=(64, 64, 3), latent_space=(100,)):
        self.img_size = img_shape  # channel_last
        self.z = latent_space
        self.optimizer = Adam(0.0002, 0.5)

        if model == 'conv':
            self.gen = generator_conv()
            self.discr = discriminator_conv()
        else:
            self.gen = generator_fc()
            self.discr = discriminator_fc()
        self.gen.compile(self.optimizer, loss='binary_crossentropy')
        self.discr.compile(self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        self.discr.trainable = False
        noise = Input(self.z)
        fake = self.gen(noise)
        out = self.discr(fake)

        self.train_gen = Model(inputs=noise, outputs=out)
        self.train_gen.compile(self.optimizer, loss='binary_crossentropy')
        self.loss_D, self.loss_G = None, None

    def Generator(self):

        return

    def Descriminator(self):

        return

    def train(self, imgs, epochs=10, batch_size=256, iteration1=5, iteration2=5):
        # load data
        imgs = (imgs - 127.5)/127.5
        bs_half = batch_size//2

        for epoch in range(epochs):
            idx = np.random.randint(0, imgs.shape[0], bs_half)
            real_img = imgs[idx]

            for _ in range(iteration1):
                # Generate a half batch of new images
                noise = np.random.normal(0, 1, size=((bs_half,) + self.z))
                fake_img = self.gen.predict(noise)
                # Train the discriminator
                loss_fake = self.discr.train_on_batch(fake_img, np.zeros((bs_half, 1)))
                loss_real = self.discr.train_on_batch(real_img, np.ones((bs_half, 1)))
                self.loss_D = 0.5 * np.add(loss_fake, loss_real)

            # Reinforce to learn a better generator
            for _ in range(iteration2):
                # Train the generator
                noise = np.random.normal(0, 1, size=((batch_size,) + self.z))
                self.loss_G = self.train_gen.train_on_batch(noise, np.ones(batch_size))

            if (epoch + 1)*10 % epochs == 0:
                print('Epoch (%d / %d): [Loss_D: %f, acc: %.2f%%]; [Loss_G: %f]' %
                  (epoch+1, epochs, self.loss_D[0], 100*self.loss_D[1], self.loss_G))
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
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i,j].axis('off')
            cnt += 1
    plt.show()
    return
# train GAN
gan = GAN(model='conv')
LEARNING_STEPS = 51
for learning_step in range(LEARNING_STEPS):
    print('LEARNING STEP # ', learning_step+1, '-'*50)
    iteration = [3, 3]
    # Adjust the learning times to balance G and D at a competitive level.
    if gan.loss_D is not None:
        acc = gan.loss_D[1]
        iteration = [int(3 * (1-acc)) + 1, int(3 * acc) + 1]
    gan.train(real, epochs=30, batch_size=128,
              iteration1=iteration[0], iteration2=iteration[1])
    if (learning_step+1)%3 == 0:
        plt_img(gan)
