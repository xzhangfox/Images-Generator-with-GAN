import os
import random
import cv2
# import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Reshape, Dense, Dropout, \
    Activation, LeakyReLU, Conv2D, Conv2DTranspose, \
    Multiply, Embedding, Concatenate, \
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


X_original = np.load('x_train.npy')

X_real = np.ndarray(shape=(X_original.shape[0], 64, 64, 3))
for i in range(X_original.shape[0]):
    X_real[i] = cv2.resize(X_original[i], (64, 64))

y_real = np.load('y_train.npy')
n_classes = y_real.max() - y_real.min() + 1
img_size = X_real[0].shape

# latent space of noise
z = (100,)
optimizer = Adam(lr=0.0002, beta_1=0.5)

# Build Generator
def generator_conv():
    label = Input((1,), dtype='int32')
    noise = Input(shape=z)

    le = Embedding(n_classes, 100)(label)
    le = Dense(4*4)(le)
    le = Reshape((4, 4, 1))(le)

    noi = Dense(4 * 4 * 256)(noise)
    noi = LeakyReLU(alpha=0.2)(noi)
    noi = Reshape((4, 4, 256))(noi)


    merge = Concatenate()([noi, le])
    ## Size: 4 x 4 257

    x = Conv2DTranspose(filters=256,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        padding='same')(merge)
    ## Size: 8 x 8 x 256
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)

    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    ## Size: 16 x 16 x 128

    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)

    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    ## Size: 32 x 32 x 64

    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)

    x = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same')(x)
    ## Size: 64 x 64 x 32

    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)

    generated = Conv2D(3, (8, 8), padding='same', activation='tanh')(x)
    ## Size: 64 x 64 x 3

    generator = Model(inputs=[noise, label], outputs=generated)
    return generator

# gen = generator()
# gen.summary()
# fake = gen.predict(np.random.normal(0, 1, size=(100,)).reshape(1, -1))
# plt.imshow(fake[0])
# plt.show()

# Build Discriminator
def discriminator_conv():
    label = Input((1,), dtype='int32')
    img = Input(img_size)

    le = Embedding(n_classes, 100)(label)
    le = Dense(img_size[0] * img_size[1])(le)
    le = Reshape((img_size[0], img_size[1], 1))(le)

    merge = Concatenate()([img, le])
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(merge)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)

    discriminator = Model(inputs=[img, label], outputs=out)
    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return discriminator

# discr = discriminator()
# discr.summary()


# def generator_fc():
#     noise = Input(shape=z)
#     x = Dense(256)(noise)
#     x = LeakyReLU(0.2)(x)
#     x = BatchNormalization()(x)
#     x = Dense(512)(x)
#     x = LeakyReLU(0.2)(x)
#     x = BatchNormalization()(x)
#     x = Dense(1024)(x)
#     x = LeakyReLU(0.2)(x)
#     x = BatchNormalization()(x)
#     x = Dense(np.prod(img_size), activation='tanh')(x)
#     generated = Reshape(img_size)(x)
#     generator = Model(inputs=noise, outputs=generated)
#     return generator
#
# def discriminator_fc():
#     img = Input(shape=img_size)
#     x = Flatten()(img)
#     x = Dense(512)(x)
#     x = LeakyReLU(0.2)(x)
#     x = Dense(256)(x)
#     x = LeakyReLU(0.2)(x)
#     out = Dense(1, activation='sigmoid')(x)
#
#     discriminator = Model(inputs=img, outputs=out)
#     return discriminator



def generator_trainer(generator, discriminator):

    discriminator.trainable = False

    gen_noise, gen_label = generator.input
    gen_out = generator.output
    out = discriminator([gen_out, gen_label])
    model = Model([gen_noise, gen_label], out)

    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    return model

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

        self.train_gen = generator_trainer(self.gen, self.discr)
        self.loss_D, self.loss_G = [], []

    def train(self, dataset, epochs=50, batch_size=128):
        # load data
        imgs, labels = dataset
        imgs = (imgs - 127.5)/127.5
        bs_half = batch_size//2

        for epoch in range(epochs):
            # Get a half batch of random real images
            idx = np.random.randint(0, imgs.shape[0], bs_half)
            real_img, real_label = imgs[idx], labels[idx]

            # Generate a half batch of new images
            noise = np.random.normal(0, 1, size=((bs_half,) + self.z))
            noise_label = np.random.randint(0, n_classes, bs_half)
            fake_img = self.gen.predict([noise, noise_label])
            # Train the discriminator
            loss_fake = self.discr.train_on_batch([fake_img, noise_label], np.zeros((bs_half, 1)))
            loss_real = self.discr.train_on_batch([real_img, real_label], np.ones((bs_half, 1)))
            self.loss_D.append(0.5 * np.add(loss_fake, loss_real))

            # Train the generator
            noise = np.random.normal(0, 1, size=((batch_size,) + self.z))
            noise_label = np.random.randint(0, n_classes, batch_size)
            loss_gen = self.train_gen.train_on_batch([noise, noise_label], np.ones(batch_size))
            self.loss_G.append(loss_gen)

            if ((epoch + 1) * 10) % epochs == 0:
                print('Epoch (%d / %d): [Loss_D_real: %f, Loss_D_fake: %f, acc: %.2f%%] [Loss_G: %f]' %
                  (epoch+1, epochs, loss_real[0], loss_fake[0], 100*self.loss_D[-1][1], loss_gen))

        return


def plt_img(gan):
    r, c = 4, 4
    noise = np.random.normal(0, 1, (r * c, 100))
    noise_label = np.arange(0, c).reshape(-1, 1)
    noise_label = np.dot(noise_label, np.ones((1, r))).astype('int32').T.reshape(-1, 1)
    gen_imgs = gan.gen.predict([noise, noise_label])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i,j].set_title("Dog Type: %d" % noise_label[cnt])
            axs[i,j].axis('off')
            cnt += 1
    plt.show()
    return
# train GAN
gan = GAN(model='conv')
LEARNING_STEPS = 100
BATCH_SIZE = 128
# EPOCHS = X_real.shape[0]//BATCH_SIZE
EPOCHS = 300
SHOWIMG_PER = 2
for learning_step in range(LEARNING_STEPS):
    print('LEARNING STEP # ', learning_step+1, '-'*50)
    gan.train([X_real, y_real], epochs=EPOCHS, batch_size=BATCH_SIZE)
    if (learning_step+1) % SHOWIMG_PER == 0:
        plt_img(gan)
