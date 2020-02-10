import cv2
import os
import random
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply,\
    BatchNormalization, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
# tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

#load data
real_100 = np.load('x_train.npy')
real_img = np.ndarray(shape=(real_100.shape[0], 64, 64, 3))
for i in range(real_100.shape[0]):
    real_img[i] = cv2.resize(real_100[i], (64, 64))
img_size = real_img[0].shape
real_label = np.load('y_train.npy')


def generator():
    noise = Input(shape=(100,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(120, 100)(label))
    model_input = multiply([noise, label_embedding])

    x = Dense(256, input_dim=100)(model_input)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Dense(512, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Dense(1024, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Dense(np.prod(img_size), kernel_initializer=weight_init, activation='tanh')(x)
    img = Reshape(img_size)(x)

    generator = Model([noise, label], img)
    return generator

def discriminator():
    img = Input(shape=img_size)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(120, np.prod(img_size))(label))
    flat_img = Flatten()(img)
    model_input = multiply([flat_img, label_embedding])

    #x = Flatten(input_shape=img_size)(model_input)
    x = Dense(512, kernel_initializer=weight_init)(model_input)
    x = LeakyReLU(0.2)(x)
    x = Dense(256, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    x = Dense(128, kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid', kernel_initializer=weight_init)(x)

    discriminator = Model([img, label], out)
    return discriminator

class GAN():
    def __init__(self, img_shape=(64, 64, 3), latent_space=(100,)):
        self.img_size = img_shape
        self.z = latent_space
        self.optimizer = Adam(0.0002, 0.5)

        self.gen = generator()
        self.discr = discriminator()
        self.discr.compile(self.optimizer,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        noise = Input(self.z)
        label = Input(shape=(1,))
        fake = self.gen([noise, label])
        validity = self.discr([fake, label])

        self.discr.trainable = False

        self.train_gen = Model([noise, label], validity)
        self.train_gen.compile(self.optimizer,
                               loss='binary_crossentropy')
        self.loss_D, self.loss_G = None, None

    def Generator(self):
        return

    def Descriminator(self):
        return

    def train(self, imgs, labels, epochs=10, batch_size=256, iteration1=5, iteration2=5):
        imgs = (imgs - 127.5)/127.5
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            idx = np.random.randint(0, imgs.shape[0], batch_size)
            real_img, real_label = imgs[idx], labels[idx]

            for _ in range(iteration1):
                noise = np.random.normal(0, 1, (batch_size, 100))
                fake_img = self.gen.predict([noise, real_label])

                loss_real = self.discr.train_on_batch([real_img, real_label], valid)
                loss_fake = self.discr.train_on_batch([fake_img, real_label], fake)
                self.loss_D = 0.5 * np.add(loss_real, loss_fake)

            for _ in range(iteration2):

                sampled_labels = np.random.randint(0, 120, batch_size).reshape(-1, 1)

                self.loss_G = self.train_gen.train_on_batch([noise, sampled_labels], valid)

            print('Epoch (%d / %d): [Loss_D: %f, acc: %.2f%%]; [Loss_G: %f]' %
                  (epoch+1, epochs, self.loss_D[0], 100*self.loss_D[1], self.loss_G))
        return



def plt_img(gan):
    r, c = 2, 5
    noise = np.random.normal(0, 1, (120, 100))
    noise_label = np.arange(0, 120).reshape(-1, 1)
    gen_imgs = gan.gen.predict([noise, noise_label])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title("Digit: %d" % noise_label[cnt])
            axs[i,j].axis('off')
            cnt += 1
    plt.show()
    return

# train GAN
gan = GAN()
LEARNING_STEPS = 500
for learning_step in range(LEARNING_STEPS):
    print('LEARNING STEP # ', learning_step+1, '-'*50)
    iteration = [5, 5]
    # Adjust the learning times to balance G and D at a competitive level.
    if gan.loss_D is not None:
        acc = gan.loss_D[1]
        iteration = [int(5 * (1-acc)) + 1, int(5 * acc) + 1]
    gan.train(real_img, real_label, epochs=100, batch_size=128,
              iteration1=iteration[0], iteration2=iteration[1])
    plt_img(gan)
