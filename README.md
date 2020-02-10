# Images Generator with Gan
In this project, we structured an image generator with the cutting-edge machine learning technic – GAN (Generative Adversarial Networks) based on the concepts of game theory, and Keras TensorFlow Core. We persued to leverage image data to generate new images under the mutual supervision of the internal neural networks.

Team Members:

Xi Zhang, Gaofeng Huang, Jun Ying

![image](https://github.com/f0000000x/Images-Generator-with-GAN/blob/master/Images/GAN.png)

# Background

Generative Adversarial Networks (GAN) is a cutting-edge technique of deep neural networks, which was first come up by Ian Goodfellow in 2014. In 2016, Yann LeCun, who is one of the leading scientists in AI, described GAN as “the coolest idea in machine learning in the last twenty years.”
GAN is a very new stuff and has a promising future. Especially in last year (2018), GAN was developed with an exponential increment. In other words, it is almost an infant technology. Although it is really new, there are bunch of models named with the suffix __GAN, such as Conditional GAN, DCGAN, Cycle GAN, Stack GAN. Fortunately, we can catch up the development of GAN now. Actually, we’ve learned every single component in GAN if I break down it.

# Motivation

Combined with the knowledge of neural network learned in Machine Learning II, we hope to learn more in-depth and interesting knowledge on this basis. After investigation, Generative Adversarial Networks completely meets our requirements. As a new technology, GAN has been developing vigorously in recent two years. It is based on our familiar neural network such as multilayer perceptron (MLP) and convolutional neural network (CNN), but brings new vitality to machine learning.


# Data Description

Our data comes from the Kaggle code competition, which was released in June 2019. Since the game has ended in August, the ranking of the leaderboard also has certain reference value for our own model progress.The open source data consists of the image archive and the annotation archive. After further study, we believe that the subfolder name of the picture package can fully perform the task of label, so our model only USES the picture compression package.  
https://www.kaggle.com/c/generative-dog-images


# Description of Models

* Based on the characteristics of stray animals, linear models are used to predict adoption rates and to find ways to increase adoption rates.
* Since cuteness cannot be accurately quantified by a limited set of features, it may be more accurate to use deep learning to process animals images directly.
* Create web pages for data visualization and user interaction. Users can enter animal features or upload images directly to get a prediction of the animal's adoption.
* (Optional)Apply GAN(Generative Adversarial Networks) as a data augmentation tool or struct a demo for entertainment.

# Reference
* Pet Statistics. (2020, February). Retrieved from https://www.aspca.org/animal-homelessness/shelter-intake-and-surrender/pet-statistics
