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


# Models

GAN is a minimax problem, which is one of zero-sum non-cooperative games. Generator wants to maximize its performance, which works to generate images as real as possible to confuse the Discriminator. Discriminator wants to distinguish a mixture of original and generated images whether real or fake. In this game, zero-sum means if Generator is improved, then there must be an increased loss of Discriminator. Our aim is to find the lowest aggregate loss of them, where there is a Nash Equilibrium.  
![image](https://github.com/f0000000x/Images-Generator-with-GAN/blob/master/Images/strGan.png)

Conditional Generative Adversarial Networks (CGAN), an extension of the GAN, allows you to generate images with specific conditions or attributes. Same as GAN, CGAN also has a generator and a discriminator. However, the difference is that both the generator and the discriminator of CGAN receive some extra conditional information, such as the class of the image, a graph, some words, or a sentence. Because of that, CGAN can make generator to generate different types of images, which will prevent generator from generating similar images after multiple trainings. Also, we can control the generator to generate an image which will have some properties we want.
![image](https://github.com/f0000000x/Images-Generator-with-GAN/blob/master/Images/strCGAN.png)

If you are interested in the details of algorithms, model processing and results, please move to our report:  
https://github.com/f0000000x/Images-Generator-with-GAN/blob/master/Final-Group-Project-Report/FinalReport.pdf


# References
* Avinash H. (2017). The GAN Zoo. GitHub. https://github.com/hindupuravinash/the-gan-zoo
* Hongyi, L. (2018). GAN Lecture 1: Introduction. YouTube. https://www.youtube.com/watch?v=DQNNMiAP5lw&list=PLJV_el3uVTsMq6JEFPW35BCiOQT soqwNw&index=1
* Jason B. (2019). How to Develop a Conditional GAN (cGAN) From Scratch. Machine Learning Mastery. https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
* Jonathan H. (2018). GAN — Ways to improve GAN performance. Towards Data Science. https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
* Jonathan H. (2018). GAN — CGAN & InfoGAN (using labels to improve GAN). Medium. https://medium.com/@jonathan_hui/gan-cgan-infogan-using-labels-to-improve-gan-8ba4de5f9c3d
* Jonathan H. (2018). GAN — Why it is so hard to train Generative Adversarial Networks! Medium. https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b
* Jonathan H. (2018). GAN — DCGAN (Deep convolutional generative adversarial networks) . Medium. https://medium.com/@jonathan_hui/gan-dcgan-deep-convolutional-generative-adversarial-networks-df855c438f
* Jon G. (2019). Conditional generative adversarial nets for convolutional face generation. Stanford University.
* Kaggle Competition. (2019). Generative Dog Images. Kaggle. https://www.kaggle.com/c/generative-dog-images
* Naoki S. (2017). Up-sampling with Transposed Convolution. Medium. https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d033
* Utkarsh D. (2018). Keep Calm and train a GAN. Pitfalls and Tips on training Generative Adversarial Networks. Medium. https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9
