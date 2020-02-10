import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model



def plt_img(gen, r=2, c=4):
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = gen.predict(noise)
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


for i in range(1, 18):
    gen = load_model('gan_generator_%d_v2.h5' % (5000*i))
    plt_img(gen, 2, 4)