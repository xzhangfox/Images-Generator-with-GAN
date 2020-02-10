import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model



def plt_img(gen, r, c):
    noise = np.random.normal(0, 1, (r * c, 100))
    noise_label = np.arange(0, c).reshape(-1, 1)
    noise_label = np.dot(noise_label, np.ones((1, r))).astype('int32').T.reshape(-1, 1)
    gen_imgs = gen.predict([noise, noise_label])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0

    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            #axs[i, j].set_title("Dog type: %d" % noise_label[cnt])
            axs[i,j].axis('off')
            cnt += 1
    plt.show()
    return


#for i in range(1, 20):
#    gen = load_model('cgan_generator_%d_v4.h5' % (3000*i))
#    plt_img(gen, 4, 4)
gen = load_model('cgan_generator_39000_v4.h5')
plt_img(gen, 8, 8)