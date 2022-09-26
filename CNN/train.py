import keras
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os.path
from skimage.io import imread 

img_path = "../img/"
max_n_traks = 2000
im_shape = (1944,2592)
im_padding = (56,8)

# Get the image data and labels
n_img = 240
r_max = 100
images = np.zeros((n_img,im_shape[0]+im_padding[0],im_shape[1]+im_padding[1],1))
labels = np.zeros((n_img,max_n_traks,3))
Plot = False

im_i = np.array(range(n_img))
np.random.shuffle(im_i)

for i in im_i:
    im = imread(os.path.join(img_path,"{:03d}.jpg".format(i)))
    lb = np.loadtxt(os.path.join(img_path,"{:03d}.txt".format(i)))
    
    # normalize labels and assign
    # but first check if label array is not empty
    if len(lb) > 0:
        lb[:,0] /= im_shape[0]
        lb[:,1] /= im_shape[1]
        lb[:,2] /= r_max
        labels[i,0:lb.shape[0]] = lb

    # normalize image and assign
    im = im / 255
    images[i,im_padding[0]//2:-im_padding[0]//2,im_padding[1]//2:-im_padding[1]//2,0] = im
    
    if Plot:
        print(im)
        print(lb)
        plt.imshow(im)
        plt.scatter(lb[:,1]*im_shape[1],
                    lb[:,0]*im_shape[0],
                    lb[:,2]*r_max)
        plt.show()


def CNN_model(max_n_traks):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(16,
                                  (3,3),
                                  activation="relu",
                                  input_shape=(im_shape[0]+im_padding[0],im_shape[1]+im_padding[1],1),
                                  padding="same",
                                 ))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Conv2D(32,
                                  (3,3),
                                  activation="relu",
                                  padding="same"
                               ))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Conv2D(64,
                                  (3,3),
                                  activation="relu",
                                  padding="same"
                                 ))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Conv2D(128,
                                  (3,3),
                                  activation="relu",
                                  padding="same"
                                 ))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Conv2D(128,
                                  (3,3),
                                  activation="relu",
                                  padding="same"
                                 ))
    model.add(keras.layers.Reshape((2000,1296)))
    model.add(keras.layers.Dense(3))

    model.summary()
    return model


model = CNN_model(max_n_traks)
model.compile(optimizer="adam",loss="mse",metrics=["mae"])
history = model.fit(images[0:200],
                    labels[0:200],
                    validation_data=(images[200:220],labels[200:220]),
                    epochs=20,
                    batch_size=1)

test_loss, test_acc = model.evaluate(images[220:240],labels[220:240])
print("test loss{}\ntest accuracy {}\n".format(test_loss, test_acc))

y_pred = model.predict(images[220:240])
for i in range(220,240):
    print("{:>8.4f} - {:>8.4f}, {:>8.4f}".format(*y_pred[i-220]))
    print("{:>8.4f} - {:>8.4f}, {:>8.4f}\n".format(*labels[i]))

# save model
model.save("t_detect.h5")
keras.backend.clear_session()
