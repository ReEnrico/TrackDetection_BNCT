import sys
sys.path.append("..")
import os

import numpy             as np
from   numpy.random      import default_rng
from numpy               import genfromtxt
from   PIL               import Image

def create_pseudo_img(nx: int, ny: int, 
                      cvs_file: np.array, 
                      color_list: list, 
                      border=0):
    """This function create artificial image with chosen size.
        The background colour is randomly drawn from an array.
        The blobs are created randomly in terms of position, 
        while the radius and their colour are extracted from arrays.
    """

    back_rng   = default_rng()
    cnt_rng    = default_rng()
    radius_rng = default_rng()
    color_rng  = default_rng()
    a_rng      = default_rng()
    b_rng      = default_rng()

    image = np.zeros((nx, ny), dtype=float)
    rows = image.shape[0]
    columns = image.shape[1]
    # here the background is created by neatly filling each row 
    # with a grey value extracted from an array
    for i in range(0, rows):
        for j in range(0, columns):
            index_back = back_rng.integers(0, len(color_list))
            image[i][j] = color_list[index_back]

    # empty np array
    mask = np.zeros((nx, ny), dtype=np.bool_)

    # count is a random integer number which set the blobs' number
    # for an image    
    count = cnt_rng.integers(0, 600, endpoint=True)
    # tracks will be used to store information about blobs' position,
    # grey-level value and radius for each image
    tracks = np.zeros((count, 4), dtype=object)
    for i in range(count):
        # both the colour and the radius of the blobs are chosen 
        # randomly from their respective arrays
        index_color = color_rng.integers(0, len(cvs_file))
        index_radius = radius_rng.integers(0, len(cvs_file))

        # here blobs coordinates are created -> (x, y, r)
        # center coord y, random position in the image
        a = a_rng.integers(border, nx - border)
        #center coord x, random position in the image
        b = b_rng.integers(border, ny - border)
        # radius
        r = cvs_file[index_radius][-1]/2
        
        # True (blobs)/False(background) image is made
        # the entire image is scanned by choosing an x, y pair in order;
        # each time their distance is less than the radius, 
        # the circle centred at (x, y) and with that radius is drawn 
        # and coloured making it a blob
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        m = x * x + y * y <= r * r
        mask = np.logical_or(mask, m)
        color = cvs_file[index_color][2]
        image[m] = color

        #contain (y, x) and radius of pseudo blob
        tracks[i] = np.array([a, b, r, color], dtype=object)

    return image, mask, tracks

def image_saving(image: np.array,
                track: int,
                name: str):
    """This function simply saves the images created in  jpg format
        and creates the corresponding text files with blob features.    
    """
    im = Image.fromarray(image)
    if im.mode != 'L':
        im = im.convert('L')
    im.save(os.path.join("{}.jpg".format(name)))
    np.savetxt(os.path.join("{}.txt".format(name)), track, fmt='%12.4e', 
                header="{:12s} {:12s} {:12s} {}".format("y", "x", "r", "color"))


if __name__ == "__main__":
    # folder for temporaly store the image
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(THIS_FOLDER, "Temp_folder")
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    # upload of background and blobs information
    cvs_data = genfromtxt('Results.csv', delimiter=',', skip_header=1)
    background_color = np.load("real_back_color.npy", allow_pickle=True)
    # loop for the creation of images
    for i in range(1000):
        name = os.path.join(folder, "{:03d}".format(i))
        image, mask, tracks = create_pseudo_img(1944, 
                                                2592,
                                                cvs_data, 
                                                background_color,
                                                border=5)
        image_saving(image, tracks, name)