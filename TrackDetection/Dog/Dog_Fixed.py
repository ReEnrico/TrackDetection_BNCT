import os

import numpy   as np
from   skimage import io

import BlobFunction as bf

#create a text file with blobs' coordinates and radius
def tabulate(blob_list: np.array, name: str):
    dump = np.savetxt("{name}.txt".format(name=name), 
                        blob_list, fmt='%12.4e', header="%10s%13s%13s"%("y","x","r"))

def dog_function(image: np.array, first_path: str, name: str):

    file_name = os.path.join(first_path, name)
    min_sigma = 3
    max_sigma = 8
    ratio = 1.3
    dog_threshold = 0.0272
    overlap = 0.5
    exclude_border = False
    dog_img = bf.Blob_Difference_of_Gaussian(image, 
                                            msigma=min_sigma,
                                            xsigma=max_sigma,
                                            ratio=ratio,
                                            thold=dog_threshold,
                                            olap=overlap,
                                            exborder=exclude_border)
    tabulate(dog_img, file_name)     
    #I don't know if this is okay, but it should be a measure
    # to prevent the RAM to become full
    dog_img = None
        
if __name__ == "__main__":

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    names = ["{:03d}".format(i) for i in range(240)]  
    images = [io.imread(os.path.join("..", "img",
                    "{}.jpg".format(name))) for name in names]
    folder = os.path.join(THIS_FOLDER, "Dog_result", "Fixed")
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    for i in range((len(images))):
        dog_function(images[i], folder, names[i])
        print(i)