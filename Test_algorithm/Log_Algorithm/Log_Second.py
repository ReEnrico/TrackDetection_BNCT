import os

import numpy   as np
from   skimage import io

import sys
sys.path.append("..")
import BlobFunction as bf

#create a text file with blobs' coordinates and radius
def tabulate(blob_list: np.array, name: str, thresh: float):
    dump = np.savetxt("{name}_{thresh:.4f}.txt".format(name=name, thresh=thresh), 
                        blob_list, fmt='%12.4e', header="%10s%13s%13s"%("y","x","r"))

def log_function(image: np.array, first_path: str, name: str):

    file_name = os.path.join(first_path, name)
    min_sigma = 3
    max_sigma = 20
    num_sigma = 10
    overlap = 0.5
    log_scale = False
    exclude_border = False
    log_threshold = 0.0254
    log_img = bf.Blob_Laplacian_of_Gaussian(image,
                                            msigma=min_sigma,
                                            xsigma=max_sigma,
                                            nusigma=num_sigma,
                                            thold=log_threshold,
                                            olap=overlap,
                                            lscale=log_scale,
                                            exborder=exclude_border)
    tabulate(log_img, file_name, log_threshold)     
    #I don't know if this is okay, but it should be a measure
    # to prevent the RAM to become full
    log_img = None
    
if __name__ == "__main__":

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    names = ["{:03d}".format(i) for i in range(1000)]  
    images = [io.imread(os.path.join("..", "..", "img_PseudoTracks",
                    "{}.jpg".format(name))) for name in names]
    folder = os.path.join(THIS_FOLDER, "Second")
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    for i in range((len(images))):
        log_function(images[i], folder, names[i])
        print(i)