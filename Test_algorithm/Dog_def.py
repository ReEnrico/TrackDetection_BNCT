import os
import time
import numpy   as np
from   skimage import io

#import sys
#sys.path.append("..")
import BlobFunction as bf

#create a text file with blobs' coordinates and radius
def tabulate(blob_list: np.array, name: str):
    dump = np.savetxt("{name}.txt".format(name=name), 
                        blob_list, fmt='%12.4e', header="%10s%13s%13s"%("y","x","r"))

def dog_function(image: np.array, first_path: str, name: str):

    file_name = os.path.join(first_path, name)
    min_sigma = 3.0
    max_sigma = 9.0
    ratio = 1.2
    overlap = 0.5
    exclude_border = False
    dog_threshold = 0.026
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
    names = ["{:03d}".format(i) for i in range(1000)]  
    images = [io.imread(os.path.join("..", "Img_test",
                    "{}.jpg".format(name))) for name in names]
    folder = os.path.join(THIS_FOLDER, "Dog_Algorithm", "New_test")
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    #timer = np.zeros(1000)
    for i in range((len(images))):
        start= time.time()
        dog_function(images[i], folder, names[i])
        end = time.time()
        #timer[i] = end - start        
        print("Image processed {img} in {count:.3f}s".format(img=names[i],
                                                             count=end - start))
    #dump_time = np.savetxt("Execution_times.txt", timer, fmt='%.3f')
