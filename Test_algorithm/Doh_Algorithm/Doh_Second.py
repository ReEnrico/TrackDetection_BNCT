import os
import glob
import time
import random

import numpy   as np
from   skimage import io

import sys
sys.path.append("..")
import BlobFunction as bf

def get_all_needed(folder_name: str, number:int):
    name_list = glob.glob(os.path.join(folder_name, "Group_{}".format(number), 
                        "Results", "*.txt"))
    name_list.sort()
    name_list = [name[-14:-11] for name in name_list]    
    image_array = [io.imread(os.path.join("..", "..", "img_PseudoTracks",
                    "{}.jpg".format(name))) for name in name_list]
    
    return name_list, image_array

    #create a text file with blobs' coordinates and radius
def tabulate(blob_list: np.array, name: str, thresh: float):
    dump = np.savetxt("{name}_{thresh:.5f}.txt".format(name=name, thresh=thresh), 
                        blob_list, fmt='%12.4e', header="%10s%13s%13s"%("y","x","r"))
    
#text file which store information about  #blobVSthreshold at fix min_sigma
def tabulate_result(blob_list: np.array, name: str):
    dump = np.savetxt("{}_result.txt".format(name), blob_list, 
                        fmt='%10.4f', header="{:15s} {:15s} {}".format("thresh", "tracks", "speed"))
    

def doh_function(image: np.array, first_path: str,
                 second_path: str, name: str):

    file_name = os.path.join(first_path, name)
    result_name = os.path.join(second_path, name)
    min_sigma = 3
    max_sigma = 20
    num_sigma = 10
    overlap = 0.5
    log_scale = False
    result = []
    doh_threshold = 0.001
    for i in range(31):
        start = time.time()
        doh_img = bf.Blob_Determinant_of_Hessian(image,
                                                 msigma=min_sigma,
                                                 xsigma=max_sigma,
                                                 nusigma=num_sigma,
                                                 thold=doh_threshold,
                                                 olap=overlap,
                                                 lscale=log_scale)
        end = time.time()
        speed = end - start
        total_blobs = len(doh_img)
        tabulate(doh_img, file_name, doh_threshold)     
        result.append([doh_threshold, total_blobs, speed]) 
        #I don't know if this is okay, but it should be a measure
        # to prevent the RAM to become full
        doh_img = None
        doh_threshold -= 0.00001
    tabulate_result(result, result_name)
    result = None

if __name__ == "__main__":

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    for i in range(1, 6):
        print("Group {}".format(i))
        names, images = get_all_needed("First", i)
        folder = os.path.join(THIS_FOLDER, "Second", "Group_{}".format(i))
        res_folder = os.path.join(folder, "Results")
        try:
            os.makedirs(folder)
        except FileExistsError:
            pass
        try:
            os.makedirs(res_folder)
        except FileExistsError:
            pass

        for i in range((len(images))):
            doh_function(images[i], folder, res_folder, names[i])