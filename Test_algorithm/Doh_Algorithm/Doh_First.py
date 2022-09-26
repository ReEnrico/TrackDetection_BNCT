import os
import glob
import time
import random

import numpy   as np
from   skimage import io

import sys
sys.path.append("..")
import BlobFunction as bf

def create_sample(numb_sample: int, numb_element: int):

    #created lists of image's name (string)
    img_txt = glob.glob(os.path.join("..", "..", "img_PseudoTracks", "*.jpg"))
    img_txt.sort()
    #list of string used for create sample group and file name
    names = [string[-7:-4] for string in img_txt]
    sample_list = []
    for i in range(numb_sample):
        selected_name = random.sample(names, numb_element)
        selected_name.sort()
        sample_list.append(selected_name)
        names = [name for name in names if not name in selected_name]
    
    return sample_list

#create a text file with blobs' coordinates and radius
def tabulate(blob_list: np.array, name: str, thresh: float):
    dump = np.savetxt("{name}_{thresh:.4f}.txt".format(name=name, thresh=thresh), 
                        blob_list, fmt='%12.4e', header="%10s%13s%13s"%("y","x","r"))
    
#text file which store information about  #blobVSthreshold at fix min_sigma
def tabulate_result(blob_list: np.array, name: str):
    dump = np.savetxt("{}_result.txt".format(name), blob_list, 
                        fmt='%10.4f', header="{:15s} {:15s} {}".format("thresh", "tracks", "speed"))
    

def doh_function(image: np.array, first_path: str,
                 second_path: str, name: str, lenght: int):

    file_name = os.path.join(first_path, name)
    result_name = os.path.join(second_path, name)
    min_sigma = 3
    max_sigma = 20
    num_sigma = 10
    overlap = 0.5
    log_scale = False
    result = []
    doh_threshold = 0.01
    for i in range(1000):
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
        if total_blobs != 0:
            tabulate(doh_img, file_name, doh_threshold)     
        result.append([doh_threshold, total_blobs, speed])
        if total_blobs >= lenght*2:
            break        
        #I don't know if this is okay, but it should be a measure
        # to prevent the RAM to become full
        doh_img = None
        doh_threshold -= 0.0004
    tabulate_result(result, result_name)
    result = None

if __name__ == "__main__":

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))  
    name_groups = create_sample(5, 120)
    numb_group = 1
    for group in name_groups:
        images = [io.imread(os.path.join("..", "..", "img_PseudoTracks", 
        "{}.jpg".format(name))) for name in group]
        texts = [np.loadtxt(os.path.join("..", "..", "img_PseudoTracks", 
        "{}.txt".format(name))) for name in group]
        folder = os.path.join(THIS_FOLDER, "Group_{}".format(numb_group))
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
            doh_function(images[i], folder, res_folder, group[i], len(texts[i]))
        numb_group += 1
