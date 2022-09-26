from math import sqrt

import numpy as np
from skimage.feature import blob_log, blob_dog, blob_doh
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage import exposure



#Questo è un modulo da importare in altri file, le tre funzioni definite richiedono come input un'immagine sotto forma di array,
#i valori di default settati sono "arbitrari"
#restituiscono un array contenente le coordinate (y - x) e il raggio dei blob

#il nome dato ai kwargs è molto brutto
def Blob_Laplacian_of_Gaussian(ndarray, msigma=1, xsigma=50, nusigma=10, thold=0.2, olap=0.5, lscale=False, exborder=False):
    invert_image = np.invert(ndarray)
    image_log = blob_log(invert_image, min_sigma=msigma, max_sigma=xsigma, num_sigma=nusigma, threshold=thold,
             overlap=olap, log_scale=lscale, exclude_border=exborder)
    image_log[:, 2] = image_log [:, 2] * sqrt(2)
    return(image_log)

def Blob_Difference_of_Gaussian(ndarray, msigma=1, xsigma=50, ratio=1.6, thold=2, olap=0.5, exborder=False):
    invert_image = np.invert(ndarray)
    image_dog = blob_dog(invert_image,  min_sigma=msigma, max_sigma=xsigma, sigma_ratio=ratio, threshold=thold,
                  overlap=olap, exclude_border=exborder)
    image_dog[:, 2] = image_dog[:, 2] * sqrt(2)
    return(image_dog)

def Blob_Determinant_of_Hessian(ndarray, msigma=1, xsigma=50, nusigma=10, thold=0.01, olap=0.5, lscale=False):
    invert_image = np.invert(ndarray)
    image_doh = blob_doh(invert_image, min_sigma=msigma, max_sigma=xsigma, num_sigma=nusigma, threshold=thold,
             overlap=olap, log_scale=lscale)
    return(image_doh)
#le tre funzioni sopra contengono semplicemente la chiamata all'algoritmo di corretto contenuto nella libreria skimage



#le seguenti effettuano il thresholding
def pure_thresholding(image: np.array, thresh = 0, sigma=0.05, auto=True):
    """Simple threshold algorithm: once set a value, everithing below 
       is ignored    
    """
    # need invert image for the function used
    invert_image = np.invert(image)
    #p0, p100 = np.percentile(image, (0, 100))
    #image = exposure.rescale_intensity(image, in_range=(p0, p100))
    # applying threshold
    if auto:
        thresh = invert_image.mean() + sigma
    else:
        thresh = 255 - thresh
    image = (invert_image > thresh).astype(int)*255
    #enhance 
    bw = closing(image > thresh, square(3))
    #remove artifacts connected to image border
    cleared = clear_border(bw)
    #label image regions
    label_image = label(cleared)
    #le tre liste seguenti sono usate per raccogliere le coordinate di blob
    centerx = []
    centery = []
    radius = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 15 and region.area <= 300 and region.eccentricity >= 0.25:
            #guardando il funzionamento di regionprops, ho trovato le due seguenti proprietà:
            #->  centroid ritorna una tupla contenente le coordinate del centro dell'area;
            #    tupla che ho spacchettato per avere le due coord x - y separate
            #->  equivalent_diameter che ritorna il diametro che averebbe l'area se fosse un cerchio,
            #    che ho dimezzato per avere il raggio
            centerx.append(region.centroid[0])
            centery.append(region.centroid[1])
            radius.append(region.equivalent_diameter/2)
                
    np_centerx = np.array(centerx)
    np_centery = np.array(centery)
    np_radius = np.array(radius)
    np_thresh = np.array(list(zip(np_centerx, np_centery, np_radius)))
       
    return np_thresh, thresh, sigma