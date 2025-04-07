from ZEMO import zemo
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def calculate_zernike_moments(image, image_size, zernike_order):
    """
    Calculate Zernike moments for a given image.
    
    Args:
        image: PIL Image object
        image_size: Size to resize the image to
        zernike_order: Order of Zernike moments to calculate
        
    Returns:
        df: DataFrame containing Zernike moments
        ZBFSTR: Zernike basis functions
        Z: Zernike moments array
    """
    zernike_moments = []

    image_array = np.array(image)
    
    resized_image = cv2.resize(image_array, (image_size, image_size))
    
    if len(resized_image.shape) == 3:
        im = resized_image[:, :, 0]
    else:
        im = resized_image
    
    ZBFSTR = zemo.zernike_bf(image_size, zernike_order, 1)
    
    Z = zemo.zernike_mom(np.double(im), ZBFSTR)
    zernike_moments.append(Z)

    df = pd.DataFrame(zernike_moments)
    
    return df, ZBFSTR, Z

def reconstruct_image(Z, image_size, ZBFSTR, show=True):
    """
    Reconstruct image from Zernike moments.
    
    Args:
        Z: Zernike moments array
        image_size: Size of the reconstructed image
        ZBFSTR: Zernike basis functions
        show: Whether to display the image using matplotlib
        
    Returns:
        reconstructed: Reconstructed image array
    """

    reconstructed = zemo.zernike_rec(Z, image_size, ZBFSTR)
    
    # reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())
    
    if show:
        plt.imshow(reconstructed, cmap='bone', interpolation='nearest')
        plt.title("Reconstructed Image")
        plt.axis('off')
        plt.show()
    
    return reconstructed