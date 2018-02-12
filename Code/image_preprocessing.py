import cv2
import numpy as np


_R_MEAN = 128
_G_MEAN = 118
_B_MEAN = 108
_RGB_MEAN = np.array([_R_MEAN, _G_MEAN, _B_MEAN]) / 255


def _resize_keep_aspect(image, short_side=256, img_size=224):
    """ Perform Resize for the input image while preserving 
    the aspect ratio. The short edge of the image is set to 
    be equal to r'self.rescale_size' and the other dimension
    is scaled accordingly to keep the aspect ratio as is.  
    """
    # Resize
    h, w, _ = image.shape
    scale = short_side / min(h, w)
    h = int(h * scale)
    w = int(w * scale)
    image = cv2.resize(image, (w, h))
    
    # Center Crop to the target size 
    hs = (h - img_size) // 2
    ws = (w - img_size) // 2
    image = image[hs:hs+img_size, ws:ws+img_size]
    return image

def preprocess_image(image, short_side=256, img_size=224, preserve_aspect=True):
    """ """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = image / 255

    if preserve_aspect is True:
        image = _resize_keep_aspect(image)
    else:
        image = cv2.resize(image, (img_size, img_size))

    image = image - _RGB_MEAN
    return np.expand_dims(image, 0)
