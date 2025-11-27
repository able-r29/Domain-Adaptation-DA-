import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def preprocess(path, train, size):
    img = np.array(Image.open(path)).astype(np.float32)
    h, w, _ = img.shape

    if h != size or w != size:
        x = int((np.random.random() if train else 0.5) * (w-size))
        y = int((np.random.random() if train else 0.5) * (h-size))
        img = img[y:y+size, x:x+size, :]
    
    if train:
        if np.random.random() < 0.5:
            img = img[:,::-1,:]
        if np.random.random() < 0.5:
            img = img[::-1,:,:]
        if np.random.random() < 0.5:
            img = np.transpose(img, axes=(1, 0, 2))

    img /= 255
    img  = img.transpose((2, 0, 1))
    img  = img.copy()
    return img

