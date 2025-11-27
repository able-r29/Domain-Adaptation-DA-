import os
import pickle
import imghdr
import io
import sys

import numpy as np
from joblib import Parallel, delayed
from PIL import Image, ImageFile


def load_as_byteio(path, size=None):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    file_type = imghdr.what(path)
    if file_type is None:
        return None

    img = Image.open(path)

    if not hasattr(img, 'layers') or img.layers != 3:
        img = img.convert('RGB')

    if size:
        width, height = img.size
        if min(width, height) != size:
            if width > height:
                h = size
                w = width / height * h
            else:
                w = size
                h = height / width * w
            img = img.resize((int(w), int(h)), resample=Image.BILINEAR)
    
    bio = io.BytesIO()
    img.save(bio, format=file_type)

    return path, bio


def listup_files(dirc, dst=None):
    if dst is None:
        dst = []
    
    children    = [os.path.join(dirc, f) for f in os.listdir(dirc)]
    files       = [f for f in children if os.path.isfile(f)]
    directories = [f for f in children if os.path.isdir(f)]
    dst.extend(files)
    for f in files:
        print('append:', f)

    for d in directories:
        listup_files(d, dst)

    return dst


def load_all(files, size):
    bios = Parallel(n_jobs=4, verbose=5)([delayed(load_as_byteio)(f, size) for f in files])

    pathed_imgs = {b[0]:b[1] for b in bios if b is not None}
    print(len(pathed_imgs), 'files is saved')

    return pathed_imgs


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else input('target directory... ')
    dst  = sys.argv[2] if len(sys.argv) > 2 else input('destination path... ')
    size = sys.argv[3] if len(sys.argv) > 3 else input('image size... ')

    if size:
        size = int(size)
    else:
        size = None
    
    files = listup_files(root)
    bios  = load_all(files, size)

    with open(dst, 'wb') as f:
        pickle.dump(bios, f)

if __name__ == "__main__":
    main()
