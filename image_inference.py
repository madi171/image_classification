"""

Image classification prediction for vivi or madi

"""
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.models import load_model

import numpy as np
import argparse
import cv2
import os

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--target", required=True,
                    help="path to input target")
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    args = vars(ap.parse_args())

    # initialize the number of epochs to train for, initial learning rate,
    DIM = 256

    target_image_path = args["target"]
    image = cv2.imread(target_image_path)
    image = cv2.resize(image, (DIM, DIM))
    image = img_to_array(image)
    image = np.array(image, dtype="float") / 255.0

    X = [image]

    print("[INFO] loading model...")
    model = load_model(args["model"])

    if model is not None:
        ret = model.predict(X)
        print("The predict result is: %f probabilty to vivi" % ret)
    else:
        print("[ERROR] Error in load model")
        exit(1)

    print("[INFO] prediction done!")
