import time
import random
import sys
import os
import re
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

if __name__ == '__main__':
    my_file = open(sys.argv[1], "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    my_file.close()
    all_images = []
    for path in data_into_list:
        img = cv2.imread(path)
        img = cv2.resize(img,(160,160))
        all_images.append(img)
    all_images = np.array(all_images)
    model = load_model("my_model.h5")
    y = model.predict(all_images)
    y_pred = np.argmax(y,axis = 1).tolist()
    print(y_pred)
    file = open('311706002.txt','w')
    for i in y_pred:
        file.write(str(i))
    file.close()
#    print(ans)