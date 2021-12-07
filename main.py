import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage import color
from skimage.io import imsave
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file1', type=str, default='img1.jpg', help='Image 1.')
parser.add_argument('--file2', type=str, default='img2.jpg', help='Image 2.')
opts = parser.parse_args()

imageA = cv2.imread(opts.file1)
imageB = cv2.imread(opts.file2)

imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

AB = utils.mosaico([imageA, imageB])
AB = utils.autocrop(AB)

imsave('mosaic.png', AB)