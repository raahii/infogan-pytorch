import cv2
from PIL import Image


def read_img(path):
    return cv2.imread(str(path))[:, :, ::-1]


def write_img(img, path):
    Image.fromarray(img).save(str(path))
