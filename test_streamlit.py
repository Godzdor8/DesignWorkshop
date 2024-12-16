from main import get_result
import cv2
import numpy as np
from PIL import Image


def test_empty():
    assert True is True


def test_picture():
    img = cv2.imread('pictures/image 1.jpg')
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = get_result(img_pil)
    assert res[0][0] > 0.9