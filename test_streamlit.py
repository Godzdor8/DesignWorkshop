from main import get_result
import cv2


def test_empty():
    assert True is True

def test_picture():
    img = cv2.imread('pictures/image 1.jpg')
    res = get_result(img)
    assert res[0][0] > 0.9