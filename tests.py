from main import get_result
import cv2


def test_empty():
    assert True is True

def test_picture():
    res = get_result(cv2.imread('pictures/image 1.jpg'))
    assert res > 0.9