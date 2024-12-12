from main import *


def otrisovka(results, img):
    _array = []
    for score, box, label in results:
        final_image = draw_object_bounding_box(img, box, label)
        _array.append((final_image, score, label))
    return _array