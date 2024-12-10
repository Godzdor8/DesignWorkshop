import urllib
import numpy as np
from main import *


def otrisovka(results):
    _array = []
    for score, box, label in results:
        req = urllib.request.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        final_image = draw_object_bounding_box(img, box, label)
        _array.append((final_image, score, label))
    return _array
    '''cv2.imshow("Object Detection", final_image)
        cv2.waitKey(0)  # Ожидание любой клавиши
        cv2.destroyAllWindows()'''