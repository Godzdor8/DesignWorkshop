from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import cv2


def draw_object_bounding_box(image_to_process, box, item):
    x, y, w, h = box
    start = (int(x), int(y))
    end = (int(x + w), int(y + h))
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    start = (int(x), int(y - 10))
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    final_image = cv2.putText(final_image, item, start, font, font_size, color, width, cv2.LINE_AA)

    return final_image


processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


def get_result(image: Image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detections.append((score, box, model.config.id2label[label.item()]))

    return detections
