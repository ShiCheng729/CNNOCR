
import cv2
from char_detector import char_detector

detector = char_detector()

img = cv2.imread('plate.jpg')

image = detector.image_preproccessing(img)
result = detector.predict(image)
print result