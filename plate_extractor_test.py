import cv2
import numpy as np
import char_detector
from PIL import Image
import time
import sys
from pdb import set_trace as bp

from plate_char_extract import plate_extractor

if __name__ == "__main__":
	detector = plate_extractor()
	img = cv2.imread(sys.argv[1])
	image = Image.open(sys.argv[1])
	image.show()
	bp()
	box,temp = detector.plate(img)
	cv2.imwrite("plate.jpg", temp)
	image = Image.open('plate.jpg')
	image.show()
	bp()

	for r in box:
		cv2.imwrite("temp1.jpg", r)
		image = Image.open('temp1.jpg')
		image.show()
		m = char_detector.image_preproccessing(r)
		result,p = char_detector.predict(m)
		print result,p
		time.sleep(1)