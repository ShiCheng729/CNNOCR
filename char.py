import cv2
import numpy as np
import char_detector



def add_padding(image):
	h_padding = 0.1 * image.shape[0]
	w_padding = 0.2 * image.shape[1]
	out_shape = (int(image.shape[0] + h_padding * 2),int(image.shape[1] + w_padding * 2))
	text_mask = np.ones(out_shape)*255
	iy = int(h_padding)
	ix = int(w_padding) 
	text_mask[iy:iy + image.shape[0], ix:ix + image.shape[1]] = image
	return text_mask
    
def add_shadow(image):
	h_padding = 0.1 * image.shape[0]
	w_padding = 0.2 * image.shape[1]
	out_shape = (int(image.shape[0] + h_padding * 2),int(image.shape[1] + w_padding * 2))
	text_mask = np.ones(out_shape)*0
	iy = int(h_padding)
	ix = int(w_padding) 
	text_mask[iy:iy + image.shape[0], ix:ix + image.shape[1]] = image
	return text_mask

def image_proccessing(img):
	ret,thresh  = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
	if cv2.contourArea(contours[0])>=cv2.contourArea(contours[1]):
		x,y,w,h = cv2.boundingRect(contours[1])
		#cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2)
	else :
		x,y,w,h = cv2.boundingRect(contours[0])
		#cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2)
	thresh = thresh[y:y+h, x:x+w]
	thresh = add_padding(thresh)+150
	thresh = add_shadow(thresh)
	cv2.imwrite("temp.jpg", thresh)
	thresh = char_detector.image_preproccessing(thresh)
	result,p = char_detector.predict(thresh)
	print result,p
	return result,p

