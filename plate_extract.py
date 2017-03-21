
import numpy as np
import cv2



def add_padding(image):
	h_padding = 0.1 * image.shape[0]
	w_padding = 0.3 * image.shape[1]
	out_shape = (int(image.shape[0] + h_padding * 2),int(image.shape[1] + w_padding * 2))
	text_mask = np.ones(out_shape)*255
	iy = int(h_padding)
	ix = int(w_padding) 
	text_mask[iy:iy + image.shape[0], ix:ix + image.shape[1]] = image
	return text_mask
	#text_mask[iy:iy + image.shape[1], ix:ix + image.shape[0]] = image
	#return text_mask
def add_shadow(image):
	h_padding = 0.1 * image.shape[0]
	w_padding = 0.3 * image.shape[1]
	out_shape = (int(image.shape[0] + h_padding * 2),int(image.shape[1] + w_padding * 2))
	text_mask = np.ones(out_shape)*0
	iy = int(h_padding)
	ix = int(w_padding) 
	text_mask[iy:iy + image.shape[0], ix:ix + image.shape[1]] = image
	return text_mask

face_cascade = cv2.CascadeClassifier('plate.xml')

img = cv2.imread('12.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plate = face_cascade.detectMultiScale(gray, 1.1, 4)
num_of_plate = len(plate)
print num_of_plate
alist=[]
i=0
if num_of_plate > 0:
	for p in plate:
		x = p[0]
		y = p[1]
		w = p[2]
		h = p[3]
		temp = gray[y:y+h, x:x+w]
		#img2 = img[x:x+w, y:y+h]
		size = w*h;
		ret,thresh  = cv2.threshold(temp,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		if len(contours) >20:
			num_of_contours = 0
			for con in contours:
				area = cv2.contourArea(con)
				x,y,w,h = cv2.boundingRect(con)
				if h*w >size*0.005 and h*w < size*0.1 and h>1.5*w:
					num_of_contours = num_of_contours+1
			alist.append([num_of_contours,i])

		i=i+1
else:
	print 'no plate'


cmax =  (max(a for (a,b) in alist))
index = 0
for (a,b) in alist:
	if a == cmax:
		index = b
		break

p = plate[index]
x = p[0]
y = p[1]
w = p[2]
h = p[3]
temp = gray[y:y+h, x:x+w]
size = w*h;
ret,thresh  = cv2.threshold(temp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
box = []
for m in contours:
	area = cv2.contourArea(m)
	x,y,w,h = cv2.boundingRect(m)
	sbox = thresh[y:y+h, x:x+w]
	if h*w >size*0.005 and h*w < size*0.1 and h>1.5*w:
			box.append([np.mean(sbox),x,y,w,h])


b = sorted(box,key=lambda colm: colm[0],reverse=True)
box = b[4]
x = box[1]
y = box[2]
w = box[3]
h = box[4]
temp1 = temp[y:y+h, x:x+w]
ret,temp1 = cv2.threshold(temp1,160,255,cv2.THRESH_BINARY)
temp1 = add_padding(temp1)
temp1 = temp1
temp1 = add_shadow(temp1)
cv2.imwrite("plate.jpg", temp1)
