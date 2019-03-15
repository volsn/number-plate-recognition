import cv2
import numpy as np

cascade = cv2.CascadeClassifier('licence_plate.xml')

for photo_id in range(0, 81):

	img = cv2.imread('out_pic/output_{}.jpg'.format(photo_id))

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	number_plate = cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in number_plate:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

	
	cv2.imshow('img', img)
	k = cv2.waitKey(0)

cv2.DestroyAllWindows()
