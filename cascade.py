import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
import sys


class NoInputDataError(Exception):
	def __init__(self, message, errors):
		super().__init__(message)
		self.errors = errors


def get_plate_num_images(files, show=False):

	cascade = cv2.CascadeClassifier('licence_plate.xml')

	if not exists('result'):
		makedirs('result')

	result_file_name = 0
	for file in files:

		img = cv2.imread(join('input', file))

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		number_plate = cascade.detectMultiScale(gray, 1.3, 5)

		for (x,y,w,h) in number_plate:
			croped_licence_plate = img[y:y+h, x:x+w]
			cv2.imwrite(join('result', '{}.png'.format(result_file_name)), croped_licence_plate)
			result_file_name += 1
			if show:
				cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

		if show:
			cv2.imshow('img', img)
			k = cv2.waitKey(0)
			
	if show:
		cv2.DestroyAllWindows()


if __name__ == '__main__':

	if not exists('input'):
		raise NoInputDataError('Data to process should be stored in `input` folder')

	files = [f for f in listdir('input') if isfile(join('input', f))]
	
	if 'show' in sys.argv:
		print('Files to be processed: \n', files)
		get_plate_num_images(files, show=True)

	get_plate_num_images(files)