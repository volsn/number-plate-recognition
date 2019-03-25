import cv2
import numpy as np
from os import listdir, makedirs, system
from os.path import isfile, join, exists, realpath
import sys
import pytesseract
import re


# Standarts of number plates for different countries
PLATE_NUMBER_STANDARTS = {
	'RU': {
		'dig': 3,
		'chr': 3,
	},
}


# Error that occures when there is no input data 
class NoInputDataError(Exception):
	def __init__(self, message, errors):
		super().__init__(message)
		self.errors = errors


def check_is_plate_valid(plate_number, country='RU'):
	"""
	Checks whether the given string may be a car plate. Russian numbers by default
	:plate_number type: srting
	:country type: string
	:rtype: bool
	"""
	
	digits = ''.join([c for c in plate_number if c.isdigit()])
	letters =  ''.join([c for c in plate_number if c.isalpha()])

	if len(digits) >= PLATE_NUMBER_STANDARTS[country]['dig'] and \
				len(letters) >= PLATE_NUMBER_STANDARTS[country]['chr']:
		return True
	return False


def find_plate_number(img_name, validation=False):
	"""
	Tries to find text on a car plate
	:img_name type: string
	:validation type: bool
	:rtype: string
	"""

	path = join('result', img_name)
	img = cv2.imread(path)
	im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

	(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	thresh = 127
	im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
	edges = cv2.Canny(im_bw, 50, 150, apertureSize = 3)

	config = ('-l rus --oem 1 --psm 6')
	text = pytesseract.image_to_string(img, config=config)

	if validation:
		if check_is_plate_valid(text):
			return text
		else:
			return 'Not a car plate'
	return text


def split_video_into_frames(file):
	"""
	Splits the given video into frames
	:file type: string
	:rtype: True
	"""


	if re.match(r'^[a-zA-Z]:\\[\\\S|*\S]?.*$', file):
		system('Xcopy {a_file_to_copy} {path_of_input_folder}'.format(
			a_file_to_copy=file,
			path_of_input_folder=realpath(__file__)
		))
		file = file.split('\\')[-1]

	comand = 'ffmpeg -i {} input/img%04d.png'.format(join('input', file))
	system(comand)
	files = [f for f in listdir('input') if isfile(join('input', f)) and f != file]
	get_plate_num_images(files)
	

def get_plate_num_images(files, show=False, validation=False, path='input', lite=False):
	"""
	Finds all the potential car plates among given photos
	:files type: list
	:show type: bool
	:validation type: bool
	:rtype: True
	"""

	cascade = cv2.CascadeClassifier('licence_plate.xml')

	if not exists('result'):
		makedirs('result')

	result_file_name = 0
	for file in files:

		if file.endswith('.mp4') or file.endswith('.avi') or file.endswith('.mpeg'):
			split_video_into_frames(file)
			return True

		img = cv2.imread(join('input', file))

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		number_plate = cascade.detectMultiScale(gray, 1.3, 5)

		for (x,y,w,h) in number_plate:
			croped_licence_plate = img[y:y+h, x:x+w]
			cv2.imwrite(join('result', '{}.png'.format(result_file_name)), croped_licence_plate)
			print(find_plate_number('{}.png'.format(result_file_name), validation=validation))
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

	
	show = False
	if 'show' in sys.argv:
		show = True
		print('Files to be processed: \n', files)
	validation = False
	if 'validate' in sys.argv:
		validation = True

	print(len(sys.argv))
	if len(sys.argv) > 1:
		if re.match(r'^[a-zA-Z]:\\[\\\S|*\S]?.*$', sys.argv[1]):
			get_plate_num_images(sys.argv[1])

	get_plate_num_images(files, show=show, validation=validation)
