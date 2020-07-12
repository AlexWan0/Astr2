import cv2
import base64
import io
from imageio import imread

def cv2_to_base64(img):
	retval, buffer = cv2.imencode('.jpg', img)
	jpg_as_text = base64.b64encode(buffer)
	return jpg_as_text.decode()

def base64_to_cv2(img_data):
	if not isinstance(img_data, str):
		img_data = img_data.decode()

	img = imread(io.BytesIO(base64.b64decode(img_data)))

	return img