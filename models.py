from skin_cancer import SkinCancerDetector
import cv2
import base64
import io
from imageio import imread
from base64_utils import cv2_to_base64, base64_to_cv2

detector_mapping = {'skin_cancer': SkinCancerDetector()}

def detect(img_data, detect_type):
	detector = detector_mapping[detect_type]
	img_size = detector.img_size

	print(img_size)

	img = base64_to_cv2(img_data)

	img = cv2.resize(img, (img_size, img_size))

	if(img.shape[-1] == 4):
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

	pipe_new, pred, disp = detector.detect(img)

	jpg_as_text = cv2_to_base64(disp)

	print(pred)
	print(jpg_as_text)

	return pipe_new, pred.astype(float).tolist(), jpg_as_text

if __name__ == '__main__':
	with open('test.txt', 'r') as file:
		img_data = file.read()
	detect(str.encode(img_data), 'skin_cancer')