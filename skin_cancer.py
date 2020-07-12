from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tf_explain.core import GradCAM
import numpy as np
import cv2
from base64_utils import cv2_to_base64, base64_to_cv2
import gc

def return_zero(x, y):
	return 0

class SkinCancerDetector():
	def __init__(self, model_path='skin_cancer.h5'):
		self.model_path = model_path
		self.img_size = 128
		self.classes = ['Bowen\'s Disease','Basal Cell Carcinoma','Benign Keratosis-like Lesions','Dermatofibroma','Melanoma','Melanocytic Nevi','Vascular Lesions']

	def detect(self, img):
		self.model = load_model(self.model_path, custom_objects={'f1_m':return_zero})

		img_orig = img.copy()
		img = preprocess_input(img)

		pred = self.model.predict(np.expand_dims(img, axis=0))
		print(pred)

		explainer = GradCAM()
		grid = explainer.explain(([img], None), self.model, class_index=np.argmax(pred[0]), image_weight=0)

		im = cv2.cvtColor(img_orig.astype(np.uint8), cv2.COLOR_BGR2RGB)
		fin = cv2.addWeighted(grid, 0.9, im, 0.1, 0)

		del self.model
		print(gc.collect())

		return (cv2_to_base64(fin), "<p><b>Detection:</b> " + self.classes[np.argmax(pred[0])] + "</p>"), pred[0], fin