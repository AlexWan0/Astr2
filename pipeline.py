import cv2
import base64
import io
from imageio import imread
from ISR.models import RDN
import numpy as np
from PIL import Image
import models
from base64_utils import cv2_to_base64, base64_to_cv2
import tensorflow as tf

class VAEAnomaly():
	def __init__(self, model_path='model.tflite'):
		self.model_path = model_path

	def vlb_num(self, img, img_ae, mu, logvar, kl_weight=1):
		kl = 0.5 * np.sum((np.exp(logvar) + np.square(mu) - 1.0 - logvar), axis=-1)
		mse = np.square(img - img_ae)
		mse = np.sum(mse, axis=(0, 1, 2))
		return kl, mse, kl_weight * kl + mse

	def __call__(self, img):
		img_orig = img.copy()
		
		img = cv2.resize(img, (128, 128))
		img = img/255.0

		interpreter = tf.lite.Interpreter(model_path=self.model_path)
		interpreter.allocate_tensors()

		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		
		interpreter.set_tensor(input_details[0]['index'], np.float32(np.expand_dims(img, axis=0)))

		interpreter.invoke()

		mean = interpreter.get_tensor(output_details[0]['index'])
		logvar = interpreter.get_tensor(output_details[1]['index'])
		ae = interpreter.get_tensor(output_details[2]['index'])

		vlb_res = self.vlb_num(img, ae[0], mean[0], logvar[0])
		
		if vlb_res[-1] > 1200:
			return img_orig, "<p><b>VAE Anomaly Detection</b> Likely out of domain</p>"
		
		if vlb_res[-1] > 1000:
			return img_orig, "<p><b>VAE Anomaly Detection</b> Possibly out of domain</p>"
		
		del interpreter

		return img_orig, "<p><b>VAE Anomaly Detection</b> Likely in domain</p>"

class SwapChannels():
	def __call__(self, img):
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "<p><b>Swap Color Channels</b></p>"

class DeNoise():
	def __call__(self, img):
		im = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
		return im, "<p><b>Basic Denoising</b></p>"

class RDNDeNoise():
	def __init__(self, weights='noise-cancel'):
		self.weights = weights

	def __call__(self, img):
		self.model = RDN(weights=self.weights)
		pred = self.model.predict(img)

		del self.model

		return pred, "<p><b>RDN Denoise</b></p>"

class Superresolution():
	def __init__(self, weights='psnr-small'):
		self.weights = weights

	def __call__(self, img):
		self.model = RDN(weights=self.weights)
		pred = self.model.predict(img)

		del self.model

		return pred, "<p><b>Superresolution</b> "+str(img.shape) + " -> " + str(pred.shape)+"</p>"

class Resize():
	def __init__(self, size):
		self.size = size
	def __call__(self, img):
		rs = cv2.resize(img, (self.size, self.size))
		return rs, "<p><b>Resize</b> "+str(img.shape) + " -> " + str(rs.shape)+"</p>"


modules = {'dn': DeNoise(), 'sr': Superresolution(), 'scc': SwapChannels(), 'rs128':Resize(128), 'vae':VAEAnomaly(), 'rdndn': RDNDeNoise()}


def run(img_data, mod_names, detect_type):
	img = base64_to_cv2(img_data)

	result = [(img_data.decode(), "<p><b>Original Image</b></p>")]

	for mn in mod_names:
		module = modules[mn]

		img, data = module(img)

		result.append((cv2_to_base64(img), data))

	pipe_new, pred, fin = models.detect(result[-1][0], detect_type)

	result.append(pipe_new)

	print(pipe_new)

	return {'pipeline': result, 'detection':(pred,fin)}

if __name__ == '__main__':
	with open('test.txt', 'r') as file:
		img_data = file.read()

	print(run(str.encode(img_data), ['dn', 'sr'])[-1])