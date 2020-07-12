from joblib import dump, load
import numpy as np

def calculate_bmi(height, weight):
	return (weight)/(height**2)

class BreastCancerModel():
	def __init__(self, model_path='bc.pkl'):
		self.model = load(model_path)
	def predict(self, data):
		if data['sex'] == 1:
			return 0

		bmi = calculate_bmi(data['height']/100, data['weight'])
		model_input = [
			data['age'],
			bmi,
			data['ogtt'],
			data['insu']
		]
		print(model_input)

		return self.model.predict(np.expand_dims(model_input, axis=0))

class CardiovascularDiseaseModel():
	def __init__(self, model_path='card.pkl'):
		self.model = load(model_path)
	def predict(self, data):
		chol = 3
		if data['chol'] < 239:
			chol = 2
		elif data['chol'] < 200:
			chol = 1

		ogtt = 3
		if data['ogtt'] < 126:
			ogtt = 2
		elif data['ogtt'] < 110:
			ogtt = 1

		model_input = [
			data['age'] * 365,
			data['sex'] + 1,
			data['height'],
			data['weight'],
			data['sbp'],
			data['dbp'],
			chol,
			ogtt,
			data['smoke'],
			data['alc'],
			data['phys']
		]
		print(model_input)

		return self.model.predict(np.expand_dims(model_input, axis=0))

class DiabetesModel():
	def __init__(self, model_path='diab.pkl'):
		self.model = load(model_path)
	def predict(self, data):
		if data['sex'] == 1:
			return 0

		bmi = calculate_bmi(data['height']/100, data['weight'])
		model_input = [
			data['preg'],
			data['ogtt'],
			data['dbp'],
			data['insu'],
			bmi,
			data['age']
		]

		print(model_input)

		return self.model.predict(np.expand_dims(model_input, axis=0))

class HeartDiseaseModel():
	def __init__(self, model_path='hd.pkl'):
		self.model = load(model_path)
	def predict(self, data):
		ogtt = 1
		if data['ogtt'] <= 120:
			ogtt = 0

		model_input = [
			data['age'],
			data['sex'],
			(data['sbp'] + data['dbp'])/2.0,
			data['chol'],
			ogtt

		]
		print(model_input)

		return self.model.predict(np.expand_dims(model_input, axis=0))

def get_patient_tags(data):
	models = [BreastCancerModel(), CardiovascularDiseaseModel(), DiabetesModel(), HeartDiseaseModel()]
	tags = ['Breast Cancer', 'Cardiovascular Disease', 'Diabetes', 'Heart Disease']

	patient_tags = []

	for t, m in zip(tags, models):
		if(m.predict(data) > 0):
			patient_tags.append(t)

	return patient_tags

if __name__ == '__main__':
	zero_data = {"age":0,"sex":0,"sbp":0,"dbp":0,"chol":0,"ogtt":0,"smoke":0,"alc":0,"phys":0,"preg":0,"bmi":0,"insu":0,"height":1,"weight":1}

	print(get_patient_tags(zero_data))