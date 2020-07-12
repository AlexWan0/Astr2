import pickle
from nltk.stem import PorterStemmer, WordNetLemmatizer
from unidecode import unidecode
import re

class MentalHealthTagger():
	def __init__(self, keywords_path='keywords_processed.pkl'):
		with open(keywords_path, 'rb') as file_in:
			self.keywords = pickle.load(file_in)
			self.stemmer = PorterStemmer()
			self.lemmatiser = WordNetLemmatizer()
			self.idx_mapping = {"Addiction":0,
								"ADHD or Attention Issues":1,
								"Anger Management":2,
								"Anxiety or Panic Attacks":3,
								"Children and Teens":4,
								"Christian or Pastoral Counseling":5,
								"Depression":6,
								"Eating and Food Issues":7,
								"Family Therapy":8,
								"Grief":9,
								"Individual Therapy":10,
								"Marriage or Couples Counseling":11,
								"Psychiatry / Medication":12,
								"Psychology":13,
								"Sex and Sexuality":14,
								"Trauma or Abuse":15}

	def preprocess(self, raw):
		result = unidecode(raw)
		result = re.sub(r"'", "", result)
		result = re.sub(r"[^a-zA-Z]", " ", result)
		result = re.sub(' +', ' ', result)
		result = result.lower()

		result = result.split(' ')
		#print('1', result)
		result = [self.stemmer.stem(w) for w in result]
		#print('2', result)
		result = [self.lemmatiser.lemmatize(w) for w in result]
		#print('3', result)
		return result

	def get_counts(self, text, sort=True):
		text_words = self.preprocess(text)

		print(text_words)

		counts = []

		for key, val in self.keywords.items():
			counter = 0
			for word in text_words:
				if word in val:
					counter += 1
			counts.append([key, counter, self.idx_mapping[key]])

		if sort:
			counts = sorted(counts, key=lambda x:-x[1])

		return counts 

if __name__ == '__main__':
	labeler = MentalHealthTagger()

	print(labeler.get_counts("distracted can't focus school homework"))