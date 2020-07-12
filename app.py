from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pipeline
import base64
from pipeline import modules
from screen import get_patient_tags
from mental_health_tagger import MentalHealthTagger
from textblob import TextBlob

labeler = MentalHealthTagger()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/mind')
def mind_page():
	return render_template('mind.html')

@app.route('/mind', methods=['POST'])
def mind():
	responses = []

	data = request.form
	text = data['text']

	tb = TextBlob(text)
	polarity = tb.sentiment[0]

	print(polarity)

	if polarity < 0:
		responses.append('I\'m sorry to hear that.')
	elif polarity < 0.4:
		responses.append('I understand.')
	else:
		responses.append('That\'s good to hear.')

	print(text)

	counts = labeler.get_counts(text)

	idx = -1

	url = 'https://www.goodtherapy.org/newsearch/search.html?search%5Bzipcode%5D=87501&search%5Bmiles%5D=25'

	print(counts, counts[0][1])

	if not counts[0][1] == 0:
		idx = counts[0][-1] + 1
		
		url = 'https://www.goodtherapy.org/newsearch/search.html?search%5Bzipcode%5D=87501&search%5Bmiles%5D=25&search%5Bspecialty%5D='+str(idx)+'&search%5BfromHomeRadio%5D=zipcode'

	resource_message = 'Here, I\'ve found some resources for you: <a target="_blank" href="'+url+'">link</a>'	

	responses.append(resource_message)

	return jsonify(responses)


@app.route('/insights')
def insights():
	return render_template('insights.html')

@app.route('/screen', methods=['GET'])
def screen_page():
	return render_template('screen.html')

@app.route('/screen', methods=['POST'])
def screen():
	data = request.form

	data_processed = {}

	for key, val in data.items():
		data_processed[key] = float(val)

	return jsonify({'tags': get_patient_tags(data_processed)})

@app.route('/detect', methods=['POST'])
def detect():
	print(request.files)
	file = request.files['file']
	print(file)

	print(request.form)

	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		image_string = base64.b64encode(file.read())

		mod_names = request.form['my-select'].split(',')

		mn_cleaned = []
		for mn in mod_names:
			if mn in modules:
				mn_cleaned.append(mn)
		
		detect_type = request.form['classifier']

		print(mod_names)
		print(mn_cleaned)
		print(detect_type)

		result = pipeline.run(image_string, mn_cleaned, detect_type)

		return jsonify({'success': True, 'data': result})

	return {'success': False}

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=80)