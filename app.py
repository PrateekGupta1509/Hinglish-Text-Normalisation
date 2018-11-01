import flask
import pickle
import pandas as pd
from decoderModel import *
import tensorflow as tf
from flask_cors import CORS

# initialize our Flask application and the Keras model variables
app = flask.Flask("Normalizer")
CORS(app)
data_info = None
encoder_model = None
decoder_model = None
graph = None

# Load Model and utilies
def load_normalizer(modelPath, dataInfoPath):
	global encoder_model
	global decoder_model
	global data_info

	# load data_info json
	with open(dataInfoPath, 'rb') as handle:
		data_info = pickle.load(handle)

	# load the pre-trained Keras model
	encoder_model, decoder_model = getDecoder(modelPath)
	print "Loaded Model!"

@app.route("/normalize", methods=["GET"])
def predict():
	with graph.as_default():
		inputText = flask.request.args.get("text")
		inputText = inputText.lower()
		df = pd.DataFrame([inputText], columns = ['input'])
		df['output'] = ''
		outputText = decodeSentence(encoder_model, decoder_model, df, data_info)
		return flask.jsonify({"output": outputText.strip('\t,\n')})

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_normalizer('model/normalization-model.h5', 'model/data-info.pkl')
	graph = tf.get_default_graph()
	app.run()