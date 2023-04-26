from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
import pickle
import numpy as np

app = Flask(__name__)

classifier=pickle.load(open("gnb_model.pkl","rb"))

@app.route('/')
def index():
	return render_template('one.html')

@app.route('/one_set', methods=['GET'])
def one_set():
	
	global age, EDUC
	name = request.args.get('name','')
	age = request.args.get('age','')
	EDUC = request.args.get('yoe','')

	return render_template('two.html')

@app.route('/two_set', methods=['GET'])
def two_set():
	
	global SES, MMSE, CDR
 
	SES = request.args.get('ses','')
	MMSE = request.args.get('mmse','')
	CDR = request.args.get('cdr','')

	return render_template('three.html')

@app.route('/third_set', methods=['GET'])
def third_set():
	
	gender = request.args.get('gender','')
	eTIV = request.args.get('etiv','')
	nWBV = request.args.get('nwbv','')
	ASF = request.args.get('asf','')

	tp =[gender,age,EDUC,SES,MMSE,CDR,eTIV,nWBV, ASF]
	tp = np.array(tp)
	tp = tp.reshape(1, -1)

	value = classifier.predict(tp)[0]
	print (value)

	if value == 0:
		return render_template('result0.html')
	elif value == 1:
		return render_template('result1.html')
	elif value == 2:
		return render_template('result2.html')

@app.route("/home")
def home():
	return render_template('one.html')

if __name__ == '__main__':

	run_simple('localhost', 5000, app)
	#app.run(host='localhost',port=5000, debug=True,threaded=True)