from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn import preprocessing

model = pickle.load(open("diabetes.pkl", 'rb'))

app = Flask(__name__)
app.config['SECRET_KEY'] = "!2345@abc"

@app.route("/")
def home():
	return render_template('index.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
	try:
		if request.method == "POST":
			preg = float(request.form.get('pregnancies'))
			glucose = float(request.form.get('glucose'))
			bp = float(request.form.get('bloodpressure'))
			st = float(request.form.get('skinthickness'))
			insulin = float(request.form.get('insulin'))
			bmi = float(request.form.get('bmi'))
			dpf = float(request.form.get('dpf'))
			age = float(request.form.get('age'))
			
		
			l = [preg, glucose, bp, st, insulin, bmi, dpf, age]
			l = np.asarray(l)
			l = np.reshape(l, (1,8))
			pred = model.predict(l)
			return render_template("result.html", prediction = pred[0])
	except:
		return render_template("result.html", no_value = 1)

if __name__ == '__main__':
	app.run(debug = True)
