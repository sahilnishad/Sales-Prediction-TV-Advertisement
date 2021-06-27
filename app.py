import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    if request.method=='POST':
        input = float(request.form['TV'])
        output = model.predict([[input]])[0]
    return render_template("index.html", result=output)

if __name__ == "__main__":
    app.run(debug=True)