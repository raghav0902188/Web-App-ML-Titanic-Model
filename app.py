from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle
import numpy as np

model = pickle.load(open('titanic_xgb.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        #Pclass
        pclass = int(request.form["Pclass"])

        #Age
        Age = int(request.form["Age"])

        #SibSp
        SibSp = int(request.form["SibSp"])

        #Sex
        Sex = int(request.form["Sex"])

        #Parch
        Parch = int(request.form["Parch"])

        #Fare
        Fare = int(request.form["Fare"])

        #Embarked
        Embarked = int(request.form["Embarked"])

        prediction = model.predict(np.array([[pclass, Sex, Age, SibSp, Parch, Fare, Embarked]]))

        output = round(prediction[0], 4)

        return render_template('home.html',prediction_text="Passenger Survival Probability is - {}".format(output))


    return render_template("home.html")



if __name__ == "__main__":
    app.run(debug=True)
    
