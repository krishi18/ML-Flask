from flask import Flask,request,render_template,redirect,url_for,session
from database_it import sqllogin
from database_it import sqlsignup 

app=Flask(__name__)
@app.route("/")
def main():
    return render_template("lkjh.html")

@app.route('/', methods =["GET", "POST"])
def home():
    return render_template("home.html")

@app.route('/login', methods=["GET","POST"])
def login():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       email = request.form.get("email")
       # getting input with name = lname in HTML form
       password = request.form.get("password")
       print(email,password)
       if (sqllogin(email,password)=='found'):
           model()
       elif sqllogin(email,password)=='nouser':
           return redirect(url_for("invaliduser"))         
       elif sqllogin(email,password)=='wrongpass':
           return redirect(url_for("incorrect"))
           
       #return redirect(url_for("home"))
        
@app.route('/signup', methods=["GET","POST"])
def signup():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       username = request.form.get("username")
       email = request.form.get("email")
       # getting input with name = lname in HTML form
       password = request.form.get("password")
       print(username,email,password)
       sqlsignup(username,email,password)
           #return redirect(url_for("model"))
       return redirect(url_for("home"))


@app.route("/invalid",methods=['GET'])
def invaliduser():
    return "<h1>User doesnt exist.</h1>" 


@app.route("/incorrect",methods=['GET'])
def invalid():
    return "Incorrect password."

import joblib  
import numpy as np  
model = joblib.load('mltmodel.pkl')   
@app.route("/predict", methods=["POST"])  
def predict():  
   age = request.form["age"]  
   sex = request.form["sex"]  
   trestbps = request.form["trestbps"]  
   chol = request.form["chol"]  
   oldpeak = request.form["oldpeak"]  
   thalach = request.form["thalach"]  
   fbs = request.form["fbs"]  
   exang = request.form["exang"]  
   slope = request.form["slope"]  
   cp = request.form["cp"]  
   thal = request.form["thal"]  
   ca = request.form["ca"]  
   restecg = request.form["restecg"]  
   arr = np.array([[age, sex, cp, trestbps,  
            chol, fbs, restecg, thalach,  
            exang, oldpeak, slope, ca,  
            thal]])  
   pred = model.predict(arr)  
   if pred == 0:  
     res_val = "NO HEART PROBLEM"  
   else:  
     res_val = "HEART PROBLEM"  
   return render_template('home.html', prediction_text='PATIENT HAS {}'.format(res_val))  

if __name__=='__main_':
    app.run(port="5000" , host="localhost")