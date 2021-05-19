from flask import Flask,render_template,request
import numpy as np
import pickle
model=pickle.load(open('model_cars.pkl','rb'))
app=Flask(__name__)
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/predict",methods=['POST'])
def predict():
    var1=[float(x) for x in request.form.values()]
    var2=[np.array(var1)]
    var3=model.predict(var2)
    var4=round(float(var3),2)

    return render_template("predict.html",pred=var4)

if __name__=="__main__":
    app.run(debug=True)
