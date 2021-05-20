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
    var2=np.log(var1)
    var3=[np.array(var2)]
    var4=model.predict(var3)
    var5=round(float(var4),2)

    return render_template("predict.html",pred=var5)

if __name__=="__main__":
    app.run(debug=True)
