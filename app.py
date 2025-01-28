from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


# Route for a home page
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            gender=request.form.get("gender"),
            age=float(request.form.get("age")),
            hypertension=int(request.form.get("hypertension")),
            heart_disease=int(request.form.get("heart_disease")),
            ever_married=request.form.get("ever_married"),
            work_type=request.form.get("work_type"),
            residence_type=request.form.get("residence_type"),
            avg_glucose_level=float(request.form.get("avg_glucose_level")),
            bmi=float(request.form.get("bmi")),
            smoking_status=request.form.get("smoking_status"),
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
