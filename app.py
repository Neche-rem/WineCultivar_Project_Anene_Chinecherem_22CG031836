from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model + scaler
model = joblib.load("model/wine_cultivar_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            alcohol = float(request.form["alcohol"])
            malic_acid = float(request.form["malic_acid"])
            ash = float(request.form["ash"])
            magnesium = float(request.form["magnesium"])
            flavanoids = float(request.form["flavanoids"])
            proline = float(request.form["proline"])

            input_data = np.array([[alcohol, malic_acid, ash,
                                     magnesium, flavanoids, proline]])

            input_scaled = scaler.transform(input_data)
            pred_class = model.predict(input_scaled)[0]

            prediction = f"Cultivar {pred_class + 1}"

        except:
            prediction = "Invalid input. Please enter valid numbers."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
