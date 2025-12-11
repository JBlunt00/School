from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_house_price_model.pkl")

# Load model at startup
model = joblib.load(MODEL_PATH)

# Features used in the model
FEATURE_COLUMNS = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "Neighborhood"
]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Collect form data
            overall_qual = float(request.form.get("OverallQual"))
            gr_liv_area = float(request.form.get("GrLivArea"))
            garage_cars = float(request.form.get("GarageCars"))
            total_bsmt_sf = float(request.form.get("TotalBsmtSF"))
            full_bath = float(request.form.get("FullBath"))
            year_built = float(request.form.get("YearBuilt"))
            neighborhood = request.form.get("Neighborhood")

            data = {
                "OverallQual": [overall_qual],
                "GrLivArea": [gr_liv_area],
                "GarageCars": [garage_cars],
                "TotalBsmtSF": [total_bsmt_sf],
                "FullBath": [full_bath],
                "YearBuilt": [year_built],
                "Neighborhood": [neighborhood]
            }

            input_df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
            pred = model.predict(input_df)[0]

            prediction = round(float(pred), 2)
        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    RESTful API endpoint.
    Expects JSON like:
    {
        "OverallQual": 7,
        "GrLivArea": 1500,
        "GarageCars": 2,
        "TotalBsmtSF": 800,
        "FullBath": 2,
        "YearBuilt": 2005,
        "Neighborhood": "CollgCr"
    }
    """
    try:
        input_json = request.get_json()

        # Create DataFrame with one row
        input_df = pd.DataFrame([input_json], columns=FEATURE_COLUMNS)

        pred = model.predict(input_df)[0]
        return jsonify({"prediction": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)