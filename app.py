
from flask import Flask, render_template, request
import os
import numpy as np

from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

# Feature order must match the model's training features.
FEATURES = [
    ("fixed_acidity", "Fixed Acidity", 7.4, 0, None),
    ("volatile_acidity", "Volatile Acidity", 0.70, 0, None),
    ("citric_acid", "Citric Acid", 0.00, 0, None),
    ("residual_sugar", "Residual Sugar", 2.6, 0, None),
    ("chlorides", "Chlorides", 0.08, 0, None),
    ("free_sulfur_dioxide", "Free Sulfur Dioxide", 15.0, 0, None),
    ("total_sulfur_dioxide", "Total Sulfur Dioxide", 46.0, 0, None),
    ("density", "Density", 0.997, 0, None),
    ("pH", "pH", 3.3, 0, 14),
    ("sulphates", "Sulphates", 0.6, 0, None),
    ("alcohol", "Alcohol", 9.4, 0, None),
]


def quality_label(score):
    if score <= 4:
        return "Below Average"
    elif score <= 6:
        return "Average"
    return "Good"


def render_home(values=None, error=None, prediction=None, status=200):
    return render_template(
        "index.html",
        features=FEATURES,
        values=values or {},
        error=error,
        prediction=prediction,
    ), status


@app.route("/", methods=["GET"])
def homePage():
    return render_home()


@app.route("/train", methods=["GET"])
def training():
    exit_code = os.system("python main.py")

    if exit_code != 0:
        return "Training failed. Check the terminal logs.", 500

    return "Training completed successfully!"


@app.route("/predict", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_home()

    values = {}
    errors = []

    # Validate all 11 form fields.
    for key, label, default, minimum, maximum in FEATURES:
        raw = request.form.get(key, "").strip()

        if not raw:
            errors.append(f"{label} is required.")
            continue

        try:
            number = float(raw)

            if not np.isfinite(number):
                raise ValueError

            if number < minimum:
                errors.append(
                    f"{label} must be at least {minimum}."
                )

            if maximum is not None and number > maximum:
                errors.append(
                    f"{label} must not exceed {maximum}."
                )

            values[key] = number

        except ValueError:
            errors.append(f"{label} must be a valid number.")

    if errors:
        return render_home(
            values=request.form,
            error=" ".join(errors),
            status=400,
        )

    try:
        # Preserve the exact feature order used by the model.
        data = np.array(
            [[values[key] for key, *_ in FEATURES]],
            dtype=float,
        )

        pipeline = PredictionPipeline()
        result = pipeline.predict(data)

        # Extract one regression prediction.
        score = float(np.asarray(result).reshape(-1)[0])

        if not np.isfinite(score):
            raise ValueError("The model returned a non-finite prediction.")

        prediction = {
            "score": round(score, 2),
            "label": quality_label(score),
            "note": (
                "This is a model estimate, not a laboratory measurement."
            ),
        }

        return render_home(
            values=values,
            prediction=prediction,
        )

    except Exception:
        app.logger.exception("Wine quality prediction failed")

        return render_home(
            values=values,
            error=(
                "Prediction failed. Check your model artifacts, "
                "feature order, and PredictionPipeline configuration."
            ),
            status=500,
        )


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
