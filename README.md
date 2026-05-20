# House Price Predictor

This project is a Flask-based web application that predicts house prices using an ensemble machine learning model trained on the House Prices dataset.

The application includes:

- A trained ensemble `.pkl` model
- A Flask backend
- A responsive HTML/CSS frontend
- Automatic handling of:
  - Numerical features
  - Categorical dropdowns
  - One-hot encoded model features
  - Robust scaling
  - Ensemble prediction blending

---

# Project Structure

The folder structure MUST remain exactly like this for the application to work correctly:

```text
house_price_predictor/
│
└── predictor/
    │
    ├── app.py
    ├── best_model.pkl
    ├── test.csv
    │
    ├── templates/
    │   └── index.html
    │
    └── static/
        └── style.css
