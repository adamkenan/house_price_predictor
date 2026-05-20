from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

# =====================================================
# FLASK SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

# =====================================================
# LOAD MODEL
# =====================================================

with open(
    r'C:\adam\AMIT_Diploma\house-prices-advanced-regression-techniques\house_price_predictor\predictor\best_model.pkl',
    'rb'
) as f:
    bundle = pickle.load(f)

models = bundle['models']
scaler = bundle['scaler']
blend_models = bundle['blend_models']
blend_weights = bundle['blend_weights']

feature_cols_full = bundle['feature_cols_full']
feature_cols_robust = bundle['feature_cols_robust']

# =====================================================
# LOAD CSV
# =====================================================

csv_path = r'C:\adam\AMIT_Diploma\house-prices-advanced-regression-techniques\house_price_predictor\predictor\test.csv'

raw_df = pd.read_csv(csv_path)

# =====================================================
# REMOVE UNUSED COLUMNS
# =====================================================

if 'Id' in raw_df.columns:
    raw_df = raw_df.drop(columns=['Id'])

# =====================================================
# DETECT FEATURE TYPES
# =====================================================

categorical_features = []
numeric_features = []

for col in raw_df.columns:

    if raw_df[col].dtype == 'object':
        categorical_features.append(col)
    else:
        numeric_features.append(col)

# =====================================================
# BUILD DROPDOWN OPTIONS
# =====================================================

categorical_options = {}

for col in categorical_features:

    values = (
        raw_df[col]
        .fillna('NA')
        .astype(str)
        .unique()
        .tolist()
    )

    # Add NA manually if missing
    if 'NA' not in values:
        values.append('NA')

    values.sort()

    categorical_options[col] = values

# =====================================================
# HOME PAGE
# =====================================================

@app.route('/')
def home():
    return render_template(
        'index.html',
        categorical_options=categorical_options,
        numeric_features=numeric_features
    )

# =====================================================
# PREDICT
# =====================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        # =================================================
        # START WITH ALL MODEL FEATURES = 0
        # =================================================

        user_input = {
            col: 0 for col in feature_cols_full
        }

        # =================================================
        # NUMERIC INPUTS
        # =================================================

        for feature in numeric_features:

            if feature not in request.form:
                continue

            value = request.form.get(feature)

            if value is None or value == '':
                value = 0
            if value is None or value == '':
                value = 0

            user_input[feature] = float(value)

        # =================================================
        # CATEGORICAL INPUTS
        # =================================================

        for feature in categorical_features:

            selected = request.form.get(feature)

            if selected is None:
                continue

            # Skip NA selections
            if selected == 'NA':
                continue

            dummy_col = f'{feature}_{selected}'

            if dummy_col in user_input:
                user_input[dummy_col] = 1

        # =================================================
        # DATAFRAME
        # =================================================

        input_df_full = pd.DataFrame([user_input])

        input_df_full = input_df_full.reindex(
            columns=feature_cols_full,
            fill_value=0
        )

        # =================================================
        # ROBUST FEATURES
        # =================================================

        input_df_robust = input_df_full.reindex(
            columns=feature_cols_robust,
            fill_value=0
        )

        robust_scaled = scaler.transform(
            input_df_robust
        )

        robust_scaled_df = pd.DataFrame(
            robust_scaled,
            columns=feature_cols_robust
        )


        # =================================================
        # MODEL PREDICTIONS
        # =================================================

        predictions = {}

        for name, info in models.items():

            track = info['track']
            model = info['model']

            if track == 'Robust':

                pred = model.predict(
                    robust_scaled_df
                )[0]

            else:

                pred = model.predict(
                    input_df_full
                )[0]

            predictions[name] = pred

        # =================================================
        # ENSEMBLE BLEND
        # =================================================

        final_log_price = 0

        for model_name, weight in zip(
            blend_models,
            blend_weights
        ):

            final_log_price += (
                predictions[model_name] * float(weight)
            )

        final_price = np.expm1(final_log_price)

        mean_std = bundle.get('y_train_std', 0.15)

        lower = np.expm1(
            final_log_price - mean_std * 0.25
        )

        upper = np.expm1(
            final_log_price + mean_std * 0.25
        )

        return render_template(
            'index.html',
            prediction=f'${final_price:,.0f}',
            lower=f'${lower:,.0f}',
            upper=f'${upper:,.0f}',
            categorical_options=categorical_options,
            numeric_features=numeric_features
        )

    except Exception as e:

        return render_template(
            'index.html',
            error=str(e),
            categorical_options=categorical_options,
            numeric_features=numeric_features
        )

# =====================================================
# RUN APP
# =====================================================

if __name__ == '__main__':
    app.run(debug=True)