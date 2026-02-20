# ============================================================
# CAR PRICE PREDICTOR - MULTI MODEL LINEAR REGRESSION
# ============================================================
# This script:
# 1. Loads car dataset
# 2. Cleans & preprocesses data
# 3. Finds Top 5 most posted car models
# 4. Trains a separate Linear Regression model for each model
# 5. Uses Gradient Descent for optimization
# 6. Stores trained models
# 7. Predicts price for new cars
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD DATA
# ============================================================
# Read CSV file into pandas dataframe
data2023 = pd.read_csv('2023data.csv')

# ============================================================
# 2. DATA CLEANING & PREPROCESSING
# ============================================================

# Drop columns that are irrelevant to price prediction
data2023.drop(
    ['addref','color','city','registered','fuel','make','body'],
    axis=1,
    errors='ignore',  # ignore if column doesn't exist
    inplace=True
)

# Convert transmission from text to numeric
# Automatic -> 1, Manual -> 0
data2023['transmission'] = data2023['transmission'].map({
    'Automatic': 1,
    'Manual': 0
})

# Handle assembly column
# Fill missing assembly as "local"
data2023['assembly'].fillna("local", inplace=True)

# Encode assembly
# Imported -> 1, Local -> 0
data2023['assembly'] = data2023['assembly'].map({
    'Imported': 1,
    'local': 0
})

# Remove rows with missing critical values
data2023.dropna(
    subset=['price','mileage','year','engine','model'],
    inplace=True
)

# ============================================================
# 3. IDENTIFY TOP 5 MOST POSTED CAR MODELS
# ============================================================
top_models = data2023['model'].value_counts().head(5).index.tolist()
print("Top 5 Models Found:", top_models)

# ============================================================
# 4. LINEAR REGRESSION UTILITIES
# ============================================================

def compute_cost(X, y, w, b):
    """
    Calculates Mean Squared Error cost.
    X -> Features
    y -> Target prices
    w -> Weights
    b -> Bias
    """
    m = len(y)  # number of samples
    predictions = X.dot(w) + b
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost


def gradient_descent(X, y, w, b, learning_rate, epochs):
    """
    Performs Gradient Descent optimization.
    Adjusts w and b iteratively to reduce cost.
    """
    m = len(y)

    for i in range(epochs):

        # Forward pass - predictions
        predictions = X.dot(w) + b

        # Compute gradients
        dw = (1/m) * X.T.dot(predictions - y)
        db = (1/m) * np.sum(predictions - y)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        # Print cost occasionally
        if i % 200 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Epoch {i} | Cost: {cost}")

    return w, b


# ============================================================
# 5. TRAINING FUNCTION FOR A SINGLE CAR MODEL
# ============================================================
def train_model(df):
    """
    Trains linear regression model for a given car dataframe.
    Returns weights, bias, and normalization stats.
    """

    # Selected features affecting price
    feature_cols = ['assembly','year','engine','transmission','mileage']

    X = df[feature_cols].values
    y = df['price'].values.reshape(-1,1)

    # -------- NORMALIZE FEATURES --------
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    # Avoid division by zero
    X_std[X_std == 0] = 1

    X = (X - X_mean) / X_std

    # -------- NORMALIZE TARGET --------
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1

    y = (y - y_mean) / y_std

    # -------- INITIALIZE PARAMETERS --------
    m, n = X.shape
    w = np.zeros((n,1))
    b = 0

    # -------- TRAIN USING GRADIENT DESCENT --------
    w, b = gradient_descent(
        X, y,
        w, b,
        learning_rate=0.01,
        epochs=1500
    )

    # -------- RETURN MODEL --------
    return {
        "w": w,
        "b": b,
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std
    }


# ============================================================
# 6. TRAIN MODELS FOR TOP 5 CARS
# ============================================================
models = {}  # dictionary to store all models

for car in top_models:
    print(f"\nTraining model for: {car}")

    # Filter dataset for this specific car
    df_car = data2023[data2023['model'] == car].copy()

    # Train and store model
    models[car] = train_model(df_car)

print("\nAll models successfully trained!")

# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================
def predict_price(car_model, assembly, year, engine, transmission, mileage):
    """
    Predicts price for a given car AND returns actual price if found.
    """

    if car_model not in models:
        return "Model not trained yet."

    model = models[car_model]

    # Create feature array
    x = np.array([assembly, year, engine, transmission, mileage])

    # Normalize features
    x_norm = (x - model["X_mean"]) / model["X_std"]

    # Predict normalized price
    pred_norm = x_norm.dot(model["w"]) + model["b"]

    # Convert back to real price
    predicted_price = float((pred_norm * model["y_std"] + model["y_mean"]).item())

    # -------------------------------------------------------
    # FIND ACTUAL PRICE FROM DATASET (if exists)
    # -------------------------------------------------------
    actual_row = data2023[
        (data2023['model'] == car_model) &
        (data2023['assembly'] == assembly) &
        (data2023['year'] == year) &
        (data2023['engine'] == engine) &
        (data2023['transmission'] == transmission) &
        (data2023['mileage'] == mileage)
    ]

    if not actual_row.empty:
        actual_price = float(actual_row.iloc[0]['price'])
    else:
        actual_price = "No exact match found in dataset"

    return predicted_price, actual_price



# ============================================================
# 8. EXAMPLE TEST
# ============================================================
predicted, actual = predict_price(
    car_model=top_models[1],
    assembly=0,
    year=2010,
    engine=1300,
    transmission=1,
    mileage=60000
)

print("Predicted Price:", predicted)
print("Actual Price:", actual)
