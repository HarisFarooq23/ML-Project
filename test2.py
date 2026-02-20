import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD DATA
# ============================================================
data2023 = pd.read_csv('2023data.csv')

# ============================================================
# 2. DATA CLEANING & PREPROCESSING
# ============================================================

data2023.drop(
    ['addref','color','city','registered','fuel','make','body'],
    axis=1,
    errors='ignore',
    inplace=True
)

# Encode transmission
data2023['transmission'] = data2023['transmission'].map({
    'Automatic': 1,
    'Manual': 0
})

# Handle assembly
data2023['assembly'].fillna("local", inplace=True)

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
    m = len(y)
    predictions = X.dot(w) + b
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost


def compute_r2(X, y, w, b):
    """
    Calculates R² score.
    """
    predictions = X.dot(w) + b
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def gradient_descent(X, y, w, b, learning_rate, epochs):
    m = len(y)

    for i in range(epochs):
        predictions = X.dot(w) + b

        dw = (1/m) * X.T.dot(predictions - y)
        db = (1/m) * np.sum(predictions - y)

        w -= learning_rate * dw
        b -= learning_rate * db

        if i % 200 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Epoch {i} | Cost: {cost}")

    return w, b


# ============================================================
# 5. TRAINING FUNCTION FOR A SINGLE CAR MODEL
# ============================================================

def train_model(df):

    feature_cols = ['assembly','year','engine','transmission','mileage']

    X = df[feature_cols].values
    y = df['price'].values.reshape(-1,1)

    # -------- NORMALIZE FEATURES --------
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
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

    # -------- CALCULATE R2 SCORE --------
    r2_score = compute_r2(X, y, w, b)
    print("Final R2 Score:", r2_score)

    return {
        "w": w,
        "b": b,
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "r2": r2_score
    }


# ============================================================
# 6. TRAIN MODELS FOR TOP 5 CARS
# ============================================================

models = {}

for car in top_models:
    print(f"\nTraining model for: {car}")

    df_car = data2023[data2023['model'] == car].copy()

    models[car] = train_model(df_car)

    print(f"Stored R2 for {car}: {models[car]['r2']}")

print("\nAll models successfully trained!")

# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================

def predict_price(car_model, assembly, year, engine, transmission, mileage):

    if car_model not in models:
        return "Model not trained yet."

    model = models[car_model]

    x = np.array([assembly, year, engine, transmission, mileage])

    # Normalize features
    x_norm = (x - model["X_mean"]) / model["X_std"]

    pred_norm = x_norm.dot(model["w"]) + model["b"]

    predicted_price = float(
        (pred_norm * model["y_std"] + model["y_mean"]).item()
    )

    # Try finding actual price
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

    return predicted_price, actual_price, model["r2"]


# ============================================================
# 8. EXAMPLE TEST
# ============================================================

predicted, actual, r2 = predict_price(
    car_model=top_models[2],
    assembly=0,
    year=2010,
    engine=1300,
    transmission=1,
    mileage=60000
)

print("\nPredicted Price:", predicted)
print("Actual Price:", actual)
print("Model R2 Score:", r2)
