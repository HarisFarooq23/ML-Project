import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================
# 1. LOAD DATA
# ============================================================

data2023 = pd.read_csv('2023data.csv')

# ============================================================
# 2. DATA CLEANING & PREPROCESSING
# ============================================================

# Drop irrelevant columns
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

# Remove missing critical values
data2023.dropna(
    subset=['price','mileage','year','engine','model'],
    inplace=True
)

# =============================import pandas as pd===============================
# 3. IDENTIFY TOP 5 MOST POSTED MODELS
# ============================================================

top_models = data2023['model'].value_counts().head(5).index.tolist()
print("Top 5 Models:", top_models)

# ============================================================
# 4. COST FUNCTION (WITH L2 REGULARIZATION)
# ============================================================

def compute_cost(X, y, w, b, lambda_reg=0.01):
    m = len(y)
    predictions = X.dot(w) + b
    mse = (1/(2*m)) * np.sum((predictions - y)**2)
    reg = (lambda_reg/(2*m)) * np.sum(w**2)
    return mse + reg


# ============================================================
# 5. GRADIENT DESCENT WITH EARLY STOPPING
# ============================================================

def gradient_descent(X, y, w, b, learning_rate, epochs,
                     lambda_reg=0.01, tolerance=1e-6):

    m = len(y)
    prev_cost = float('inf')

    for i in range(epochs):

        predictions = X.dot(w) + b

        dw = (1/m) * X.T.dot(predictions - y) + (lambda_reg/m) * w
        db = (1/m) * np.sum(predictions - y)

        w -= learning_rate * dw
        b -= learning_rate * db

        cost = compute_cost(X, y, w, b, lambda_reg)

        if abs(prev_cost - cost) < tolerance:
            print(f"Early stopping at epoch {i}")
            break

        prev_cost = cost

        if i % 200 == 0:
            print(f"Epoch {i} | Cost: {cost}")

    return w, b


# ============================================================
# 6. R2 SCORE FUNCTION
# ============================================================

def r2_score(X, y, w, b):
    predictions = X.dot(w) + b
    ss_total = np.sum((y - y.mean())**2)
    ss_res = np.sum((y - predictions)**2)
    return 1 - (ss_res / ss_total)


# ============================================================
# 7. TRAIN MODEL FUNCTION
# ============================================================

def train_model(df):

    # ---- FEATURE ENGINEERING ----
    current_year = 2023
    df['age'] = current_year - df['year']

    feature_cols = ['assembly','age','engine','transmission','mileage']

    X = df[feature_cols].values
    y = df['price'].values.reshape(-1,1)

    # ---- NORMALIZE FEATURES ----
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X = (X - X_mean) / X_std

    # ---- NORMALIZE TARGET ----
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1
    y = (y - y_mean) / y_std

    # ---- TRAIN / VALIDATION SPLIT ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    m, n = X_train.shape
    w = np.zeros((n,1))
    b = 0

    # ---- TRAIN ----
    w, b = gradient_descent(
        X_train, y_train,
        w, b,
        learning_rate=0.01,
        epochs=2000,
        lambda_reg=0.01
    )

    # ---- VALIDATION SCORE ----
    val_r2 = r2_score(X_val, y_val, w, b)
    print("Validation R2:", val_r2)

    return {
        "w": w,
        "b": b,
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std
    }


# ============================================================
# 8. TRAIN MODELS FOR TOP 5 CARS
# ============================================================

models = {}

for car in top_models:
    print(f"\nTraining model for: {car}")
    df_car = data2023[data2023['model'] == car].copy()
    models[car] = train_model(df_car)

print("\nAll models trained successfully!")


# ============================================================
# 9. PREDICTION FUNCTION
# ============================================================

def predict_price(car_model, assembly, year, engine,
                  transmission, mileage):

    if car_model not in models:
        return "Model not trained."

    model = models[car_model]

    age = 2023 - year

    x = np.array([assembly, age, engine,
                  transmission, mileage])

    x = (x - model["X_mean"]) / model["X_std"]

    pred_norm = x.dot(model["w"]) + model["b"]
    pred = pred_norm * model["y_std"] + model["y_mean"]

    return float(pred.item())


# ============================================================
# 10. EXAMPLE TEST
# ============================================================

test_price = predict_price(
    car_model=top_models[0],
    assembly=0,
    year=2015,
    engine=1300,
    transmission=1,
    mileage=60000
)

print("Predicted Price:", test_price)