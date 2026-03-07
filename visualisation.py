import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set beautiful plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 11

# ============================================================
# 1. LOAD DATA
# ============================================================

data2023 = pd.read_csv('2023data.csv')
print("✅ Data loaded successfully!")
print(f"📊 Dataset shape: {data2023.shape}")

# ============================================================
# 2. DATA CLEANING & PREPROCESSING
# ============================================================

# Store original shape for visualization
original_shape = data2023.shape

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

print(f"✅ Data cleaned! Shape after cleaning: {data2023.shape}")
print(f"🗑️ Removed {original_shape[0] - data2023.shape[0]} rows")

# ============================================================
# 3. EXPLORATORY DATA VISUALIZATION
# ============================================================

def create_eda_visualizations(df):
    """Create comprehensive EDA visualizations"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Distribution of Prices
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['price'], bins=50, color='#2E86AB', edgecolor='white', alpha=0.7)
    ax1.set_xlabel('Price (PKR)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('📈 Distribution of Car Prices', fontweight='bold')
    ax1.axvline(df['price'].mean(), color='red', linestyle='--', label=f'Mean: {df["price"].mean():,.0f}')
    ax1.axvline(df['price'].median(), color='green', linestyle='--', label=f'Median: {df["price"].median():,.0f}')
    ax1.legend()
    
    # 2. Price by Transmission
    ax2 = fig.add_subplot(gs[0, 1])
    transmission_prices = [df[df['transmission'] == 1]['price'], df[df['transmission'] == 0]['price']]
    bp = ax2.boxplot(transmission_prices, labels=['Automatic', 'Manual'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#A23B72', '#F18F01']):
        patch.set_facecolor(color)
    ax2.set_ylabel('Price (PKR)')
    ax2.set_title('🚗 Price Distribution by Transmission', fontweight='bold')
    ax2.set_yscale('log')
    
    # 3. Price by Assembly
    ax3 = fig.add_subplot(gs[0, 2])
    assembly_prices = [df[df['assembly'] == 1]['price'], df[df['assembly'] == 0]['price']]
    bp = ax3.boxplot(assembly_prices, labels=['Imported', 'Local'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#F24236', '#7EB09B']):
        patch.set_facecolor(color)
    ax3.set_ylabel('Price (PKR)')
    ax3.set_title('🌍 Price Distribution by Assembly', fontweight='bold')
    ax3.set_yscale('log')
    
    # 4. Price vs Age
    ax4 = fig.add_subplot(gs[1, 0])
    df_sampled = df.sample(min(1000, len(df)))  # Sample for better visualization
    scatter = ax4.scatter(2023 - df_sampled['year'], df_sampled['price'], 
                          c=df_sampled['engine'], cmap='viridis', alpha=0.6, s=30)
    ax4.set_xlabel('Age (years)')
    ax4.set_ylabel('Price (PKR)')
    ax4.set_title('📉 Price vs Age (colored by Engine Size)', fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Engine Size (cc)')
    ax4.set_yscale('log')
    
    # 5. Price vs Mileage
    ax5 = fig.add_subplot(gs[1, 1])
    scatter = ax5.scatter(df_sampled['mileage'], df_sampled['price'], 
                          c=df_sampled['year'], cmap='coolwarm', alpha=0.6, s=30)
    ax5.set_xlabel('Mileage (km)')
    ax5.set_ylabel('Price (PKR)')
    ax5.set_title('🛣️ Price vs Mileage (colored by Year)', fontweight='bold')
    plt.colorbar(scatter, ax=ax5, label='Year')
    ax5.set_yscale('log')
    
    # 6. Engine Size Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(df['engine'], bins=30, color='#A23B72', edgecolor='white', alpha=0.7)
    ax6.set_xlabel('Engine Size (cc)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('⚙️ Engine Size Distribution', fontweight='bold')
    
    # 7. Top Models Bar Chart
    ax7 = fig.add_subplot(gs[2, 0])
    top_10_models = df['model'].value_counts().head(10)
    bars = ax7.barh(range(len(top_10_models)), top_10_models.values, color=sns.color_palette("viridis", 10))
    ax7.set_yticks(range(len(top_10_models)))
    ax7.set_yticklabels(top_10_models.index)
    ax7.set_xlabel('Number of Listings')
    ax7.set_title('🏆 Top 10 Most Listed Models', fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, top_10_models.values)):
        ax7.text(val, bar.get_y() + bar.get_height()/2, f' {val}', va='center')
    
    # 8. Correlation Heatmap
    ax8 = fig.add_subplot(gs[2, 1:])
    numeric_cols = ['price', 'year', 'mileage', 'engine', 'transmission', 'assembly']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax8)
    ax8.set_title('🔗 Feature Correlation Matrix', fontweight='bold', pad=20)
    
    plt.suptitle('🚗 Car Price Prediction - Exploratory Data Analysis', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# Create EDA visualizations
create_eda_visualizations(data2023)

# ============================================================
# 4. IDENTIFY TOP 5 MOST POSTED MODELS
# ============================================================

top_models = data2023['model'].value_counts().head(5).index.tolist()
print("\n" + "="*50)
print("🎯 TOP 5 MODELS SELECTED FOR ANALYSIS")
print("="*50)
for i, model in enumerate(top_models, 1):
    count = data2023['model'].value_counts()[model]
    print(f"{i}. {model}: {count} listings ({count/len(data2023)*100:.1f}% of total)")

# Visualize top models
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Pie chart
model_counts = data2023['model'].value_counts().head(5)
other_count = len(data2023) - model_counts.sum()
sizes = list(model_counts.values) + [other_count]
labels = list(model_counts.index) + ['Other Models']
colors = sns.color_palette('husl', 6)
axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
axes[0].set_title('📊 Market Share of Top 5 Models', fontweight='bold', fontsize=14)

# Bar chart with average prices
model_avg_prices = [data2023[data2023['model'] == model]['price'].mean() for model in top_models]
bars = axes[1].bar(range(len(top_models)), model_avg_prices, color=sns.color_palette("husl", 5))
axes[1].set_xticks(range(len(top_models)))
axes[1].set_xticklabels(top_models, rotation=45, ha='right')
axes[1].set_ylabel('Average Price (PKR)')
axes[1].set_title('💰 Average Price by Model', fontweight='bold', fontsize=14)

# Add value labels on bars
for bar, price in zip(bars, model_avg_prices):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{price:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ============================================================
# 5. COST FUNCTION (WITH L2 REGULARIZATION)
# ============================================================

def compute_cost(X, y, w, b, lambda_reg=0.01):
    m = len(y)
    predictions = X.dot(w) + b
    mse = (1/(2*m)) * np.sum((predictions - y)**2)
    reg = (lambda_reg/(2*m)) * np.sum(w**2)
    return mse + reg

# ============================================================
# 6. GRADIENT DESCENT WITH EARLY STOPPING
# ============================================================

def gradient_descent(X, y, w, b, learning_rate, epochs,
                     lambda_reg=0.01, tolerance=1e-6, model_name=""):

    m = len(y)
    prev_cost = float('inf')
    cost_history = []
    weight_history = []
    
    # Create figure for real-time monitoring
    if model_name:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'📈 Training Progress - {model_name}', fontsize=14, fontweight='bold')

    for i in range(epochs):

        predictions = X.dot(w) + b

        dw = (1/m) * X.T.dot(predictions - y) + (lambda_reg/m) * w
        db = (1/m) * np.sum(predictions - y)

        w -= learning_rate * dw
        b -= learning_rate * db

        cost = compute_cost(X, y, w, b, lambda_reg)
        cost_history.append(cost)
        weight_history.append(w.copy())

        if abs(prev_cost - cost) < tolerance:
            print(f"  ⏹️ Early stopping at epoch {i}")
            break

        prev_cost = cost

        if i % 200 == 0 and model_name:
            print(f"  Epoch {i} | Cost: {cost:.6f}")

    # Plot training progress
    if model_name:
        ax1.plot(cost_history, color='#2E86AB', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cost')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot weight evolution (first few features)
        weight_history = np.array(weight_history).squeeze()
        if weight_history.ndim > 1:
            for j in range(min(3, weight_history.shape[1])):
                ax2.plot(weight_history[:, j], label=f'Feature {j+1}', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Weight Value')
        ax2.set_title('Weight Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    return w, b, cost_history

# ============================================================
# 7. R2 SCORE FUNCTION
# ============================================================

def r2_score(X, y, w, b):
    predictions = X.dot(w) + b
    ss_total = np.sum((y - y.mean())**2)
    ss_res = np.sum((y - predictions)**2)
    return 1 - (ss_res / ss_total)

# ============================================================
# 8. TRAIN MODEL FUNCTION WITH VISUALIZATIONS
# ============================================================

def train_model(df, model_name=""):

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

    # ---- TRAIN WITH VISUALIZATION ----
    print(f"\n  Training {model_name}...")
    w, b, cost_history = gradient_descent(
        X_train, y_train,
        w, b,
        learning_rate=0.01,
        epochs=2000,
        lambda_reg=0.01,
        model_name=model_name
    )

    # ---- VALIDATION SCORE ----
    val_r2 = r2_score(X_val, y_val, w, b)
    print(f"  ✅ Validation R² Score: {val_r2:.4f}")
    
    # Create model performance visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Training loss curve
    axes[0, 0].plot(cost_history, color='#2E86AB', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cost')
    axes[0, 0].set_title(f'📉 Training Loss Curve - {model_name}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Feature importance
    feature_importance = w.flatten()
    colors = ['#F24236' if x < 0 else '#2E86AB' for x in feature_importance]
    bars = axes[0, 1].bar(feature_cols, feature_importance, color=colors)
    axes[0, 1].set_xlabel('Features')
    axes[0, 1].set_ylabel('Weight')
    axes[0, 1].set_title('⚖️ Feature Importance')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Predictions vs Actual (Validation set)
    y_val_pred = X_val.dot(w) + b
    y_val_actual = y_val
    
    # Denormalize for interpretation
    y_val_actual_denorm = y_val_actual * y_std + y_mean
    y_val_pred_denorm = y_val_pred * y_std + y_mean
    
    axes[1, 0].scatter(y_val_actual_denorm, y_val_pred_denorm, alpha=0.6, color='#A23B72', s=50)
    max_val = max(y_val_actual_denorm.max(), y_val_pred_denorm.max())
    min_val = min(y_val_actual_denorm.min(), y_val_pred_denorm.min())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 0].set_xlabel('Actual Price (PKR)')
    axes[1, 0].set_ylabel('Predicted Price (PKR)')
    axes[1, 0].set_title('🎯 Predictions vs Actual')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals distribution
    residuals = (y_val_pred_denorm - y_val_actual_denorm).flatten()
    axes[1, 1].hist(residuals, bins=30, color='#F18F01', edgecolor='white', alpha=0.7)
    axes[1, 1].set_xlabel('Prediction Error (PKR)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('📊 Residual Distribution')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add statistics text
    axes[1, 1].text(0.05, 0.95, f'Mean Error: {residuals.mean():.2f}\nStd Error: {residuals.std():.2f}', 
                    transform=axes[1, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

    return {
        "w": w,
        "b": b,
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "r2_score": val_r2,
        "feature_cols": feature_cols
    }

# ============================================================
# 9. TRAIN MODELS FOR TOP 5 CARS WITH COMPARISON
# ============================================================

print("\n" + "="*60)
print("🎯 TRAINING MODELS FOR TOP 5 CARS")
print("="*60)

models = {}
model_performance = {}

for car in top_models:
    print(f"\n📌 Training model for: {car}")
    df_car = data2023[data2023['model'] == car].copy()
    models[car] = train_model(df_car, model_name=car)
    model_performance[car] = models[car]["r2_score"]

print("\n✅ All models trained successfully!")

# Visualize model performance comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# R² Scores comparison
models_names = list(model_performance.keys())
r2_scores = list(model_performance.values())
colors = sns.color_palette("viridis", len(models_names))
bars = ax1.bar(models_names, r2_scores, color=colors)
ax1.set_xlabel('Car Model')
ax1.set_ylabel('R² Score')
ax1.set_title('📊 Model Performance Comparison (R² Score)')
ax1.tick_params(axis='x', rotation=45)
ax1.set_ylim([0, 1])
ax1.axhline(y=0.7, color='red', linestyle='--', label='Good Performance Threshold')
ax1.legend()

# Add value labels on bars
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.3f}', ha='center', va='bottom')

# Sample sizes comparison
sample_sizes = [len(data2023[data2023['model'] == model]) for model in top_models]
bars = ax2.bar(models_names, sample_sizes, color=colors)
ax2.set_xlabel('Car Model')
ax2.set_ylabel('Number of Listings')
ax2.set_title('📈 Training Data Size by Model')
ax2.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, size in zip(bars, sample_sizes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{size}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ============================================================
# 10. PREDICTION FUNCTION WITH VISUALIZATION
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

def visualize_prediction(car_model, features_dict, predicted_price):
    """Create a beautiful visualization of the prediction"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'🚗 Price Prediction Analysis - {car_model}', fontsize=16, fontweight='bold')
    
    # 1. Feature importance gauge
    ax1 = axes[0, 0]
    model = models[car_model]
    importance = model["w"].flatten()
    features = model["feature_cols"]
    
    # Normalize importance for visualization
    importance_norm = importance / np.max(np.abs(importance))
    colors = ['#F24236' if x < 0 else '#2E86AB' for x in importance_norm]
    
    y_pos = np.arange(len(features))
    ax1.barh(y_pos, importance_norm, color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features)
    ax1.set_xlabel('Relative Importance')
    ax1.set_title('⚡ Feature Impact on Price')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 2. Price meter
    ax2 = axes[0, 1]
    
    # Get similar cars for context
    df_car = data2023[data2023['model'] == car_model]
    price_range = [df_car['price'].min(), df_car['price'].max()]
    
    # Create gauge-like visualization
    ax2.barh([0], [price_range[1] - price_range[0]], left=price_range[0], 
             height=0.3, color='lightgray', alpha=0.5)
    ax2.barh([0], [predicted_price - price_range[0]], left=price_range[0], 
             height=0.3, color='#2E86AB')
    ax2.scatter(predicted_price, 0, color='red', s=200, zorder=5, 
               marker='D', edgecolor='white', linewidth=2)
    
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel('Price (PKR)')
    ax2.set_title('💰 Predicted Price Context')
    ax2.set_yticks([])
    ax2.grid(True, axis='x', alpha=0.3)
    
    # Add price range text
    ax2.text(price_range[0], 0.2, f'Min: {price_range[0]:,.0f}', ha='center')
    ax2.text(price_range[1], 0.2, f'Max: {price_range[1]:,.0f}', ha='center')
    ax2.text(predicted_price, -0.2, f'Predicted: {predicted_price:,.0f}', 
            ha='center', fontweight='bold')
    
    # 3. Feature values spider chart
    ax3 = axes[1, 0]
    
    # Normalize feature values for spider chart
    df_car = data2023[data2023['model'] == car_model]
    feature_values = np.array([assembly, 2023-year, engine, transmission, mileage])
    feature_ranges = []
    
    for i, feature in enumerate(features):
        min_val = df_car[feature].min()
        max_val = df_car[feature].max()
        if max_val > min_val:
            norm_val = (feature_values[i] - min_val) / (max_val - min_val)
        else:
            norm_val = 0.5
        feature_ranges.append(norm_val)
    
    # Create spider chart
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    feature_ranges += feature_ranges[:1]
    angles += angles[:1]
    
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    ax3.plot(angles, feature_ranges, 'o-', linewidth=2, color='#A23B72')
    ax3.fill(angles, feature_ranges, alpha=0.25, color='#A23B72')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(features)
    ax3.set_title('🕷️ Feature Profile (Normalized)', pad=20)
    ax3.set_ylim(0, 1)
    
    # 4. Confidence interval
    ax4 = axes[1, 1]
    
    # Calculate confidence interval based on similar cars
    similar_cars = df_car[
        (df_car['year'].between(year-2, year+2)) &
        (df_car['mileage'].between(mileage*0.8, mileage*1.2))
    ]
    
    if len(similar_cars) > 0:
        prices = similar_cars['price'].values
        mean_price = prices.mean()
        std_price = prices.std()
        
        x = np.linspace(mean_price - 3*std_price, mean_price + 3*std_price, 100)
        y = (1/(std_price * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mean_price)/std_price)**2)
        
        ax4.plot(x, y, color='#2E86AB', linewidth=2)
        ax4.fill_between(x, y, where=(x >= predicted_price - std_price) & (x <= predicted_price + std_price), 
                        color='#2E86AB', alpha=0.3, label='±1 Std Dev')
        ax4.axvline(predicted_price, color='red', linestyle='--', linewidth=2, label='Prediction')
        
        ax4.set_xlabel('Price (PKR)')
        ax4.set_ylabel('Density')
        ax4.set_title('📊 Prediction Confidence')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add confidence text
        confidence = 1 - abs(predicted_price - mean_price) / (3*std_price) if std_price > 0 else 0.5
        confidence = max(0, min(1, confidence))
        ax4.text(0.05, 0.95, f'Confidence: {confidence:.1%}', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax4.text(0.5, 0.5, 'Insufficient similar cars\nfor confidence estimation', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.show()

# ============================================================
# 11. EXAMPLE TEST WITH VISUALIZATION
# ============================================================

print("\n" + "="*60)
print("🎯 TESTING PREDICTION MODEL")
print("="*60)

test_car = top_models[0]
test_features = {
    'assembly': 0,
    'year': 2015,
    'engine': 1300,
    'transmission': 1,
    'mileage': 60000
}

test_price = predict_price(
    car_model=test_car,
    assembly=test_features['assembly'],
    year=test_features['year'],
    engine=test_features['engine'],
    transmission=test_features['transmission'],
    mileage=test_features['mileage']
)

print(f"\n📊 Test Prediction Results:")
print(f"   Model: {test_car}")
print(f"   Year: {test_features['year']}")
print(f"   Engine: {test_features['engine']}cc")
print(f"   Mileage: {test_features['mileage']:,} km")
print(f"   Transmission: {'Automatic' if test_features['transmission'] == 1 else 'Manual'}")
print(f"   Assembly: {'Imported' if test_features['assembly'] == 1 else 'Local'}")
print(f"\n💰 Predicted Price: PKR {test_price:,.2f}")

# Create detailed prediction visualization
visualize_prediction(test_car, test_features, test_price)

# ============================================================
# 12. FINAL SUMMARY VISUALIZATION
# ============================================================

print("\n" + "="*60)
print("📊 FINAL MODEL SUMMARY")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🚗 Car Price Prediction Model - Final Summary', fontsize=18, fontweight='bold', y=1.02)

# 1. Model performance summary
ax1 = axes[0, 0]
models_list = list(models.keys())
r2_scores = [models[m]['r2_score'] for m in models_list]
training_sizes = [len(data2023[data2023['model'] == m]) for m in models_list]

scatter = ax1.scatter(training_sizes, r2_scores, s=200, c=r2_scores, 
                     cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1)
for i, model in enumerate(models_list):
    ax1.annotate(model, (training_sizes[i], r2_scores[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax1.set_xlabel('Training Data Size')
ax1.set_ylabel('R² Score')
ax1.set_title('Performance vs Training Size')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='R² Score')

# 2. Average feature weights across all models
ax2 = axes[0, 1]
feature_names = ['Assembly', 'Age', 'Engine', 'Transmission', 'Mileage']
all_weights = np.array([models[m]['w'].flatten() for m in models_list])
mean_weights = np.mean(all_weights, axis=0)
std_weights = np.std(all_weights, axis=0)

x_pos = np.arange(len(feature_names))
ax2.bar(x_pos, mean_weights, yerr=std_weights, capsize=5, 
        color='#2E86AB', edgecolor='black', alpha=0.7)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(feature_names)
ax2.set_xlabel('Features')
ax2.set_ylabel('Average Weight')
ax2.set_title('📊 Average Feature Importance Across All Models')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
ax2.grid(True, axis='y', alpha=0.3)

# 3. Price distribution for each model
ax3 = axes[1, 0]
price_data = [data2023[data2023['model'] == m]['price'] for m in models_list]
bp = ax3.boxplot(price_data, labels=models_list, patch_artist=True)
for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(models_list))):
    patch.set_facecolor(color)
ax3.set_ylabel('Price (PKR)')
ax3.set_title('💰 Price Distribution by Model')
ax3.tick_params(axis='x', rotation=45)
ax3.set_yscale('log')
ax3.grid(True, axis='y', alpha=0.3)

# 4. Prediction accuracy by price range
ax4 = axes[1, 1]
all_predictions = []
all_actuals = []

for model_name in models_list:
    model = models[model_name]
    df_car = data2023[data2023['model'] == model_name].copy()
    df_car['age'] = 2023 - df_car['year']
    
    X = df_car[feature_names].values
    X = (X - model["X_mean"]) / model["X_std"]
    y = df_car['price'].values.reshape(-1,1)
    
    pred_norm = X.dot(model["w"]) + model["b"]
    pred = pred_norm * model["y_std"] + model["y_mean"]
    
    all_predictions.extend(pred.flatten())
    all_actuals.extend(y.flatten())

# Create price bins
price_bins = pd.cut(all_actuals, bins=10)
accuracy_by_bin = []
for bin_name in price_bins.cat.categories:
    mask = price_bins == bin_name
    if mask.sum() > 0:
        preds_in_bin = np.array(all_predictions)[mask]
        actuals_in_bin = np.array(all_actuals)[mask]
        mape = np.mean(np.abs((preds_in_bin - actuals_in_bin) / actuals_in_bin)) * 100
        accuracy_by_bin.append((bin_name, mape))

bins = [str(bin_.left)[:8] + ' - ' + str(bin_.right)[:8] for bin_, _ in accuracy_by_bin]
accuracies = [acc for _, acc in accuracy_by_bin]

bars = ax4.bar(range(len(bins)), accuracies, color=plt.cm.RdYlGn_r(accuracies/20))
ax4.set_xticks(range(len(bins)))
ax4.set_xticklabels(bins, rotation=45, ha='right', fontsize=8)
ax4.set_xlabel('Price Range (PKR)')
ax4.set_ylabel('Mean Absolute Percentage Error (%)')
ax4.set_title('🎯 Prediction Accuracy by Price Range')
ax4.axhline(y=10, color='red', linestyle='--', label='10% Error Threshold')
ax4.legend()
ax4.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Analysis complete! All visualizations generated successfully.")
print("📈 Check the plots above for detailed insights into your car price prediction model.")