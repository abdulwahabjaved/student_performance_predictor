import pandas as pd

df=pd.read_csv("student_data.csv")

print("/n rows and columns:")
print(df.shape)
print("\nPehli 5 rows:")
print(df.head())

print("\nCOLUMN NAMES:")
print(df.columns.tolist())


print("\nDATATYPES:")
print(df.dtypes)

print("\nBASIC STATS:")
print(df.describe())

print("\nKOI MISSING VALUES HAIN?")
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder

# ── Step 2: Preprocessing ──────────────────────────
print("\n" + "="*40)
print("STEP 2 - PREPROCESSING")

# Text columns dhundo
text_columns = df.select_dtypes(include='object').columns
print("\nText columns jo convert hongi:")
print(list(text_columns))

# Label Encoding - text ko numbers mein badlo
le = LabelEncoder()
df_encoded = df.copy()  # original data safe rakhne ke liye copy

for col in text_columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Check karo
print("\nPEHLE (original):")
print(df[['sex', 'school', 'address']].head(3))

print("\nBAD MEIN (encoded):")
print(df_encoded[['sex', 'school', 'address']].head(3))

# Features aur Target alag karo
X = df_encoded.drop('G3', axis=1)  # Input features
y = df_encoded['G3']               # Target (jo predict karni hai)

print("\nX shape (features):", X.shape)
print("y shape (target):", y.shape)
print("\nPreprocessing complete! ✅")


from sklearn.model_selection import train_test_split

# ── Step 3: Data Split ──────────────────────────
print("\n" + "="*40)
print("STEP 3 - DATA SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% testing ke liye
    random_state=42     # har baar same split mile
)

print("Total data:    ", len(X))
print("Training data: ", len(X_train))
print("Testing data:  ", len(X_test))
print("\nSplit complete! ✅")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# ── Step 4: Model Training ──────────────────────────
print("\n" + "=" * 40)
print("STEP 4 - MODEL TRAINING")

# Teen models banao
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Har model ko train karo aur test karo
results = {}

for name, model in models.items():
    # Train karo
    model.fit(X_train, y_train)

    # Predict karo
    predictions = model.predict(X_test)

    # Score calculate karo
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    # Save karo
    results[name] = {
        'model': model,
        'predictions': predictions,
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }

    print(f"\n{name}:")
    print(f"  R² Score : {r2:.3f}  ← 1.0 perfect hota hai")
    print(f"  RMSE     : {rmse:.3f}  ← kam ho utna acha")
    print(f"  MAE      : {mae:.3f}  ← kam ho utna acha")

# Best model dhundo
best_name = max(results, key=lambda n: results[n]['r2'])
print(f"\n🏆 Best Model: {best_name} (R²={results[best_name]['r2']:.3f})")
print("\nTraining complete! ✅")


import matplotlib.pyplot as plt

# ── Step 5: Visualization ──────────────────────────
print("\n" + "="*40)
print("STEP 5 - GRAPHS BAN RAHE HAIN...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Student Performance Analysis', fontsize=16, fontweight='bold')

# ── Graph 1: Grade Distribution ──────────────────
ax1 = axes[0, 0]
ax1.hist(df['G3'], bins=20, color='steelblue', edgecolor='white')
ax1.axvline(df['G3'].mean(), color='red', linestyle='--',
            label=f'Average: {df["G3"].mean():.1f}')
ax1.set_title('Final Grade (G3) Distribution')
ax1.set_xlabel('Grade')
ax1.set_ylabel('Number of Students')
ax1.legend()

# ── Graph 2: Model Comparison ────────────────────
ax2 = axes[0, 1]
model_names = list(results.keys())
r2_scores   = [results[n]['r2'] for n in model_names]
colors      = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax2.bar(model_names, r2_scores, color=colors, edgecolor='white')
for bar, val in zip(bars, r2_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', fontweight='bold')
ax2.set_title('Model Comparison (R² Score)')
ax2.set_ylabel('R² Score (Higher = Better)')
ax2.set_ylim(0, 1.1)
ax2.set_xticklabels(model_names, rotation=10)

# ── Graph 3: Actual vs Predicted ─────────────────
ax3 = axes[1, 0]
best_preds = results[best_name]['predictions']
ax3.scatter(y_test, best_preds, alpha=0.6, color='steelblue', s=50)
ax3.plot([0, 20], [0, 20], color='red', linestyle='--', label='Perfect Line')
ax3.set_title(f'Actual vs Predicted ({best_name})')
ax3.set_xlabel('Actual Grade')
ax3.set_ylabel('Predicted Grade')
ax3.legend()

# ── Graph 4: Feature Importance ──────────────────
ax4 = axes[1, 1]
rf_model   = results['Random Forest']['model']
importance = pd.Series(rf_model.feature_importances_, index=X.columns)
top10      = importance.sort_values(ascending=True).tail(10)
top10.plot(kind='barh', ax=ax4, color='steelblue')
ax4.set_title('Top 10 Important Features')
ax4.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('student_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graphs save ho gaye! ✅")
print("\nFile: student_analysis.png")

# ── Step 6: Naye Student Ka Grade Predict Karo ────
print("\n" + "="*40)
print("STEP 6 - PREDICTION")

# Ek naya student banao
# (yeh values tum khud badal sakte ho!)
naya_student = {
    'school': 0,        # 0=GP, 1=MS
    'sex': 0,           # 0=Female, 1=Male
    'age': 17,
    'address': 1,       # 1=Urban, 0=Rural
    'famsize': 0,       # 0=GT3, 1=LE3
    'Pstatus': 1,       # 1=Together, 0=Apart
    'Medu': 3,          # Mother education (0-4)
    'Fedu': 2,          # Father education (0-4)
    'Mjob': 1,
    'Fjob': 2,
    'reason': 0,
    'guardian': 0,
    'traveltime': 1,    # 1=<15 min
    'studytime': 3,     # 3=5-10 hours
    'failures': 0,      # 0=koi failure nahi
    'schoolsup': 0,
    'famsup': 1,
    'paid': 0,
    'activities': 1,
    'nursery': 1,
    'higher': 1,        # 1=higher education chahiye
    'internet': 1,      # 1=internet hai
    'romantic': 0,      # 0=no relationship
    'famrel': 4,        # family relations (1-5)
    'freetime': 3,
    'goout': 2,
    'Dalc': 1,
    'Walc': 1,
    'health': 4,
    'absences': 2,
    'G1': 12,           # 1st term grade
    'G2': 5,           # 2nd term grade
}

# DataFrame banao
student_df = pd.DataFrame([naya_student])

# Best model se predict karo
best_model     = results[best_name]['model']
predicted_grade = best_model.predict(student_df)[0]

print(f"\nStudent ki details:")
print(f"  Study Time : {naya_student['studytime']} (5-10 hrs/week)")
print(f"  Failures   : {naya_student['failures']}")
print(f"  G1 Grade   : {naya_student['G1']}")
print(f"  G2 Grade   : {naya_student['G2']}")
print(f"  Absences   : {naya_student['absences']}")
print(f"\n🎯 Predicted Final Grade (G3): {predicted_grade:.1f} / 20")

if predicted_grade >= 15:
    print("⭐ Excellent Student!")
elif predicted_grade >= 10:
    print("✅ Average Student - Pass!")
else:
    print("⚠️  At Risk - Needs Help!")