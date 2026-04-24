from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ── Model Train Karo ──────────────────────────────
df = pd.read_csv('student_data.csv')
df_encoded = df.copy()

le = LabelEncoder()
text_columns = df.select_dtypes(include='str').columns
for col in text_columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop('G3', axis=1)
y = df_encoded['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model ready!")

# ── Routes ────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Form se data lo
    student = {
        'school': 0, 'sex': int(request.form['sex']),
        'age': int(request.form['age']),
        'address': 1, 'famsize': 0, 'Pstatus': 1,
        'Medu': int(request.form['Medu']),
        'Fedu': int(request.form['Fedu']),
        'Mjob': 1, 'Fjob': 2, 'reason': 0, 'guardian': 0,
        'traveltime': 1,
        'studytime': int(request.form['studytime']),
        'failures': int(request.form['failures']),
        'schoolsup': 0, 'famsup': 1, 'paid': 0,
        'activities': 1, 'nursery': 1, 'higher': 1,
        'internet': int(request.form['internet']),
        'romantic': int(request.form['romantic']),
        'famrel': int(request.form['famrel']),
        'freetime': 3, 'goout': 2, 'Dalc': 1, 'Walc': 1,
        'health': int(request.form['health']),
        'absences': int(request.form['absences']),
        'G1': int(request.form['G1']),
        'G2': int(request.form['G2']),
    }

    student_df = pd.DataFrame([student])
    grade = model.predict(student_df)[0]
    grade = round(grade, 1)

    if grade >= 15:
        status = "⭐ Excellent!"
        color  = "green"
    elif grade >= 10:
        status = "✅ Pass!"
        color  = "blue"
    else:
        status = "⚠️ At Risk!"
        color  = "red"

    return render_template('index.html',
                           prediction=grade,
                           status=status,
                           color=color)

if __name__ == '__main__':
    app.run(debug=True)