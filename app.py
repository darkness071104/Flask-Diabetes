from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

clf = joblib.load('diabetes.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        user_data = pd.DataFrame(columns=['gender', 'age', 'hypertension',
                                          'heart_disease', 'smoking_history', 'bmi',
                                          'HbA1c_level', 'blood_glucose_level'])

        for column in user_data.columns:
            user_input = request.form.get(column)
            user_data.at[0, column] = float(user_input)

        user_data = user_data.apply(pd.to_numeric, errors='coerce')

        # Normalize the user data using the fitted scaler
        user_data_scaled_array = scaler.transform(user_data.values.reshape(1, -1))

        user_prediction = clf.predict(user_data_scaled_array)
        prediction = 'Diabetes' if user_prediction[0] == 1 else 'Not Diabetes'

    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

