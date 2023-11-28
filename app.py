from flask import Flask, render_template, request
import pickle
import pandas as pd

clf = pickle.load(open('diabetes.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == "POST":
        gender = request.form['gender']
        age = request.form['age']
        hypertension = request.form['hypertension']
        heart_disease = request.form['heart_disease']
        smoking_history = request.form['smoking_history']
        bmi = request.form['bmi']
        HbA1c_level = request.form['HbA1c_level']
        blood_glucose_level = request.form['blood_glucose_level']

        # user_input = [gender, age, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]

        user_data = pd.DataFrame(columns=['gender', 'age', 'hypertension',
                                        'heart_disease', 'smoking_history', 'bmi',
                                        'HbA1c_level', 'blood_glucose_level'])

        user_data['gender'] = gender
        user_data['age'] = age
        user_data['hypertension'] = hypertension
        user_data['heart_disease'] = heart_disease
        user_data['smoking_history'] = smoking_history
        user_data['bmi'] = bmi
        user_data['HbA1c_level'] = HbA1c_level
        user_data['blood_glucose_level'] = blood_glucose_level
        
        # pred = clf.predict(user_data)
        return user_data

if __name__ == "__main__":
    app.run(debug=True)

