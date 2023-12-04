import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib

diabetes = pd.read_csv('diabetes_prediction_dataset.csv')

# Preprocessing 
smoking_history_dict = {
    'never': 0,
    'former': 1,
    'current': 2,
    'not current': 1,
    'ever': 1
}
diabetes['smoking_history'] = diabetes.smoking_history.map(smoking_history_dict)

gender_dict = {'Female': 0, 'Male': 1}
diabetes['gender'] = diabetes.gender.map(gender_dict)


mode_imputer = SimpleImputer(strategy='most_frequent')
diabetes['smoking_history'] = mode_imputer.fit_transform(
    diabetes[['smoking_history']]
)
diabetes['gender'] = mode_imputer.fit_transform(diabetes[['gender']])

diabetes = pd.DataFrame(diabetes, columns=['gender', 'age', 'hypertension',
                                           'heart_disease', 'smoking_history',
                                           'bmi', 'HbA1c_level',
                                           'blood_glucose_level', 'diabetes'])


data_n = 100000
diabetes = diabetes.sample(n=data_n, random_state=1)
feature_cols = ['gender', 'age', 'hypertension', 'heart_disease',
                'smoking_history',  'bmi', 'HbA1c_level',
                'blood_glucose_level']
X = diabetes[feature_cols]
y = diabetes['diabetes']



scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns=['gender', 'age', 'hypertension',
                                           'heart_disease', 'smoking_history',
                                           'bmi', 'HbA1c_level',
                                           'blood_glucose_level'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# Tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)

clf = clf.fit(X_train, y_train)

# Export 
joblib.dump(clf, 'diabetes.pkl')
joblib.dump(scaler, 'scaler.pkl')