from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("cancer_patient_data_sets.csv")
df.columns = df.columns.str.replace(" ", "_")
df['Level'] = df['Level'].replace({'Low': 1, 'Medium': 2, 'High': 3})

# Encode categorical variables
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

# Splitting data into features and target variable
X = df.drop('Level', axis=1)
y = df['Level']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        features = [float(request.form.get(col)) for col in X.columns]
        input_data = np.array([features])
        prediction = model.predict(input_data)

        # If 'Level' is not in label_encoders, it means it's the target variable
        if 'Level' not in label_encoders:
            cancer_level = prediction[0]
        else:
            cancer_level = label_encoders['Level'].inverse_transform(prediction)[0]

        return render_template('result.html', level=cancer_level)
    return render_template('predict.html', columns=X.columns)



@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
