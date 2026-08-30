from flask import Flask, render_template, request
import pandas as pd
import pickle
import json

app = Flask(__name__)

# Load model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Load dataset
car = pd.read_csv('Cleaned_Car.csv')

# Dropdown values
companies = sorted(car['company'].unique())
fuels = sorted(car['fuel_type'].unique())

# Company -> Models mapping
company_model_dict = {}

for company in companies:
    company_model_dict[company] = sorted(
        car[car['company'] == company]['name'].unique()
    )


@app.route('/')
def home():

    return render_template(
        'index.html',
        companies=companies,
        fuels=fuels,
        company_model_dict=json.dumps(company_model_dict)
    )


@app.route('/predict', methods=['POST'])
def predict():

    company = request.form['company']
    name = request.form['name']
    year = int(request.form['year'])
    kms_driven = int(request.form['kms_driven'])
    fuel_type = request.form['fuel_type']

    data = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=[
            'name',
            'company',
            'year',
            'kms_driven',
            'fuel_type'
        ]
    )

    prediction = model.predict(data)[0]

    return render_template(
        'index.html',
        prediction=f"₹ {round(prediction):,}",
        selected_company=company,
        selected_name=name,
        selected_year=year,
        selected_km=kms_driven,
        selected_fuel=fuel_type,
        companies=companies,
        fuels=fuels,
        company_model_dict=json.dumps(company_model_dict)
    )


if __name__ == '__main__':
    app.run(debug=True)