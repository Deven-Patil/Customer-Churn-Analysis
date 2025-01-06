import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

# Load dataset and model
df_1 = pd.read_csv("first_telc.csv")
model = pickle.load(open("model.sav", "rb"))

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    # Collect form inputs
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']
    inputQuery14 = request.form['query14']
    inputQuery15 = request.form['query15']
    inputQuery16 = request.form['query16']
    inputQuery17 = request.form['query17']
    inputQuery18 = request.form['query18']
    inputQuery19 = request.form['query19']
    
    # Create a new data row from the form inputs
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7,
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]

    # Create a dataframe with proper column names
    new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
                                         'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                         'PaymentMethod', 'tenure'])

    # Concatenate the new input with the original dataframe
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Handle non-numeric and missing 'tenure' values, filling NaNs with the median
    df_2['tenure'] = pd.to_numeric(df_2['tenure'], errors='coerce')
    if df_2['tenure'].isna().sum() > 0:
        print("Warning: NaN values detected in 'tenure' column before filling")
        print(df_2['tenure'][df_2['tenure'].isna()])
    
    df_2['tenure'].fillna(df_2['tenure'].median(), inplace=True)
    df_2['tenure'] = df_2['tenure'].astype(int, errors='ignore')
    
    # Check for remaining NaN values after filling
    if df_2['tenure'].isna().sum() > 0:
        print("Error: NaN values detected in 'tenure' column after filling")
        print(df_2['tenure'][df_2['tenure'].isna()])
        return render_template('home.html', output1="Error: Invalid input for tenure", output2="", query1=inputQuery1, query2=inputQuery2, query3=inputQuery3, query4=inputQuery4, query5=inputQuery5, query6=inputQuery6, query7=inputQuery7, query8=inputQuery8, query9=inputQuery9, query10=inputQuery10, query11=inputQuery11, query12=inputQuery12, query13=inputQuery13, query14=inputQuery14, query15=inputQuery15, query16=inputQuery16, query17=inputQuery17, query18=inputQuery18, query19=inputQuery19)

    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2['tenure'], range(1, 80, 12), right=False, labels=labels)

    # Drop the 'tenure' column as it's no longer needed
    df_2.drop(columns=['tenure'], axis=1, inplace=True)

    # One-hot encode categorical variables
    new_df_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                          'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                          'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

    # Align new_df_dummies to the model's input shape (handling missing columns)
    missing_cols = set(model.feature_names_in_) - set(new_df_dummies.columns)
    for c in missing_cols:
        new_df_dummies[c] = 0
    new_df_dummies = new_df_dummies[model.feature_names_in_]

    # Predict churn and probability
    single = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]

    # Display results
    if single == 1:
        o1 = "This Customer is likely to Churn !"
        o2 = f"Confidence: {probability[0] * 100:.2f}%"
    else:
        o1 = "This Customer is likely to Stay with Us "
        o2 = f"Confidence: {probability[0] * 100:.2f}%"

    # Render results in the template
    return render_template('home.html', output1=o1, output2=o2,
                           query1=request.form['query1'],
                           query2=request.form['query2'],
                           query3=request.form['query3'],
                           query4=request.form['query4'],
                           query5=request.form['query5'],
                           query6=request.form['query6'],
                           query7=request.form['query7'],
                           query8=request.form['query8'],
                           query9=request.form['query9'],
                           query10=request.form['query10'],
                           query11=request.form['query11'],
                           query12=request.form['query12'],
                           query13=request.form['query13'],
                           query14=request.form['query14'],
                           query15=request.form['query15'],
                           query16=request.form['query16'],
                           query17=request.form['query17'],
                           query18=request.form['query18'],
                           query19=request.form['query19'])

if __name__ == "__main__":
    app.run(debug=True)


