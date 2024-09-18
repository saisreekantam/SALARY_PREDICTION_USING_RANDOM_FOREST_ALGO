# File: app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Streamlit app title
st.title("Software Engineer Salary Predictor")

# Upload CSV file through Streamlit
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Check if necessary columns exist
    required_columns = ['education', 'country', 'years_of_experience', 'salary']
    if not all(col in df.columns for col in required_columns):
        st.error(f"CSV file must contain the following columns: {', '.join(required_columns)}")
    else:
        # Preprocess the data
        le_education = LabelEncoder()
        le_country = LabelEncoder()

        df['education'] = le_education.fit_transform(df['education'])
        df['country'] = le_country.fit_transform(df['country'])

        # Features and target
        X = df[['education', 'country', 'years_of_experience']]
        y = df['salary']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict on the test set to evaluate model performance
        y_pred = model.predict(X_test)
        error = np.sqrt(mean_squared_error(y_test, y_pred))

        # Input form for user data
        st.write("Enter the following details to predict your salary:")
        education = st.selectbox("Education Level", le_education.classes_)
        country = st.selectbox("Country", le_country.classes_)
        years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=1)

        # Encode inputs
        input_data = np.array([[le_education.transform([education])[0],
                                le_country.transform([country])[0],
                                years_of_experience]])

        # Predict salary based on user inputs
        if st.button("Predict Salary"):
            prediction = model.predict(input_data)
            st.write(f"Predicted Salary: ${prediction[0]:,.2f}")
            st.write(f"Model Error (RMSE): ${error:,.2f}")


