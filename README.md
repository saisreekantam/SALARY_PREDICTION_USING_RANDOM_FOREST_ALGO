# Software Engineer Salary Predictor

This project is a web application built using [Streamlit](https://streamlit.io/) to predict software engineer salaries based on user inputs. It uses machine learning techniques, specifically a RandomForestRegressor, to make salary predictions based on features such as education, country, and years of experience.

## Features

- Upload a CSV file with salary data to train the model.
- Display a preview of the dataset.
- Train the model using a RandomForestRegressor.
- Predict salary based on user inputs.
- Display the model's Root Mean Squared Error (RMSE) for evaluation.
  
## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/saisreekantam/SALARY_PREDICTION_USING_RANDOMFOREST_ALGO.git
   cd SALARY_PREDICTION_USING_RANDOMFOREST_ALGO
## Upload a dataset:
You will be prompted to upload a CSV file. The file must include the following columns:

-education
-country
-years_of_experience
-salary

## Predict Salary:
After the model is trained on the dataset, you can enter your education level, country, and years of experience to get a salary prediction.

## Usage
Upload the dataset:
Click the "Upload your dataset (CSV file)" button to upload a CSV file that includes salary information.

Check the dataset preview:
Once the dataset is uploaded, a preview of the first few rows of the data is displayed.

Input details for salary prediction:
Enter your education level, country, and years of experience using the interactive form provided.

Predict salary:
Click the "Predict Salary" button to get your predicted salary based on the RandomForestRegressor model. The Root Mean Squared Error (RMSE) will also be displayed to provide an estimate of the model’s prediction accuracy.


