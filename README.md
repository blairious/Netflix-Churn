# Netflix_Churn
Practice AI Model training that takes synthetic user information and projects whether or not the user is likely to leave the platform.

## Main Program - main.py

This is the main program to run the prediction software. It uses a Streamlit web ui to take in a single user's data and uses the model to predict whether or not a user will leave the platform. This program will also show suggested measures to prevent the user from leaving, as well as a graph outlining the top contributing factors to its decision based on SHAP values.

## Training Script - model_training.py

This script handle model training, based on the available dataset. This script uses XGClassifier to train. It also outputs scores for model accuracy, based on F1 and ROC-AUC.

## Data Cleaning Script - clean_data.py

This script is used to fill in gaps in the existing user dataset. It is set to create needed data sythetically that emulates real-world statistics. For the sake of experimentation, this script can be used to alter data at any time.


