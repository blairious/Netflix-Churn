#Class that trains model on data.
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score


#Loads dataset into dataframe
df = pd.read_csv('netflix_users.csv')

#Converts churn to boolean
df['Churn_Flag'] = df['Churn_Flag'].astype(bool)

#Converts categorical columns to pandas 'category' dtype
cat_cols = ['Country', 'Subscription_Type', 'Favorite_Genre']
df[cat_cols] = df[cat_cols].astype('category')


#Sets churn as target and establishes training variables
Train = df[['Age', 'Country', 'Subscription_Type', 'Favorite_Genre', 'Watch_Time_Hours', 'Loyalty', 'Usage_Ratio']]
Target = df['Churn_Flag']

#Splits data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(Train, Target, test_size=0.2, random_state=33)

#Defines hyperparameter grid.
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 25, 30],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'alpha': [0, 0.5, 1],         
    'reg_lambda': [1, 2],    
    'scale_pos_weight': [1, 2, 3]
}

#Initializes model.
base_model = XGBClassifier(
    enable_categorical=True,
    random_state=33,
    eval_metric='logloss'
)

#Randomized search with 5-fold cross-validation.
rs = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=100,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

#Fit randomized search.
rs.fit(X_train, y_train)

#Best model chosen from search.
model = rs.best_estimator_

#Makes predictions.
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

#Evaluates model.
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print("Best Parameters:", rs.best_params_)
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC Score: {roc:.2f}")

#Saves tuned model for use in the main program.
model.save_model('Churn_Predict.json')