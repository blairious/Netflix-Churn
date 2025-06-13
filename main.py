import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
# Ensure matplotlib is set to use the 'agg' backend for Streamlit compatibility

# Load trained model
model = XGBClassifier(enable_categorical=True)
model.load_model('Churn_Predict.json')

# Method to predict churn
def model_parse(dataframe):
    prediction = model.predict(dataframe)
    return 'highly likely' if prediction[0] == 1 else 'NOT likely'

# Method to generate suggestion from top contributing feature
def SuggestionCreate(dataframe):
    explainer = shap.Explainer(model)  # Use Explainer directly with XGBClassifier
    shap_values = explainer(dataframe)

    shap_df = pd.DataFrame({
        'feature': dataframe.columns,
        'shap_value': shap_values.values[0]
    }).sort_values(by='shap_value', ascending=False)

    main_factor = shap_df.iloc[0]['feature']

    if main_factor == 'Age':
        return 'Push targeted advertising based on user age demographic.'
    elif main_factor == 'Country':
        return 'Push targeted advertising based on national trends in user country.'
    elif main_factor == 'Subscription_Type':
        return 'Offer limited discount on subscription.'
    elif main_factor == 'Usage_Ratio' or 'Watch_Time_Hours':
        return 'Push targeted advertising based on user interests.'
    else:
        return 'Continue normal engagement.'

# Streamlit UI initialized:
st.title('NETFLIX')
st.header('Retention Helper')
st.divider()
st.text('Enter User information to get retention strategy:')

# Inputs in UI.
CName = st.text_input('Customer Name:')
CAge = st.number_input('Age:', step=1)
Country = st.selectbox('Country:',('USA', 'UK', 'Canada', 'France', 'Mexico', 'Japan', 'Australia', 'Germany', 'Brazil', 'India'))
SubType = st.selectbox('Subscription Level:', ('Premium', 'Basic', 'Standard'))
FavoriteGenre = st.selectbox('Favorite Genre:', ('Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Documentary', 'Romance'))
StartDate = st.date_input('Subscription Start Date:')
LastLogin = st.date_input('Last Login Date:')
WatchTime = st.number_input('Hours Watched:', step=0.1)

st.divider()

if st.button('Predict'):
    # Creates DataFrame with correct dtype.
    df = pd.DataFrame([{
        'Age': int(CAge),
        'Country': Country,
        'Subscription_Type': SubType,
        'Favorite_Genre': FavoriteGenre,
        'Watch_Time_Hours': WatchTime,
        'Loyalty': (LastLogin - StartDate).days,  # Loyalty in days
        'Usage_Ratio': 0.0,  # Placeholder, will be calculated later
        }])

# Matches categories used during training exactly.
    df['Country'] = pd.Categorical(df['Country'], categories=['France', 'USA', 'UK', 'Canada', 'Mexico', 'Japan', 'Australia', 'Germany', 'Brazil', 'India'])
    df['Subscription_Type'] = pd.Categorical(df['Subscription_Type'], categories=['Premium', 'Basic', 'Standard'])
    df['Favorite_Genre'] = pd.Categorical(df['Favorite_Genre'], categories=['Drama', 'Sci-Fi', 'Comedy', 'Documentary', 'Romance', 'Action', 'Horror'])

# Converts to category dtype.
    df['Country'] = df['Country'].astype('category')
    df['Subscription_Type'] = df['Subscription_Type'].astype('category')
    df['Favorite_Genre'] = df['Favorite_Genre'].astype('category')

# Ensures other columns have correct numeric types.
    df['Age'] = df['Age'].astype(int)
    df['Watch_Time_Hours'] = df['Watch_Time_Hours'].astype(float)

    df['Usage_Ratio'] = df['Watch_Time_Hours'] / (LastLogin - StartDate).days

# Runs prediction and suggestion.
    PreResult = model_parse(df)
    if PreResult != 'NOT likely':
        SugResult = SuggestionCreate(df)
    else:
        SugResult = 'Continue to offer great service!'

    st.write('Based on the input data:')
    st.write(f'{CName} is {PreResult} to drop their membership.')
    st.write(f'Suggested action for user is: {SugResult}')

    #Plot SHAP values on the strealit page
    shap_values = shap.Explainer(model)(df)
    shap.plots.bar(shap_values, max_display=10)
    st.pyplot(plt.gcf()) 