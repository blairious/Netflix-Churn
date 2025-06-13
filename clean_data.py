import pandas as pd
import numpy as np

#Load data and parse 'Last_Login' as datetime
df = pd.read_csv('netflix_users.csv')
df['Last_Login'] = pd.to_datetime(df['Last_Login'], errors='coerce')

#Define churn cutoff date
churnpoint = pd.to_datetime('2024-12-07')

#Creates a 'Subscription_Start' column that is a random date, earlier than 'Last_Login'
df['Subscription_Start'] = (
    df['Last_Login'] 
    - pd.to_timedelta(np.random.randint(30, 365, size=len(df)), unit='d')
    - pd.to_timedelta(df['Watch_Time_Hours'] / 24, unit='d')
).dt.floor('D')

#Add a Loyalty column that is the number of hours user has been subscribed.
df['Loyalty'] = (df['Last_Login'] - df['Subscription_Start']).dt.days * 24

#Create a usage ratio based on last_Login, Subscription_Start, and Hours_Watched
df['Usage_Ratio'] = (df['Watch_Time_Hours'] + 1) / df['Loyalty']


#Create a churn flag of 0 or 1 based on the Last_Login date being before the churnpoint and the Usage_Ratio being less than the netflix average.
df['Churn_Flag'] = ((df['Last_Login'] < churnpoint) & (df['Usage_Ratio'] < 0.13)).astype(int)

#Iterates through user who have Churn_Flag 1 and is from the USA, UK, or Japan, this gives them a 75% chance of having Churn_Flag 0.
mask = (df['Churn_Flag'] == 1) & (df['Country'].isin(['USA', 'UK', 'Japan']))
df.loc[mask, 'Churn_Flag'] = np.random.choice([0, 1], size=mask.sum(), p=[0.75, 0.25])

#Saves updated data
df.to_csv('netflix_users.csv', index=False)