import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm

data = pd.read_csv('Analyses/User Data/Clustered/usersUK_nr8.csv')
data['account_creation_date'] = pd.to_datetime(data['account_creation_date'], unit='s')


reference_date = data['account_creation_date'].min() # Use earliest date as reference 
data['days_since_creation'] = (data['account_creation_date'] - reference_date).dt.days
print(data['days_since_creation'].describe())


X = data['days_since_creation']
y = data['distance_to_center']

# Add a constant to the independent variable 
X = sm.add_constant(X)
model = sm.OLS(y, X).fit() # Fit regression
print(model.summary())
