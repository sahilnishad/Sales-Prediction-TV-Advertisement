"""
We are trying to predict the sales of a product given a particular amount of money spent on TV advertisemnet
"""


#importing necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

#loading dataset into dataframe 'df'
df = pd.read_csv('data.csv')

#dividing the input dataset into dependent and independent variable
X= df.drop('Sales',axis=1)
y = df['Sales']

#creating and fitting linear regression model
model = LinearRegression()
model.fit(X, y)

#saving the trained model into a pickle file
pickle.dump(model, open('model.pkl','wb'))