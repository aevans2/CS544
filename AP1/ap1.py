# imports
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

# read in data 
data = pd.read_csv('http://download1503.mediafire.com/om4ihso3mmlg/x5tc22bwb8daayv/auto-mpg_noblank.csv', index_col=0)

# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars =['displacement', 'horsepower', 'weight', 'acceleration'], y_vars='mpg', size=7, aspect=0.7)

# create fitted models
lm1 = smf.ols(formula='mpg ~ displacement', data=data).fit()
lm2 = smf.ols(formula='mpg ~ horsepower', data=data).fit()
lm3 = smf.ols(formula='mpg ~ weight', data=data).fit()
lm4 = smf.ols(formula='mpg ~ acceleration', data=data).fit()

# print the R-squared value for the model
print("The R-squared value for the model that includes displacement is ",lm1.rsquared)
print("The R-squared value for the model that includes horsepower is ",lm2.rsquared)
print("The R-squared value for the model that includes weight is ",lm3.rsquared)
print("The R-squared value for the model that includes acceleration is ",lm4.rsquared)

# create X and y
feature_cols = ['weight']
X = data[feature_cols]
y = data.mpg

# instantiate and fit
lm5 = smf.ols(formula='mpg ~ displacement', data=data).fit()

# print the coefficient
print("\n")
print("The coefficients for the best predictor is: ",lm5.params)

# create a fitted model with all four features
feature_cols2 = ['displacement', 'horsepower', 'weight', 'acceleration']

# create X and y
X = data[feature_cols2]
y = data.mpg

# instantiate and fit
lm6 = smf.ols(formula='mpg ~ displacement + horsepower + weight + acceleration', data=data).fit()

# print the coefficient
print("\n")
print("The summary including the R-squared value for the model that includes all four predictors:\n",feature_cols2)
print(lm6.summary())

# create a new Series with a dummy variable called cylinders
cylinder_dummies = pd.get_dummies(data.cylinders, prefix='cylinders').iloc[:, 1:]

# concatenate the dummy variable columns onto the DataFrame (axis=0 means rows, axis=1 means columns)
data = pd.concat([data, cylinder_dummies], axis=1)

# create X and y
feature_cols3 = ['displacement', 'horsepower', 'weight', 'acceleration', 'cylinders_4', 'cylinders_5', 'cylinders_6', 'cylinders_8']
X = data[feature_cols3]
y = data.mpg

# instantiate, fit
lm7 = LinearRegression()
lm7.fit(X, y)

# print coefficients
print("\n")
print("The coefficients of the model including cylinders:")
print(feature_cols3)
print(lm7.coef_)