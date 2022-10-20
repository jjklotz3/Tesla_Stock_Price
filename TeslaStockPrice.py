import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV

#ignore warnings
warnings.filterwarnings('ignore')
print('-'*25)

#Load the Data into File and make the entire datafram visible in the terminal
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199
tesla_price = pd.read_csv("TSLA.csv")

#print(tesla_price.head())
#print(tesla_price.isnull().sum())
tesla_price['Open/Close Difference'] = tesla_price['Open'].sub(tesla_price['Close'], axis = 0)
tesla_price['Average Price'] = tesla_price[['Open','Close','High','Low']].mean(axis=1)
#print(tesla_price.head())

tesla_price['Date'] = pd.to_datetime(tesla_price['Date'], errors='coerce') #The errors paramter gets rid of the parsing error if the datetime format is in string format instead of numbers (i.e "Jan")



#Deal with Posted on dates. Transform the 'Posted' on column into single integers as we cant use one hot encoding for a column with so much variability
tesla_price['Month'] = tesla_price['Date'].dt.month
tesla_price['Day'] = tesla_price['Date'].dt.day
tesla_price['Year'] = tesla_price['Date'].dt.year

tesla_price = tesla_price.drop(['Date'], axis=1)
tesla_price = tesla_price.dropna()
tesla_price = tesla_price.drop_duplicates()

#print(tesla_price.head())
#print(tesla_price.info())

nums = tesla_price.select_dtypes(exclude="object").columns
nums = nums.delete(7)
#print(nums)

target = 'Average Price'
X = tesla_price.loc[:,tesla_price.columns!=target]
y = tesla_price.loc[:,tesla_price.columns==target]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 0)

print(X_train.shape)
print(y_train.shape)
sc = StandardScaler()
X_train[nums] = sc.fit_transform(X_train[nums])                 #THE TARGET VALUE NEEDS TO BE LEFT OUT IF IT IS A NUMERICAL VALUE OTHERWISE IT WILL NOT WORK
X_test[nums] = sc.transform(X_test[nums])


models = {
    "Linear regression":LinearRegression(),
    "Lasso ":LassoCV(),
    "Ridge":RidgeCV(),
    "ElasticNet":ElasticNetCV(),
    "RandomForest": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(),
}

Results = {
    "Model":[],
    "Train Score":[],
    "Test Score":[],
    "RMSE":[]
}

for name, model in models.items():
    model.fit(X_train,np.log(y_train))
    train_s = model.score(X_train,np.log(y_train))
    test_s = model.score(X_test,np.log(y_test))
    predictions = model.predict(X_test)
    RMSE = mean_squared_error((predictions),np.log(y_test))
    Results["Model"].append(name)
    Results["Train Score"].append(train_s)
    Results["Test Score"].append(test_s)
    Results["RMSE"].append(RMSE)
    print("Model: " , name)
    print("Train Score: " , train_s)
    print("Test Score : " , test_s)
    print("RMSE : " , round(RMSE,2))
    print("===========================")

#We determined that the random tree forest is the best model to go with so we continue with this model

    

