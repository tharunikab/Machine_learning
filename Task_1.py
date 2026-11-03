import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error

diabetes=load_diabetes()
X=pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
y=diabetes.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
print("\n **DATA LOADED AND SPLIT SUCCESSFULLY**\n")
lr=LinearRegression()
lr.fit(X_train,y_train)

linear_coefficients=pd.Series(lr.coef_,index=X.columns)
print("TOP FIVE FEATURES BY COEFFICIENT STRENGTH(LINEAR REGRESSION):")

ridge=Ridge(alpha=1.0)
ridge.fit(X_train,y_train)

lasso=Lasso(alpha=0.1)
lasso.fit(X_train,y_train)

rmse_lr=np.sqrt(mean_squared_error(y_test,lr.predict(X_test)))
rmse_ridge=np.sqrt(mean_squared_error(y_test,ridge.predict(X_test)))
rmse_lasso=np.sqrt(mean_squared_error(y_test,lasso.predict(X_test)))

zeros_lr=np.sum(lr.coef_==0)
zeros_ridge=np.sum(ridge.coef_==0)
zeros_lasso=np.sum(np.isclose(lasso.coef_,0.0,atol=1e-5))

results_df=pd.DataFrame({
    "Model":["Linear Regression","Ridge(alpha=1.0)","Lasso(alpha=0.1)"],
    "TestRMSE(Lower is Better)":[rmse_lr,rmse_ridge,rmse_lasso],
    "Zeroed-out Coefficients":[zeros_lr,zeros_ridge,zeros_lasso]
})

print("FINAL SUBMISSION TABLE:")
print(results_df.to_string(index=False))
print("\n")
