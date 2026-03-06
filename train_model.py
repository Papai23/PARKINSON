import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv("parkinsons.csv")

X = data.drop(["name","status"], axis=1)
y = data["status"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = XGBClassifier()

model.fit(X_train,y_train)

joblib.dump(model,"xgboost_model.pkl")

print("Model trained and saved")