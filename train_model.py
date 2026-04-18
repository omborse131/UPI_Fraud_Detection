import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# load dataset
data = pd.read_csv("fraud_dataset.csv")

# select features
data = data[['amount','session_duration','authentication_attempts',
             'transaction_velocity','failed_transaction_count',
             'transaction_type','is_fraud']]

# convert categorical column
data['transaction_type'] = data['transaction_type'].astype('category').cat.codes

# split features and target
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# 🔹 ADD THIS LINE HERE
print(X.columns)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# save model
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Model trained successfully")