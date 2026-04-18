# UPI_Fraud_Detection
The UPI Fraud Detection System is a machine learning–based application that identifies fraudulent UPI transactions by analyzing transaction data patterns. Using a trained Random Forest model and a Flask web interface, the system predicts whether a transaction is genuine or fraudulent, helping enhance financial security in digital payment systems.
🎯 Features
Machine learning model for fraud prediction
Real-time transaction analysis
Web interface for user input and results
Model persistence using Joblib
Explainability support using SHAP (optional)
Lightweight and easy to run locally
🧠 Machine Learning Model
Algorithm: Random Forest Classifier (from scikit-learn)
Trained on historical UPI transaction dataset
Handles imbalanced data
Saved model: model.pkl
Saved feature columns: model_columns.pkl
🛠️ Tech Stack
Backend: Flask
ML Library: scikit-learn
Model Serialization: Joblib
Data Handling: pandas
Explainability (optional): SHAP
Frontend: HTML, CSS

Project Structure
upi-fraud-detection/
│
├── app.py
├── model.pkl
├── model_columns.pkl
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── requirements.txt
└── README.md

Clone the Repository
git clone <your-repo-link>
cd upi-fraud-detection

Create Virtual Environment
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
python app.py

Open in browser:

http://127.0.0.1:5000/
