# 🚀 Spaceship Titanic Prediction API

This project provides an end-to-end machine learning solution for the Kaggle **"Spaceship Titanic"** competition. It includes a complete pipeline for data preprocessing, advanced feature engineering, model training, and deployment as a containerized web API using **Flask** and **Docker**.

---

## ✨ Features

- **Advanced Feature Engineering**  
  Creation of features like `GroupSize`, `TotalSpend`, and `Deck` from raw data to improve model accuracy.

- **Robust Modeling**  
  Uses a tuned `XGBClassifier` model, a high-performance gradient boosting algorithm.

- **Prediction API**  
  A simple Flask API to serve real-time predictions.

- **Containerized Application**  
  Packaged with Docker for portability and easy deployment.

- **CI/CD Automation**  
  A basic GitHub Actions workflow is included to automatically build the Docker image on code changes.

---

## ⚙️ Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/c21051997/Spaceship-Titanic-Deployed.git
cd spaceship-titanic-project
```
### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🧠 Train the Model
Run the training script to process raw data from the data/ folder and save the trained model and scaler artifacts into the model/ directory.

```bash
python model/train.py
```

### 🐳 Build the Docker Image

```bash
docker build -t titanic-spaceship-app .
```

### 🌐 Run the Docker Container

```bash
docker run -p 8000:8000 titanic-spaceship-app
```

---

## 🧪 Testing the API
Once the container is running, you can test the /predict endpoint.

### 1. Send a request using curl:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  --data @payload.json \
  http://localhost:8000/predict
```

### ✅ Expected Response:
```json
{
  "transported": false
}
```

--- 

## ✨ Features

- Ensure Docker is installed and running on your system.

- For consistent model results, maintain the same preprocessing and feature engineering steps in both training and prediction phases.

- Extend the API or model logic as needed for custom use cases or additional features.