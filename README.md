# 🔧 Remaining Useful Life (RUL) Prediction Dashboard

An end-to-end machine learning project for predicting **Remaining Useful Life (RUL)** of industrial systems using **physics-informed feature engineering** and **sensor data**.

The project includes data preprocessing, feature engineering, model training, and an interactive **Streamlit dashboard** for real-time predictions and visualization.

---

## 🚀 Overview

This project aims to estimate how much useful life a system has left before failure by leveraging:

* Sensor data from drilling/industrial operations
* Physics-based feature engineering
* Machine learning models (XGBoost)
* Interactive visualization for monitoring and analysis

The final output is a **deployable web app** where users can upload sensor data and get RUL predictions instantly.

---

## 🧠 Key Features

* 📊 **RUL Prediction Model**
  Trained using engineered features capturing system stress, damage, and operational behavior.

* ⚙️ **Physics-Informed Features**
  Includes derived variables such as damage accumulation (`D`), degradation rate (`d_dot`), and stress indicators.

* 📂 **CSV Upload Interface**
  Users can upload raw sensor data directly into the app.

* 📉 **Interactive Visualizations (Plotly)**

  * RUL trend over time
  * Feature trend explorer
  * Dynamic filtering

* 🕒 **Time-Based Filtering**
  Analyze specific operational windows using datetime selection.

* 📥 **Export Results**
  Download predicted RUL values as a CSV file.

---

## 🏗️ Project Structure

```bash
.
├── RUL3.0.py              # Streamlit app
├── rul_model.pkl          # Trained ML model
├── requirements.txt       # Dependencies
├── .streamlit/
│   └── config.toml        # App configuration (upload limits, etc.)
```

---

## ⚙️ Installation (Local)

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m streamlit run RUL3.0.py
```

---

## 🌐 Deployment

This app is designed to run on **Streamlit Cloud**.

Make sure the following are included in your repo:

* `requirements.txt`
* `rul_model.pkl`
* `.streamlit/config.toml`

---

## 📊 How It Works

1. Upload a CSV file containing sensor data

2. The app:

   * Cleans and preprocesses the data
   * Applies feature engineering
   * Loads the trained model
   * Predicts RUL

3. Results are displayed as:

   * Time-series plots
   * Summary metrics
   * Downloadable predictions

---

## ⚠️ Requirements for Input Data

Your CSV must contain all required feature columns used during training.

If any are missing, the app will notify you.

---

## 🧪 Model

* Model Type: **XGBoost Regressor**
* Target: **Remaining Useful Life (RUL)**
* Training approach:

  * Physics-informed feature engineering
  * Supervised learning

---

## 📌 Key Insights

* RUL is primarily driven by **damage accumulation**
* Operational conditions (pressure, torque, RPM) influence **rate of degradation**
* System behavior is **non-linear and multi-regime**

---

## 📈 Future Improvements

* Real-time data streaming support
* Model retraining pipeline
* Advanced anomaly detection
* Cloud storage integration for large datasets

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.


---

## 👤 Author

**TEEJAY XP**
