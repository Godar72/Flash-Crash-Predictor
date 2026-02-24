# Predictive Modeling of Flash Crashes in Cryptocurrency Market using GRU 📈🤖

An early warning anomaly detection system built to predict flash crashes and extreme volatility events in high frequency cryptocurrency markets.

Developed as an academic research project at the Symbiosis Institute of Technology, Department of Artificial Intelligence and Machine Learning, this system uses a Gated Recurrent Unit GRU neural network to process streaming time series data and detect pre crash volatility clusters before a severe price drop occurs.

---

## 🎯 Project Objective

Traditional financial models focus on forecasting future prices using regression techniques. This project reframes market prediction as a binary classification problem.

The goal is to classify market conditions into:

- Normal market state  
- High risk crash state  

By predicting the probability of a crash in real time, the system provides a critical reaction window for algorithmic trading systems to:

- Withdraw liquidity  
- Hedge open positions  
- Reduce exposure  

---

## 🛠️ Tech Stack and Architecture

- Language: Python 3.x  
- Deep Learning Framework: TensorFlow and Keras  
- Data Processing: Pandas, NumPy, Scikit Learn  
- Model Architecture: Gated Recurrent Unit GRU  
- Key Techniques:
  - Sliding window sequence generation with 500 tick memory  
  - Cost sensitive learning using class weighting  
  - Early stopping for overfitting prevention  

---

## 📂 Repository Structure

- `BTCUSDT_labeled.csv`  
  High frequency cryptocurrency dataset containing engineered price features and binary crash labels.  
  Ensure this file is placed in the root directory before execution.

- `prepare_sequences.py`  
  Data preprocessing pipeline.  
  - Applies MinMaxScaler  
  - Generates 3D sequential arrays  
  - Saves `X.npy`, `y.npy`, and `scaler.pkl`

- `train_model.ipynb`  
  Main Jupyter Notebook for:
  - Model training  
  - Evaluation  
  - Visualization  

- `scaler.pkl`  
  Saved Scikit Learn scaler for transforming future real time input data.

- `flash_crash_model.h5`  
  Trained GRU model weights generated after training.

---

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Godar72/Flash-Crash-Predictor.git
cd Flash-Crash-Predictor
```

### 2. Install Required Dependencies

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

---

## 💻 Usage Instructions

The pipeline runs in two stages.

### Stage 1: Data Preprocessing

Convert raw CSV data into sequential arrays.

```bash
python prepare_sequences.py
```

Output files generated:

- `X.npy`  
- `y.npy`  
- `scaler.pkl`  

The script uses a sliding window approach. It feeds the previous 500 price ticks to predict the current market state.

---

### Stage 2: Model Training and Evaluation

Open the Jupyter Notebook.

```bash
jupyter notebook train_model.ipynb
```

Run all cells. The notebook will:

- Load preprocessed sequences  
- Apply `compute_class_weight` to address severe class imbalance  
- Train the GRU model with EarlyStopping  
- Generate a Confusion Matrix  
- Produce a Classification Report  
- Evaluate recall on crash events  

---

## 🧠 Methodology Highlights

### Why GRU instead of LSTM

GRUs use a two gate structure:

- Reset gate  
- Update gate  

This architecture reduces computational overhead compared to LSTM. It improves training speed and lowers inference latency, which is critical for high frequency trading systems that require near real time predictions.

---

### Handling Class Imbalance

Financial datasets are heavily skewed toward normal states. Crash events may represent less than 0.1 percent of observations.

To address this:

- Cost sensitive learning is applied  
- Higher penalty is assigned to misclassifying crash events  
- Class weighting prioritizes recall for Class 1  

This ensures the model does not default to predicting only the majority class.

---

## 👥 Authors

- Aditya Godar  
- Kavan Patel  
- Abhay Raj Mosaita  
- Tanay Singh  
