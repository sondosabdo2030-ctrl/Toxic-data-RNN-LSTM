# Toxic Data Classification using RNN & LSTM

## 📋 About the Project
This project implements a Deep Learning model using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks in Keras/TensorFlow to classify and detect toxic text data.

## ⚙️ How it Works
1. **Data Preparation:** Loads and processes the dataset (`cellula toxic data (1).csv`).
2. **Model Architecture:** Uses a Sequential model combining Embedding, LSTM, and Dense layers to learn text sequences and patterns.
3. **Training & Prediction:** Trains the model on the training data and evaluates it on the test set (`X_test`) using `model.predict` and `np.argmax`.
4. **Evaluation:** Generates a detailed performance classification report including Precision, Recall, F1-Score, and Accuracy using `scikit-learn`classification report.

## 📁 Files in Repository
* `RNN.ipynb`: Jupyter notebook containing the RNN model implementation.
* `LSTM.ipynb`: Jupyter notebook containing the LSTM model implementation.
* `best_lstm_model.keras`: Saved trained Keras model weights and configuration.
* `cellula toxic data (1).csv`: The dataset used for the project.
* `requirements.txt`: List of required Python libraries.
