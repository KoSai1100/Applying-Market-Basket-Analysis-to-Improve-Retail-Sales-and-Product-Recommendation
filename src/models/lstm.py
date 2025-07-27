 # Ensure TensorFlow is installed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import io
import base64

def plot_forecast(y_true, y_pred):
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('LSTM Forecast: Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

def plot_loss_curve(history):
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

def prepare_data(sales_df, seq_length=30):
    sales_df = sales_df.sort_values('Date')
    features = ['Total_Price', 'Month', 'DayOfWeek', 'IsWeekend', 'RollingMean7']
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(sales_df[features])

    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i + seq_length])
        y.append(data_scaled[i + seq_length][0])  # Predict Total_Price only
    return np.array(X), np.array(y), scaler

def build_model(input_shape, lstm_units=64, dropout_rate=0.2, dense_units=32, learning_rate=0.001):
    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape, return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, patience=10):
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )
    return history

def predict(model, X):
    y_pred_scaled = model.predict(X).flatten()
    return y_pred_scaled

def inverse_transform(scaled, scaler, col_idx=0):
    dummy = np.zeros((len(scaled), scaler.scale_.shape[0]))
    dummy[:, col_idx] = scaled
    return scaler.inverse_transform(dummy)[:, col_idx]

class LSTMModel:
    def __init__(self):
        self.model = None
        self.scaler = None

    def run(self, sales_df):

        # Feature engineering if columns are missing
        if 'Month' not in sales_df.columns:
            sales_df['Month'] = pd.to_datetime(sales_df['Date']).dt.month
        if 'DayOfWeek' not in sales_df.columns:
            sales_df['DayOfWeek'] = pd.to_datetime(sales_df['Date']).dt.dayofweek
        if 'IsWeekend' not in sales_df.columns:
            sales_df['IsWeekend'] = sales_df['DayOfWeek'].isin([5, 6]).astype(int)
        if 'RollingMean7' not in sales_df.columns:
            sales_df['RollingMean7'] = sales_df['Total_Price'].rolling(window=7, min_periods=1).mean()

        X, y, scaler = prepare_data(sales_df)
        self.scaler = scaler

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.model = build_model(X_train.shape[1:])
        history = train_model(self.model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, patience=3)

        y_pred_scaled = predict(self.model, X_val)
        y_pred = inverse_transform(y_pred_scaled, scaler)
        y_true = inverse_transform(y_val, scaler)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        forecast_plot = plot_forecast(y_true, y_pred)
        loss_curve_plot = plot_loss_curve(history)
        return {
            "mse": mse,
            "mae": mae,
            "predictions": y_pred.tolist(),
            "y_true": y_true.tolist(),
            "forecast_plot": forecast_plot,
            "loss_curve_plot": loss_curve_plot
        }
        
    @staticmethod
    def get_metrics():
        return {"example_metric": 0.95}