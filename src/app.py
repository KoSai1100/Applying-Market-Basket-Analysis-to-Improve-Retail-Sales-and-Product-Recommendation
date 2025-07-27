import pandas as pd
from flask import Flask, render_template, request
from models.random_forest import RandomForestModel
from models.lstm import LSTMModel
from models.apriori import AprioriModel
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# --- Move these functions up here ---
def plot_feature_importance(importances):
    plt.figure(figsize=(6,4))
    pd.Series(importances).plot(kind='bar')
    plt.title('Feature Importances')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

def plot_lstm_predictions(y_true, y_pred):
    plt.figure(figsize=(8,4))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('LSTM: Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

def plot_apriori_lift(rules):
    top_rules = rules.head(10).copy()

    # Format labels cleanly: {'Milk'} → {'Bread'} → Milk → Bread
    top_rules['rule'] = top_rules.apply(
        lambda row: f"{', '.join(sorted([item for item in row['antecedents']]))} → {', '.join(sorted([item for item in row['consequents']]))}",
        axis=1
    )

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_rules['rule'], top_rules['lift'], color='skyblue')
    plt.xlabel("Lift")
    plt.title("Top 10 Association Rules by Lift")
    plt.gca().invert_yaxis()  # Highest lift at the top

    # Add lift values to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')

    plt.tight_layout()

    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_models', methods=['POST'])
def run_models():
    analysis_type = request.form['analysis_type']
    file = request.files['data_file']
    input_data = pd.read_csv(file)

    rf_results = lstm_results = apriori_results = None
    rf_plot = lstm_plot = apriori_plot = None
    forecast_plot = loss_curve_plot = None
    if analysis_type == "random_forest":
        rf_model = RandomForestModel()
        rf_results = rf_model.run(input_data)
        if 'feature_importances' in rf_results:
            rf_plot = plot_feature_importance(rf_results['feature_importances'])

    elif analysis_type == "lstm":
        lstm_model = LSTMModel()
        lstm_results = lstm_model.run(input_data)
        forecast_plot = lstm_results.get('forecast_plot')
        loss_curve_plot = lstm_results.get('loss_curve_plot')
        if 'predictions' in lstm_results and 'y_true' in lstm_results:
            lstm_plot = plot_lstm_predictions(lstm_results['y_true'], lstm_results['predictions'])

    elif analysis_type == "apriori":
        apriori_model = AprioriModel()
        apriori_results = apriori_model.run(input_data)
        if 'rules_df' in apriori_results:
            apriori_plot = plot_apriori_lift(apriori_results['rules_df'])

    return render_template(
        'results.html',
        rf_results=rf_results,
        lstm_results=lstm_results,
        apriori_results=apriori_results,
        forecast_plot=forecast_plot,
        loss_curve_plot=loss_curve_plot,
        rf_plot=rf_plot,
        lstm_plot=lstm_plot,
        apriori_plot=apriori_plot
    )

@app.route('/comparison')
def comparison():
    # Get model comparison metrics
    rf_metrics = RandomForestModel.get_metrics()
    lstm_metrics = LSTMModel.get_metrics()
    apriori_metrics = AprioriModel.get_metrics()
    
    return render_template('comparison.html', rf_metrics=rf_metrics, lstm_metrics=lstm_metrics, apriori_metrics=apriori_metrics)

if __name__ == '__main__':
    app.run(debug=True)

