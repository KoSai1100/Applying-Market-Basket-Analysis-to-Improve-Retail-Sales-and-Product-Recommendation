

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def run(self, input_data):
        # 1. Aggregate customer features
        customer_df = input_data.groupby('Customer_ID').agg({
            'Customer_Age': 'first',
            'Transaction_ID': 'count',  # transaction frequency
            'Product_Category': lambda x: x.mode()[0],  # most frequent category
            'Total_Price': 'mean'  # average transaction amount
        }).rename(columns={
            'Transaction_ID': 'transaction_frequency',
            'Total_Price': 'avg_transaction_amount'
        })

        # Encode product category
        customer_df['Product_Category'] = LabelEncoder().fit_transform(customer_df['Product_Category'])

        # 2. Feature scaling
        scaler = StandardScaler()
        features = ['Customer_Age', 'transaction_frequency', 'Product_Category', 'avg_transaction_amount']
        X_scaled = scaler.fit_transform(customer_df[features])

        # 3. KMeans clustering for segmentation
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        customer_df['segment'] = kmeans.fit_predict(X_scaled)

        # 4. Silhouette score
        sil_score = silhouette_score(X_scaled, customer_df['segment'])

        # 5. Supervised model: Predict segment from features
        X = customer_df[features]
        y = customer_df['segment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        return {
            "accuracy": accuracy,
            "silhouette_score": sil_score,
            "classification_report": class_report,
            "feature_importances": dict(zip(features, self.model.feature_importances_))
        }

    @staticmethod
    def get_metrics():
        return {"info": "RandomForest metrics placeholder"}