# Applying Market Basket Analysis to Improve Retail Sales and Product Recommendation

## Executive Summary

This project focuses on improving retail sales performance and enhancing product recommendation systems through the integration of Market Basket Analysis (MBA) and machine learning techniques. Using a synthetic retail dataset containing 50,000 transaction records and 20 variables from 2020 to 2023, the project applies the Apriori algorithm, Random Forest, and Long Short-Term Memory (LSTM) models to uncover customer purchasing patterns, segment customers, and forecast future sales trends.

The project aims to help retailers make data-driven decisions by identifying frequently purchased product combinations, improving cross-selling strategies, optimizing inventory management, and delivering personalized marketing campaigns.

## Business Problem

Retail businesses generate massive amounts of transactional and customer data daily, but many organizations struggle to transform this data into actionable insights.

## Main Challenges:
  - Difficulty identifying products frequently purchased together
  - Poor customer segmentation for targeted marketing
  - Inaccurate sales forecasting
  - Overstocking and stock shortages
  - Limited personalized recommendation systems
  - Challenges handling large-scale retail datasets
## Business Impact:
  - Reduced customer satisfaction
  - Lower sales opportunities
  - Inefficient inventory management
  - Weak marketing performance
## Methodology
### 1. Data Collection & Preprocessing
    - Exploratory Data Analysis (EDA)
    - Data cleaning and handling missing values
    - Feature engineering
    - Data normalization and encoding
  <img width="672" height="540" alt="image" src="https://github.com/user-attachments/assets/b5e392eb-adff-48d2-b97e-d637ea785fb0" />
### 2. Market Basket Analysis (MBA)
The Apriori algorithm was used to identify product associations and purchasing patterns.

### Metrics Used:
  - Support
  - Confidence
  - Lift
  - Business Use Cases:
  - Product bundling
  - Cross-selling
  - Product recommendation systems
  - Store layout optimization
  - 
<img width="570" height="242" alt="image" src="https://github.com/user-attachments/assets/2a926a32-596e-488f-9913-db578762b643" />

### 3. Customer Segmentation
Random Forest was applied to classify customers based on purchasing behaviors.

### Features:
  - Customer age
  - Transaction frequency
  - Product categories
  - Average transaction value
    
<img width="603" height="534" alt="image" src="https://github.com/user-attachments/assets/793c87b1-de95-42cc-92c3-478ee1c69df3" />

### 4. Sales Forecasting
LSTM networks were used for predicting future sales trends and seasonal demand.

### Evaluation Metrics:
- RMSE
- MAE
- MAPE
<img width="282" height="78" alt="image" src="https://github.com/user-attachments/assets/35ae00d3-bcb5-4a29-9b89-81f1353ef674" />
<img width="611" height="529" alt="image" src="https://github.com/user-attachments/assets/b17b98b2-710f-4f47-82cf-7f1644682301" />
<img width="610" height="348" alt="image" src="https://github.com/user-attachments/assets/e6af4d9e-4479-48dc-8ddb-cef90eb4c0b7" />

## Skills Used
### Technical Skills
  - Python Programming
  - Data Cleaning
  - Data Visualization
  - Exploratory Data Analysis (EDA)
  - Feature Engineering
  - Predictive Analytics
  - Machine Learning
  - Deep Learning
### Machine Learning Models
  - Apriori Algorithm
  - Random Forest
  - LSTM Networks
### Libraries & Tools
  - Python
  - Pandas
  - NumPy
  - Scikit-learn
  - TensorFlow / Keras
  - Matplotlib
  - Seaborn
  - Mlxtend

## Result & Business Recommendation
### Results

The project successfully identified hidden purchasing patterns and improved retail decision-making through machine learning and analytics.

### Key Findings:
  - Frequently purchased product combinations were identified
  - Customer groups were successfully segmented
  - Sales trends and seasonal demand were accurately forecasted
  - Recommendation systems were improved using association rules

## Business Recommendations

### 1. Improve Cross-Selling
Place complementary products together and create bundle promotions.

### 2. Personalized Marketing
Use customer segmentation to deliver targeted campaigns and recommendations.

### 3. Better Inventory Management
Use forecasting insights to reduce overstocking and stock shortages.

### 4. Recommendation System Enhancement
Implement real-time recommendation engines using MBA insights.

### 5. Data-Driven Retail Strategy
Integrate machine learning into retail operations for long-term business optimization.
