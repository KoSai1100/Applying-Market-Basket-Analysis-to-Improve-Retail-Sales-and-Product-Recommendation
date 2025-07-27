from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

def prepare_basket_data(cleaned_df):
    basket = cleaned_df.groupby(['Customer_ID', 'Product_Category'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket

def run_apriori(basket, min_support=0.07):
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    return frequent_itemsets

def generate_association_rules(frequent_itemsets, metric="lift", min_threshold=1.0):
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    return rules.sort_values(by='lift', ascending=False)

class AprioriModel:
    def __init__(self):
        pass

    def run(self, input_data):
        # Example: expects input_data to have 'Customer_ID', 'Product_Category', 'Quantity'
        try:
            basket = prepare_basket_data(input_data)
            frequent_itemsets = run_apriori(basket)
            rules = generate_association_rules(frequent_itemsets)
            return {
                "num_rules": len(rules),
                "example_rules": rules.head(5).to_dict(orient="records"),
                "rules_df": rules
                }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_metrics():
        # Example static metrics
        return {"support": 0.8, "confidence": 0.7}