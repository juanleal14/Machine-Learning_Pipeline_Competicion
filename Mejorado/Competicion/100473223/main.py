import argparse
import numpy as np
import pandas as pd
import sys
import pickle as pkl
import os

# CRÍTICO: Importar FeatureEngineer para que pickle pueda deserializar el modelo

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering transformer"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        def safe_divide(a, b):
            return np.where(b != 0, a / b, 0)
        
        X['Years_per_Company_Ratio'] = safe_divide(
            X['Total Active Years'], 
            X['Number of Other Companies'] + 1
        )
        X['Promotion_Lag'] = X['Years at Current Company'] - X['Years Since Last Promotion']
        X['Income_per_Year'] = safe_divide(
            X['Yearly Income'], 
            X['Total Active Years'] + 1
        )
        X['Manager_Stability'] = safe_divide(
            X['Years with Current Manager'], 
            X['Years at Current Company'] + 1
        )
        
        satisfaction_cols = ['Job Satisfaction', 'Environment Satisfaction', 
                            'Work Life Balance Satisfaction']
        X['Overall_Satisfaction'] = X[satisfaction_cols].mean(axis=1, skipna=True)
        
        X['Low_Satisfaction'] = (X['Overall_Satisfaction'] < 3).astype(int)
        X['Recent_Hire'] = (X['Years at Current Company'] < 2).astype(int)
        X['Overdue_Promotion'] = (X['Years Since Last Promotion'] > 3).astype(int)
        
        if 'Miles from Home to Work' in X.columns:
            X['Long_Commute'] = (X['Miles from Home to Work'] > 20).astype(int)
        
        if 'Job Level' in X.columns:
            job_level_numeric = pd.factorize(X['Job Level'])[0]
            X['Age_x_JobLevel'] = X['Age'] * job_level_numeric
        else:
            X['Age_x_JobLevel'] = 0
        
        median_income = X['Yearly Income'].median()
        X['LowSat_LowIncome'] = (
            (X['Overall_Satisfaction'] < 3) & 
            (X['Yearly Income'] < median_income)
        ).astype(int)
        
        return X

def predict(input):
    """
    Generates the prediction using the trained model.
    
    This function:
    1. Loads the trained model (with optimal threshold)
    2. Preprocesses the input data
    3. Generates predictions using the optimal threshold
    4. Returns a dataframe with ID and Attrition columns
    
    @param input: the input dataframe with features
    @return: a dataframe with two columns: ID, Attrition
    """
    
    # Load the model and optimal threshold
    here = os.path.dirname(os.path.abspath(__file__))
    model_candidates = ["best_model_improved.pkl"]
    
    model = None
    threshold = 0.5  # default
    
    for model_name in model_candidates:
        model_path = os.path.join(here, model_name)
        if os.path.isfile(model_path):
            with open(model_path, "rb") as f:
                loaded = pkl.load(f)
            
            # Check if it's the improved model (dictionary format)
            if isinstance(loaded, dict) and 'model' in loaded:
                model = loaded['model']
                threshold = loaded['optimal_threshold']

    
    # Prepare output dataframe
    output = pd.DataFrame()
    output['ID'] = input['ID']
    
    # Remove ID and Attrition (if present) from features
    X = input.drop(columns=['ID'], errors='ignore')
    if 'Attrition' in X.columns:
        X = X.drop(columns=['Attrition'])
    
    # Generate predictions
    try:
        # Get probabilities for optimal threshold
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
            # Extract probability of positive class (Attrition = Yes)
            if y_proba.ndim == 2:
                pos_proba = y_proba[:, 1]
            else:
                pos_proba = y_proba
            
            # Apply optimal threshold
            y_pred = (pos_proba >= threshold).astype(int)
        else:
            # Fallback to direct prediction if no predict_proba
            y_pred = model.predict(X)
        
        # Convert to 'Yes'/'No' labels
        attrition = np.where(y_pred == 1, 'Yes', 'No')
        output['Attrition'] = attrition
        
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    
    return output


if __name__ == '__main__':
    
    # Creates a parser to receive the input argument.
    parser = argparse.ArgumentParser(
        description='Predict employee attrition using trained ML model.'
    )
    parser.add_argument('file', help='Path to the data file (CSV format).')
    args = parser.parse_args()
    
    # Read the argument and load the data.
    try:
        data = pd.read_csv(args.file)
    except:
        print("Error: the input file does not have a valid format.", file=sys.stderr)
        exit(1)
    
    # Validate that ID column exists
    if 'ID' not in data.columns:
        print("Error: missing 'ID' column in input.", file=sys.stderr)
        exit(1)
    
    # Computes the predictions using the trained model
    output = predict(data)
    
    # Writes the output to stdout
    print('ID,Attrition')
    for r in output.itertuples():
        print(f'{r.ID},{r.Attrition}')