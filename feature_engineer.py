import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, accuracy_score,
                              precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings(action='ignore')
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Crea features derivadas para capturar patrones no lineales"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Evitar división por cero
        def safe_divide(a, b):
            return np.where(b != 0, a / b, 0)
        
        # === RATIOS Y PROPORCIONES ===
        # Estabilidad laboral
        X['Years_per_Company_Ratio'] = safe_divide(
            X['Total Active Years'], 
            X['Number of Other Companies'] + 1
        )
        
        # Retraso en promoción
        X['Promotion_Lag'] = X['Years at Current Company'] - X['Years Since Last Promotion']
        
        # Ingreso por año de experiencia
        X['Income_per_Year'] = safe_divide(
            X['Yearly Income'], 
            X['Total Active Years'] + 1
        )
        
        # Estabilidad con manager actual
        X['Manager_Stability'] = safe_divide(
            X['Years with Current Manager'], 
            X['Years at Current Company'] + 1
        )
        
        # === SCORES AGREGADOS ===
        # Satisfacción general (promedio de satisfacciones)
        satisfaction_cols = ['Job Satisfaction', 'Environment Satisfaction', 
                            'Work Life Balance Satisfaction']
        
        # Manejar valores faltantes temporalmente para el cálculo
        X['Overall_Satisfaction'] = X[satisfaction_cols].mean(axis=1, skipna=True)
        
        # === INDICADORES DE RIESGO (variables binarias) ===
        # Baja satisfacción general
        X['Low_Satisfaction'] = (X['Overall_Satisfaction'] < 3).astype(int)
        
        # Empleado recién contratado
        X['Recent_Hire'] = (X['Years at Current Company'] < 2).astype(int)
        
        # Promoción atrasada
        X['Overdue_Promotion'] = (X['Years Since Last Promotion'] > 3).astype(int)
        
        # Commute largo (si la columna existe)
        if 'Miles from Home to Work' in X.columns:
            X['Long_Commute'] = (X['Miles from Home to Work'] > 20).astype(int)
        
        # === INTERACCIONES IMPORTANTES ===
        # Edad vs. Nivel de trabajo (senior joven o junior viejo = señal)
        X['Age_x_JobLevel'] = X['Age'] * pd.factorize(X.get('Job Level', pd.Series([0]*len(X))))[0]
        
        # Satisfacción baja + salario bajo = alto riesgo
        X['LowSat_LowIncome'] = (
            (X['Overall_Satisfaction'] < 3) & 
            (X['Yearly Income'] < X['Yearly Income'].median())
        ).astype(int)
        
        return X
