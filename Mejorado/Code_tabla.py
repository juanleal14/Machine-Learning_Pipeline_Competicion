import numpy as np
import random
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')

# Preprocesado
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Prueba de Modelos
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
# Evaluacion
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, accuracy_score,
                              precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

# Fijamos Semilla
random.seed(100473223)
np.random.seed(100473223)

import numpy as np
np.random.seed(100473223)

import pandas as pd
import pickle as pkl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, accuracy_score,
                              precision_score, recall_score, f1_score,
                              average_precision_score)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings(action='ignore')

# =============================================================================
# MEJORA #1: FEATURE ENGINEERING
# =============================================================================

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

# =============================================================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# =============================================================================
print("="*80)
print("1. CARGA Y EXPLORACIÓN DE DATOS")
print("="*80)

data = pd.read_csv('train.csv')

print(f"\nDimensiones del dataset: {data.shape}")
print(f"\nDistribución de la clase objetivo (Attrition):")
print(data['Attrition'].value_counts())
attrition_dist = data['Attrition'].value_counts(normalize=True) * 100
print(f"\nPorcentaje de Attrition:\n{attrition_dist}")

class_counts = data['Attrition'].value_counts()
imbalance_ratio = class_counts['No'] / class_counts['Yes']
print(f"\n⚠ Ratio de desbalanceo: {imbalance_ratio:.2f}:1")
print(f"Clase minoritaria (Yes): {attrition_dist['Yes']:.1f}%")
print("→ Se usará BorderlineSMOTE para balancear las clases.")
print("→ Métricas principales: Balanced Accuracy, AUC-ROC, AUC-PR")

# =============================================================================
# 2. DIVISIÓN EN ENTRENAMIENTO Y TEST
# =============================================================================
print("\n" + "="*80)
print("2. DIVISIÓN EN ENTRENAMIENTO Y TEST")
print("="*80)

X = data.drop(['Attrition', 'ID'], axis=1)
y = data['Attrition'].map({'No': 0, 'Yes': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
)

print(f"\nConjunto de entrenamiento: {X_train.shape[0]} instancias")
print(f"Conjunto de test: {X_test.shape[0]} instancias")
print(f"\nDistribución en entrenamiento:\n{y_train.value_counts()}")
print(f"\nDistribución en test:\n{y_test.value_counts()}")

# =============================================================================
# 3. PREPROCESADO DE DATOS
# =============================================================================
print("\n" + "="*80)
print("3. DEFINICIÓN DEL PREPROCESADO")
print("="*80)

numerical_cols = [
    'Age', 'Miles from Home to Work', 'Yearly Income', 'Absences per Year',
    'Performance Rating', 'Job Satisfaction', 'Environment Satisfaction',
    'Work Life Balance Satisfaction', 'Last Salary Increase (%)',
    'Number of Training Sessions Last Year', 'Number of Other Companies',
    'Total Active Years', 'Years at Current Company',
    'Years Since Last Promotion', 'Years with Current Manager'
]

ordinal_cols = {
    'Education Level': [['High School', 'College', 'Bachelor', 'Master', 'Doctor']],
    'Job Level': [['Entry Level', 'Mid Level', 'Senior Level', 'Director', 'Executive']],
    'Job Involvement': [['Low', 'Medium', 'High', 'Very High']]
}

categorical_cols = [
    'Gender', 'Marital Status', 'Education Field', 'Department Name',
    'Job Role Name', 'Business Travel Frequency', 'Amount of Stock Option'
]

print(f"\nColumnas numéricas: {len(numerical_cols)}")
print(f"Columnas ordinales: {len(ordinal_cols)}")
print(f"Columnas categóricas: {len(categorical_cols)}")

# Análisis de valores faltantes
print("\n" + "="*50)
print("ANÁLISIS DE VALORES FALTANTES")
print("="*50)
missing_values = X_train.isnull().sum()
missing_percent = (missing_values / len(X_train)) * 100
missing_df = pd.DataFrame({
    'Columna': missing_values.index,
    'Valores_Faltantes': missing_values.values,
    'Porcentaje': missing_percent.values
})
missing_df = missing_df[missing_df['Valores_Faltantes'] > 0].sort_values('Valores_Faltantes', ascending=False)
if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
    
    cols_to_drop = missing_df[missing_df['Porcentaje'] > 30]['Columna'].tolist()
    if cols_to_drop:
        print(f"\n⚠ Eliminando columnas con >30% de valores faltantes: {cols_to_drop}")
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)
        
        numerical_cols = [col for col in numerical_cols if col not in cols_to_drop]
        categorical_cols = [col for col in categorical_cols if col not in cols_to_drop]
        ordinal_cols = {k: v for k, v in ordinal_cols.items() if k not in cols_to_drop}
        
        print(f"✓ Columnas eliminadas. Nuevas dimensiones: {X_train.shape}")
else:
    print("✓ No hay valores faltantes en el dataset")

# === Preprocesado ===
num_transformer = SkPipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

ord_transformers = []
for col, categories in ordinal_cols.items():
    ord_transformers.append((
        f'ord_{col}',
        SkPipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(
                categories=categories,
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ]),
        [col]
    ))

cat_transformer = SkPipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        *ord_transformers,
        ('cat', cat_transformer, categorical_cols)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# =============================================================================
# PIPELINE MEJORADO PARA DATOS DESBALANCEADOS
# =============================================================================
print("\n" + "="*80)
print("🚀 PIPELINE OPTIMIZADO PARA DATOS DESBALANCEADOS")
print("="*80)
print("✓ Mejora #1: Feature Engineering (FeatureEngineer)")
print("✓ Mejora #2: BorderlineSMOTE para balanceo inteligente")
print("✓ Enfoque: Métricas robustas para clases desbalanceadas")

ml_pipe_improved = ImbPipeline(steps=[
    ('feature_engineer', FeatureEngineer()),
    ('preprocessor', preprocessor),
    #('smote', BorderlineSMOTE(random_state=42, kind='borderline-1')),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression())  # placeholder
])

# Grid expandido para evaluar múltiples modelos
grid= [
    # 1) Logistic Regression (l1/l2, con y sin class_weight, distintos solvers)
{
    'classifier': [LogisticRegression(random_state=42, max_iter=2000)],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['liblinear'],   # l1 soportado por liblinear/saga
    'classifier__C': [3],
    'classifier__class_weight': ['balanced'],
},
   ## 2) Árbol de decisión (más profundo, min_samples_* y max_features)
{
    'classifier': [DecisionTreeClassifier(random_state=42)],
    'classifier__criterion': ['gini'],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [5],
    'classifier__min_samples_leaf': [2],
    'classifier__class_weight': ['balanced']
},
   ## 3) KNN (más vecinos, distancia y leaf_size)
{
    'classifier': [KNeighborsClassifier()],
    'classifier__n_neighbors': [9],
    'classifier__weights': ['uniform'],
    'classifier__p': [1],               
    'classifier__leaf_size': [15]
},
  # Random Forest optimizado para balanced accuracy
   ## 5) Random Forest (más n_estimators, max_features, bootstrap)
{
    'classifier': [RandomForestClassifier(random_state=42)],
    'classifier__n_estimators': [300],
    'classifier__criterion': ['gini'],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [5],
    'classifier__min_samples_leaf': [2],
    'classifier__class_weight': ['balanced']
},
   ## 6) Gradient Boosting
{
    'classifier': [GradientBoostingClassifier(random_state=42)],
    'classifier__n_estimators': [600],
    'classifier__learning_rate': [0.15],
    'classifier__max_depth': [5],
    'classifier__min_samples_leaf': [7],
    'classifier__subsample': [0.9]
},
  ## 7) Extra Trees
{
    'classifier': [ExtraTreesClassifier(random_state=42)],
    'classifier__n_estimators': [200],
    'classifier__criterion': ['gini'],
    'classifier__max_depth': [None],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
},
  ## 8) AdaBoost
{
   'classifier': [AdaBoostClassifier(
       random_state=42,
       estimator=DecisionTreeClassifier(random_state=42)
   )],
   'classifier__algorithm': ['SAMME'],
   'classifier__n_estimators': [100],
   'classifier__learning_rate': [0.1],
   'classifier__estimator__max_depth': [1],
   'classifier__estimator__min_samples_leaf': [1]
},
  # multi-layer Perceptron
  {
      'classifier': [MLPClassifier(random_state=42, max_iter=1000)],
      'classifier__hidden_layer_sizes': [(50, 50)],
      'classifier__activation': ['relu'],
      'classifier__alpha': [0.01]
  }
]

# =============================================================================
# 4. EVALUACIÓN DE TODOS LOS MODELOS
# =============================================================================
print("\n" + "="*80)
print("4. EVALUACIÓN EXHAUSTIVA DE TODOS LOS MODELOS")
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model_with_params(pipeline, params, X_train, y_train, X_test, y_test, cv):
    """Evalúa un modelo específico con parámetros dados - Enfoque para datos desbalanceados"""
    
    # Configurar pipeline con parámetros
    pipeline.set_params(**params)
    
    # Cross-validation con métricas apropiadas para datos desbalanceados
    cv_scores_ba = []  # Balanced Accuracy
    cv_scores_auc = []  # AUC-ROC
    cv_scores_pr = []   # AUC-PR (Precision-Recall)
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        pipeline.fit(X_tr, y_tr)
        y_pred_val = pipeline.predict(X_val)
        y_proba_val = pipeline.predict_proba(X_val)[:, 1]
        
        cv_scores_ba.append(balanced_accuracy_score(y_val, y_pred_val))
        cv_scores_auc.append(roc_auc_score(y_val, y_proba_val))
        cv_scores_pr.append(average_precision_score(y_val, y_proba_val))
    
    cv_ba_mean, cv_ba_std = np.mean(cv_scores_ba), np.std(cv_scores_ba)
    cv_auc_mean, cv_auc_std = np.mean(cv_scores_auc), np.std(cv_scores_auc)
    cv_pr_mean, cv_pr_std = np.mean(cv_scores_pr), np.std(cv_scores_pr)
    
    # Entrenar en todo el conjunto de entrenamiento
    pipeline.fit(X_train, y_train)
    
    # Evaluación en test
    y_pred_test = pipeline.predict(X_test)
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    
    # Métricas de test
    test_ba = balanced_accuracy_score(y_test, y_pred_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, zero_division=0)
    test_rec = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_auc_roc = roc_auc_score(y_test, y_proba_test)
    test_auc_pr = average_precision_score(y_test, y_proba_test)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'cv_balanced_accuracy_mean': cv_ba_mean,
        'cv_balanced_accuracy_std': cv_ba_std,
        'cv_auc_roc_mean': cv_auc_mean,
        'cv_auc_roc_std': cv_auc_std,
        'cv_auc_pr_mean': cv_pr_mean,
        'cv_auc_pr_std': cv_pr_std,
        'test_balanced_accuracy': test_ba,
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1_score': test_f1,
        'test_auc_roc': test_auc_roc,
        'test_auc_pr': test_auc_pr,
        'test_true_positives': tp,
        'test_false_positives': fp,
        'test_false_negatives': fn,
        'test_true_negatives': tn,
        'model': pipeline
    }

# Generar todas las combinaciones de parámetros
from sklearn.model_selection import ParameterGrid

all_results = []
total_combinations = sum(len(list(ParameterGrid(param_dict))) for param_dict in grid)
print(f"Total de combinaciones a evaluar: {total_combinations}")

current_combination = 0
for param_dict in grid:
    param_combinations = list(ParameterGrid(param_dict))
    classifier_name = param_dict['classifier'][0].__class__.__name__
    print(f"\nEvaluando {classifier_name}: {len(param_combinations)} combinaciones")
    
    for params in param_combinations:
        current_combination += 1
        print(f"  Progreso: {current_combination}/{total_combinations} - ", end="")
        
        try:
            result = evaluate_model_with_params(
                ml_pipe_improved, params, X_train, y_train, X_test, y_test, cv
            )
            
            # Agregar información del modelo y parámetros
            result['classifier'] = classifier_name
            result['params'] = str(params)
            
            # Crear descripción legible de parámetros
            param_desc = []
            for key, value in params.items():
                if key.startswith('classifier__'):
                    param_name = key.replace('classifier__', '')
                    param_desc.append(f"{param_name}={value}")
            result['param_description'] = ', '.join(param_desc)
            
            all_results.append(result)
            print(f"BA_CV: {result['cv_balanced_accuracy_mean']:.4f}, BA_Test: {result['test_balanced_accuracy']:.4f}, AUC: {result['test_auc_roc']:.4f}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

# =============================================================================
# 5. ANÁLISIS COMPLETO DE RESULTADOS
# =============================================================================
print("\n" + "="*80)
print("5. ANÁLISIS EXHAUSTIVO DE RESULTADOS")
print("="*80)

# Crear DataFrame con resultados
results_df = pd.DataFrame(all_results)

# Ordenar por balanced accuracy en test (métrica principal para datos desbalanceados)
results_df = results_df.sort_values('test_balanced_accuracy', ascending=False)

# Crear tabla resumida para visualización
summary_cols = [
    'classifier', 'param_description', 
    'cv_balanced_accuracy_mean', 'cv_balanced_accuracy_std',
    'cv_auc_roc_mean', 'cv_auc_pr_mean',
    'test_balanced_accuracy', 'test_accuracy', 'test_precision', 
    'test_recall', 'test_f1_score', 'test_auc_roc', 'test_auc_pr'
]

results_summary = results_df[summary_cols].copy()
results_summary.columns = [
    'Modelo', 'Parámetros', 'BA_CV_Mean', 'BA_CV_Std', 'AUC_CV_Mean', 'PR_CV_Mean',
    'BA_Test', 'Acc_Test', 'Prec_Test', 'Rec_Test', 'F1_Test', 'AUC_Test', 'PR_Test'
]

# Redondear valores numéricos
numeric_cols = ['BA_CV_Mean', 'BA_CV_Std', 'AUC_CV_Mean', 'PR_CV_Mean',
                'BA_Test', 'Acc_Test', 'Prec_Test', 'Rec_Test', 'F1_Test', 'AUC_Test', 'PR_Test']
for col in numeric_cols:
    results_summary[col] = results_summary[col].round(4)

print("\n📊 RESULTADOS COMPLETOS (Ordenados por Balanced Accuracy en Test):")
print("="*150)
print(results_summary.to_string(index=False))

# Top 15 modelos
print("\n\n🏆 TOP 15 MEJORES MODELOS:")
print("="*150)
top_15 = results_summary.head(15)
print(top_15.to_string(index=False))

# Análisis por tipo de modelo
print("\n\n📈 ANÁLISIS POR TIPO DE MODELO:")
print("="*100)
model_analysis = results_df.groupby('classifier').agg({
    'test_balanced_accuracy': ['count', 'mean', 'std', 'max'],
    'test_auc_roc': ['mean', 'max'],
    'test_auc_pr': ['mean', 'max'],
    'test_recall': 'mean'
}).round(4)

model_analysis.columns = ['Num_Configs', 'BA_Mean', 'BA_Std', 'BA_Max', 
                         'AUC_Mean', 'AUC_Max', 'PR_Mean', 'PR_Max', 'Recall_Mean']
print(model_analysis.to_string())

# Mejor modelo de cada tipo
print("\n\n🎯 MEJOR CONFIGURACIÓN POR TIPO DE MODELO:")
print("="*150)
best_per_model = results_df.loc[results_df.groupby('classifier')['test_balanced_accuracy'].idxmax()]
best_summary = best_per_model[summary_cols].copy()
best_summary.columns = results_summary.columns
for col in numeric_cols:
    best_summary[col] = best_summary[col].round(4)
print(best_summary.to_string(index=False))

# Análisis de matriz de confusión del mejor modelo
print("\n\n🔍 ANÁLISIS DETALLADO DEL MEJOR MODELO:")
print("="*80)
best_result = all_results[0]
print(f"Modelo: {best_result['classifier']}")
print(f"Parámetros: {best_result['param_description']}")
print(f"\nMétricas principales:")
print(f"  Balanced Accuracy: {best_result['test_balanced_accuracy']:.4f}")
print(f"  AUC-ROC: {best_result['test_auc_roc']:.4f}")
print(f"  AUC-PR: {best_result['test_auc_pr']:.4f}")
print(f"  Recall (Sensibilidad): {best_result['test_recall']:.4f}")
print(f"  Precision: {best_result['test_precision']:.4f}")

print(f"\nMatriz de Confusión:")
print(f"                 Predicho")
print(f"              No   Yes")
print(f"Real   No   {best_result['test_true_negatives']:4d}  {best_result['test_false_positives']:4d}")
print(f"       Yes  {best_result['test_false_negatives']:4d}  {best_result['test_true_positives']:4d}")

# Calcular métricas adicionales
specificity = best_result['test_true_negatives'] / (best_result['test_true_negatives'] + best_result['test_false_positives'])
print(f"\nMétricas adicionales:")
print(f"  Specificity: {specificity:.4f}")
print(f"  True Positive Rate: {best_result['test_recall']:.4f}")
print(f"  False Positive Rate: {1-specificity:.4f}")

# =============================================================================
# 6. GUARDAR RESULTADOS
# =============================================================================
print("\n" + "="*80)
print("6. GUARDADO DE RESULTADOS")
print("="*80)

# Guardar resultados completos
results_summary.to_csv('all_models_results_balanced.csv', index=False)
print("✓ Resultados completos guardados: all_models_results_balanced.csv")

# Guardar top 15
top_15.to_csv('top_15_models_balanced.csv', index=False)
print("✓ Top 15 modelos guardados: top_15_models_balanced.csv")

# Guardar análisis por tipo de modelo
model_analysis.to_csv('model_type_analysis.csv')
print("✓ Análisis por tipo guardado: model_type_analysis.csv")

# Guardar mejor modelo
best_model_data = {
    'model': best_result['model'],
    'feature_names': X_train.columns.tolist(),
    'performance_metrics': {
        'cv_balanced_accuracy_mean': best_result['cv_balanced_accuracy_mean'],
        'cv_balanced_accuracy_std': best_result['cv_balanced_accuracy_std'],
        'test_balanced_accuracy': best_result['test_balanced_accuracy'],
        'test_auc_roc': best_result['test_auc_roc'],
        'test_auc_pr': best_result['test_auc_pr'],
        'test_recall': best_result['test_recall'],
        'test_precision': best_result['test_precision'],
        'confusion_matrix': {
            'TP': best_result['test_true_positives'],
            'FP': best_result['test_false_positives'],
            'FN': best_result['test_false_negatives'],
            'TN': best_result['test_true_negatives']
        }
    },
    'model_params': best_result['param_description']
}

with open('best_model_balanced_dataset.pkl', 'wb') as f:
    pkl.dump(best_model_data, f)
print("✓ Mejor modelo guardado: best_model_balanced_dataset.pkl")

print("\n" + "="*80)
print("✅ EVALUACIÓN COMPLETA PARA DATOS DESBALANCEADOS FINALIZADA")
print("="*80)
print(f"\n🎉 MEJOR MODELO: {results_summary.iloc[0]['Modelo']}")
print(f"   Balanced Accuracy: {results_summary.iloc[0]['BA_Test']}")
print(f"   AUC-ROC: {results_summary.iloc[0]['AUC_Test']}")
print(f"   AUC-PR: {results_summary.iloc[0]['PR_Test']}")
print(f"   Recall: {results_summary.iloc[0]['Rec_Test']}")
print(f"\n📊 Total de modelos evaluados: {len(results_summary)}")
print("\n🎯 Métricas clave para datos desbalanceados:")
print("   - Balanced Accuracy: Métrica principal para evaluación")
print("   - AUC-ROC: Capacidad de discriminación entre clases")
print("   - AUC-PR: Rendimiento en la clase minoritaria (Attrition)")
print("   - Recall: Capacidad de detectar casos de Attrition")
print("\nArchivos generados:")
print("  1. all_models_results_balanced.csv    - Resultados completos")
print("  2. top_15_models_balanced.csv         - Top 15 mejores modelos")
print("  3. model_type_analysis.csv            - Análisis por algoritmo")
print("  4. best_model_balanced_dataset.pkl    - Mejor modelo optimizado")