import numpy as np
import random
import pandas as pd
import pickle as pkl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, accuracy_score,
                              precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
warnings.filterwarnings(action='ignore')

random.seed(100473223)
np.random.seed(100473223)
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

# Ratio de desbalanceo (solo informativo)
class_counts = data['Attrition'].value_counts()
imbalance_ratio = class_counts['No'] / class_counts['Yes']
print(f"\n⚠ Ratio de desbalanceo: {imbalance_ratio:.2f}:1")
print(f"Clase minoritaria (Yes): {attrition_dist['Yes']:.1f}%")
print("→ Se usará SMOTE en el pipeline de entrenamiento y CV.")

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
else:
    print("✓ No hay valores faltantes en el dataset")
print(f"\nTotal de columnas con valores faltantes: {len(missing_df)}")
    
# Eliminar columnas con >30% de valores faltantes
cols_to_drop = missing_df[missing_df['Porcentaje'] > 30]['Columna'].tolist()
if cols_to_drop:
    print(f"\n⚠ Eliminando columnas con >30% de valores faltantes: {cols_to_drop}")
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)
    
    # Actualizar listas de columnas
    numerical_cols = [col for col in numerical_cols if col not in cols_to_drop]
    categorical_cols = [col for col in categorical_cols if col not in cols_to_drop]
    ordinal_cols = {k: v for k, v in ordinal_cols.items() if k not in cols_to_drop}
    
    print(f"✓ Columnas eliminadas. Nuevas dimensiones: {X_train.shape}")
else:
    print("✓ No hay valores faltantes en el dataset")
# === Sustituye todo tu bloque de preprocesado y pipeline por este ===
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1) Transformadores por tipo de columna (pipelines internas OK aquí)
num_transformer = SkPipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Un pipeline ordinal por cada columna ordinal (imputar + codificar)
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
            # Si quisieras escalar también los ordinales, añade ('scaler', StandardScaler())
        ]),
        [col]
    ))

cat_transformer = SkPipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# 2) Un único ColumnTransformer como preprocesador (no es Pipeline → OK para ImbPipeline)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        *ord_transformers,
        ('cat', cat_transformer, categorical_cols)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)
# === Contador de fits para GridSearchCV ===
from sklearn.base import BaseEstimator, TransformerMixin

_FIT_COUNT = 0  # contador global

class FitCounter(BaseEstimator, TransformerMixin):
    def __init__(self, print_every=500):
        self.print_every = print_every

    def fit(self, X, y=None):
        # amazonq-ignore-next-line
        global _FIT_COUNT
        _FIT_COUNT += 1
        if _FIT_COUNT % self.print_every == 0:
            print(f"Completados {_FIT_COUNT} fits...")
        return self

    def transform(self, X):
        return X  # passthrough
# 3) Pipeline FINAL con SMOTE (Sin nested Pipeline en pasos intermedios)
ml_pipe_smote = ImbPipeline(steps=[
    #('fit_counter', FitCounter(print_every=500)),
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression())  # placeholder; lo sobreescribe el Grid
])


grid= [
    # amazonq-ignore-next-line
    # 1) Logistic Regression (l1/l2, con y sin class_weight, distintos solvers)
 #{
 #    'classifier': [LogisticRegression(random_state=42, max_iter=2000)],
 #    'classifier__penalty': ['l2', 'l1'],
 #    'classifier__solver': ['liblinear', 'saga'],   # l1 soportado por liblinear/saga
 #    'classifier__C': [0.01, 0.1, 1, 3, 10],
 #    'classifier__class_weight': [None, 'balanced']
 #},
 #   ## 2) Árbol de decisión (más profundo, min_samples_* y max_features)
 #{
 #    'classifier': [DecisionTreeClassifier(random_state=42)],
 #    'classifier__criterion': ['gini', 'entropy', 'log_loss'],
 #    'classifier__max_depth': [None, 5, 10, 20, 30],
 #    'classifier__min_samples_split': [2, 5, 10, 20],
 #    'classifier__min_samples_leaf': [1, 2, 5, 10],
 #    'classifier__max_features': [None, 'sqrt', 'log2'],
 #    'classifier__class_weight': [None, 'balanced']
 #},
 #   ## 3) KNN (más vecinos, distancia y leaf_size)
 #{
 #    'classifier': [KNeighborsClassifier()],
 #    'classifier__n_neighbors': [3, 5, 7, 9, 11, 15],
 #    'classifier__weights': ['uniform', 'distance'],
 #    'classifier__p': [1, 2],                # manhattan / euclid
 #    'classifier__leaf_size': [15, 30, 45]
 #},
 #   ## 4) MLP (más capas, early stopping, tasa aprendizaje y batch)
    # Random Forest optimizado para balanced accuracy
    #{
    #    'classifier': [RandomForestClassifier(random_state=42)],
    #    'classifier__n_estimators': [300, 500, 700],
    #    'classifier__max_depth': [15, 20, 25],
    #    'classifier__min_samples_split': [5, 8],
    #    'classifier__min_samples_leaf': [2, 3],
    #    'classifier__max_features': ['sqrt', 'log2'],
    #    'classifier__class_weight': ['balanced']
    #},
 #   ## 5) Random Forest (más n_estimators, max_features, bootstrap)
 #{
 #    'classifier': [RandomForestClassifier(random_state=42)],
 #    'classifier__n_estimators': [100, 300, 600],
 #    'classifier__criterion': ['gini', 'entropy'],
 #    'classifier__max_depth': [None, 10, 20, 40],
 #    'classifier__min_samples_split': [2, 5, 10],
 #    'classifier__min_samples_leaf': [1, 2, 4],
 #    'classifier__max_features': ['sqrt', 'log2', None],
 #    'classifier__bootstrap': [True, False],
 #    'classifier__class_weight': [None, 'balanced']
 #},
 #   ## 6) Gradient Boosting (añado subsample, min_samples_leaf)
 #{
 #    'classifier': [GradientBoostingClassifier(random_state=42)],
 #    'classifier__n_estimators': [100, 200, 400],
 #    'classifier__learning_rate': [0.02, 0.05, 0.1, 0.2],
 #    'classifier__max_depth': [2, 3, 5],
 #    'classifier__min_samples_leaf': [1, 2, 5],
 #    'classifier__subsample': [0.6, 0.8, 1.0]
 #},
    ## 7) Extra Trees (robusto y rápido)
 #{
 #    'classifier': [ExtraTreesClassifier(random_state=42)],
 #    'classifier__n_estimators': [200, 500, 800],
 #    'classifier__criterion': ['gini', 'entropy', 'log_loss'],
 #    'classifier__max_depth': [None, 10, 20, 40],
 #    'classifier__min_samples_split': [2, 5, 10],
 #    'classifier__min_samples_leaf': [1, 2, 4],
 #    'classifier__max_features': ['sqrt', 'log2', None],
 #    'classifier__bootstrap': [False],           # ExtraTrees suele ir sin bootstrap
 #    'classifier__class_weight': [None, 'balanced']
 #},
    ## 8) AdaBoost (corrigido para scikit-learn >=1.2)
#{
#    'classifier': [AdaBoostClassifier(
#        random_state=42,
#        estimator=DecisionTreeClassifier(random_state=42)
#    )],
#    'classifier__algorithm': ['SAMME'],  # ← evita el warning (no usa SAMME.R)
#    'classifier__n_estimators': [100, 300, 600],
#    'classifier__learning_rate': [0.02, 0.05, 0.1, 0.2, 0.5],
#    'classifier__estimator__max_depth': [1, 2, 3],
#    'classifier__estimator__min_samples_leaf': [1, 2, 5]
#},
    # Gradient Boosting - Excelente para balanced accuracy
    {
        'classifier': [GradientBoostingClassifier(random_state=42)],
        'classifier__n_estimators': [200, 400, 600],
        'classifier__learning_rate': [0.05, 0.1, 0.15],
        'classifier__max_depth': [3, 4, 5],
        'classifier__subsample': [0.8, 0.9],
        'classifier__min_samples_leaf': [3, 5, 7]
    },
    # Extra Trees - Muy bueno para balanced accuracy
    #{
    #    'classifier': [ExtraTreesClassifier(random_state=42)],
    #    'classifier__n_estimators': [400, 600],
    #    'classifier__max_depth': [20, 30],
    #    'classifier__min_samples_split': [5, 8],
    #    'classifier__min_samples_leaf': [2, 4],
    #    'classifier__max_features': ['sqrt'],
    #    'classifier__class_weight': ['balanced']
    #}
    #{
    #    'classifier': [RandomForestClassifier(random_state=42)],
    #    'classifier__n_estimators': [300, 500],
    #    'classifier__class_weight': ['balanced']
    #}
]

from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search_smote = GridSearchCV(
    estimator=ml_pipe_smote,   # tu ImbPipeline(preprocessor → SMOTE → classifier)
    param_grid=grid,
    scoring='balanced_accuracy',
    cv=cv,
    n_jobs=-1,
    error_score='raise',
    verbose = 1,
)


print("\nIniciando búsqueda CON SMOTE integrado (CV estratificada)...")
_FIT_COUNT = 0
search_smote.fit(X_train, y_train)
print("✓ Búsqueda CON SMOTE completada!")

print(f"\nMejor modelo CON SMOTE:")
print(f"  Clasificador: {search_smote.best_estimator_['classifier'].__class__.__name__}")
print(f"  Balanced Accuracy (CV): {search_smote.best_score_:.4f}")

best_model = search_smote.best_estimator_

# =============================================================================
# 5. EVALUACIÓN EN TEST (SOLO MODELO CON SMOTE)
# =============================================================================
print("\n" + "="*80)
print("5. EVALUACIÓN EN TEST (SOLO CON SMOTE)")
print("="*80)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

ba = balanced_accuracy_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

metrics_df = pd.DataFrame({
    'Métrica': ['Balanced Accuracy', 'Accuracy', 'Precision (Yes)', 'Recall (Yes)', 'F1-Score (Yes)', 'AUC-ROC'],
    'Con SMOTE': [ba, acc, prec, rec, f1, auc]
})
print("\nMétricas en test (CON SMOTE):")
print(metrics_df.to_string(index=False))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\nMatriz de Confusión (CON SMOTE):")
print(cm)
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

print("\nREPORTE DE CLASIFICACIÓN (CON SMOTE):")
print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))


# =============================================================================
# 6. GUARDAR MODELO Y RESULTADOS
# =============================================================================
print("\n" + "="*80)
print("7. GUARDADO DE MODELO Y RESULTADOS")
print("="*80)

with open('best_model_final.pkl', 'wb') as f:
    pkl.dump(best_model, f)
print("✓ Modelo guardado: best_model_final.pkl (CON SMOTE)")

metrics_df.to_csv('metrics_smote.csv', index=False)
print("✓ Métricas guardadas: metrics_smote.csv")

print("\n" + "="*80)
print("PROCESO COMPLETADO (SOLO SMOTE)")
print("="*80)
print("\nArchivos generados:")
print("  1. best_model_final.pkl - Modelo entrenado (CON SMOTE)")
print("  2. metrics_smote.csv    - Métricas de test")
print(f"\nBalanced Accuracy en test (SMOTE): {ba:.4f}")
print("Model Parameters: ")
for param, value in best_model.get_params().items():
    print(f"  {param}: {value}")