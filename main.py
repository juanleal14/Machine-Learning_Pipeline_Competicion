import numpy as np
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

warnings.filterwarnings(action='ignore')

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
    # Usa 'sparse=False' para compatibilidad con scikit-learn <1.2
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

# 3) Pipeline FINAL con SMOTE (ahora sí, sin nested Pipeline en pasos intermedios)
ml_pipe_smote = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression())  # placeholder; lo sobreescribe el Grid
])


grid = [
    {
        'classifier': [LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)],
        'classifier__solver': ['lbfgs', 'liblinear'],
        'classifier__C': [0.01, 0.1, 1, 10],
    },
    {
        'classifier': [DecisionTreeClassifier(random_state=42, class_weight='balanced')],
        'classifier__criterion': ['entropy', 'gini'],
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
    },
    {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [3, 5, 7, 11],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__p': [1, 2],
    },
    {
        'classifier': [MLPClassifier(random_state=42, max_iter=500)],
        'classifier__hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__alpha': [0.0001, 0.001, 0.01],
    },
    {
        'classifier': [RandomForestClassifier(random_state=42, class_weight='balanced')],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__criterion': ['entropy', 'gini'],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5],
    },
    {
        'classifier': [GradientBoostingClassifier(random_state=42)],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
    }
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search_smote = GridSearchCV(
    ml_pipe_smote,
    grid,
    scoring='balanced_accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("\nIniciando búsqueda CON SMOTE integrado (CV estratificada)...")
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
# 6. VISUALIZACIONES (SOLO CON SMOTE)
# =============================================================================
print("\n" + "="*80)
print("6. GENERACIÓN DE VISUALIZACIONES (SOLO CON SMOTE)")
print("="*80)

# Matriz de confusión
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.ylabel('Valor Real')
plt.xlabel('Predicción')
plt.title(f'Confusion Matrix (SMOTE)\nBalanced Acc: {ba:.3f}')
plt.tight_layout()
plt.savefig('confusion_smote.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: confusion_smote.png")

# Barras de métricas
plt.figure(figsize=(8,5))
plt.bar(metrics_df['Métrica'], metrics_df['Con SMOTE'], alpha=0.85)
plt.ylabel('Score')
plt.title('Métricas en Test (CON SMOTE)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('metrics_smote.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: metrics_smote.png")

# =============================================================================
# 7. GUARDAR MODELO Y RESULTADOS
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
print("  3. confusion_smote.png  - Matriz de confusión")
print("  4. metrics_smote.png    - Gráfico de métricas")
print(f"\nBalanced Accuracy en test (SMOTE): {ba:.4f}")
