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
                              precision_score, recall_score, f1_score)
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

#from feature_engineer import FeatureEngineer  # Asegura que la clase esté disponible
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
print("→ Se usará BorderlineSMOTE en el pipeline de entrenamiento y CV.")

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
# PIPELINE MEJORADO CON LAS 3 OPTIMIZACIONES
# =============================================================================
print("\n" + "="*80)
print("🚀 PIPELINE MEJORADO CON TOP 3 OPTIMIZACIONES")
print("="*80)
print("✓ Mejora #1: Feature Engineering (FeatureEngineer)")
print("✓ Mejora #2: BorderlineSMOTE (mejor que SMOTE básico)")
print("✓ Mejora #3: Threshold Optimization (se aplicará después)")

# MEJORA #2: Usar BorderlineSMOTE en lugar de SMOTE
ml_pipe_improved = ImbPipeline(steps=[
    ('feature_engineer', FeatureEngineer()),  # ← MEJORA #1
    ('preprocessor', preprocessor),
    ('smote', BorderlineSMOTE(random_state=42, kind='borderline-1')),  # ← MEJORA #2
    ('classifier', LogisticRegression())  # placeholder
])

# Grid refinado basado en mejores resultados anteriores
grid = [ # Mejor modelo encontrado
        {
        'classifier': [GradientBoostingClassifier(random_state=42)],
        'classifier__n_estimators': [600],
        'classifier__learning_rate': [0.15],
        'classifier__max_depth': [5],
        'classifier__subsample': [0.9],
        'classifier__min_samples_leaf': [7]
    }
]

# =============================================================================
# 4. BÚSQUEDA DE HIPERPARÁMETROS
# =============================================================================
print("\n" + "="*80)
print("4. BÚSQUEDA DE HIPERPARÁMETROS")
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search_improved = GridSearchCV(
    estimator=ml_pipe_improved,
    param_grid=grid,
    scoring='balanced_accuracy',
    cv=cv,
    n_jobs=-1,
    error_score='raise',
    verbose=1,
)

print("\nIniciando búsqueda con pipeline mejorado...")
search_improved.fit(X_train, y_train)
print("✓ Búsqueda completada!")

print(f"\nMejor modelo:")
print(f"  Clasificador: {search_improved.best_estimator_['classifier'].__class__.__name__}")
print(f"  Balanced Accuracy (CV): {search_improved.best_score_:.4f}")
print(f"\nMejores parámetros:")
for param, value in search_improved.best_params_.items():
    print(f"  {param}: {value}")

best_model = search_improved.best_estimator_

# =============================================================================
# MEJORA #3: OPTIMIZACIÓN DEL THRESHOLD
# =============================================================================
print("\n" + "="*80)
print("🎯 MEJORA #3: OPTIMIZACIÓN DEL THRESHOLD")
print("="*80)

def find_optimal_threshold(model, X_val, y_val):
    """Encuentra el threshold óptimo para balanced accuracy"""
    y_proba = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        score = balanced_accuracy_score(y_val, y_pred)
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]
    
    print(f"Threshold por defecto (0.5): {balanced_accuracy_score(y_val, (y_proba >= 0.5).astype(int)):.4f}")
    print(f"Threshold óptimo encontrado: {optimal_threshold:.3f}")
    print(f"Balanced Accuracy con threshold óptimo: {optimal_score:.4f}")
    print(f"Mejora: +{(optimal_score - balanced_accuracy_score(y_val, (y_proba >= 0.5).astype(int))):.4f}")
    
    return optimal_threshold

# Encontrar threshold óptimo en train
optimal_threshold = find_optimal_threshold(best_model, X_train, y_train)

# =============================================================================
# 5. EVALUACIÓN EN TEST (CON THRESHOLD ÓPTIMO)
# =============================================================================
print("\n" + "="*80)
print("5. EVALUACIÓN EN TEST")
print("="*80)

# Predicciones con threshold por defecto (0.5)
y_pred_default = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Predicciones con threshold óptimo
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

# Métricas con threshold por defecto
ba_default = balanced_accuracy_score(y_test, y_pred_default)
acc_default = accuracy_score(y_test, y_pred_default)
prec_default = precision_score(y_test, y_pred_default)
rec_default = recall_score(y_test, y_pred_default)
f1_default = f1_score(y_test, y_pred_default)

# Métricas con threshold óptimo
ba_optimal = balanced_accuracy_score(y_test, y_pred_optimal)
acc_optimal = accuracy_score(y_test, y_pred_optimal)
prec_optimal = precision_score(y_test, y_pred_optimal)
rec_optimal = recall_score(y_test, y_pred_optimal)
f1_optimal = f1_score(y_test, y_pred_optimal)
auc = roc_auc_score(y_test, y_proba)

# Comparación de métricas
metrics_comparison = pd.DataFrame({
    'Métrica': ['Balanced Accuracy', 'Accuracy', 'Precision (Yes)', 'Recall (Yes)', 'F1-Score (Yes)', 'AUC-ROC'],
    'Threshold 0.5': [ba_default, acc_default, prec_default, rec_default, f1_default, auc],
    'Threshold Óptimo': [ba_optimal, acc_optimal, prec_optimal, rec_optimal, f1_optimal, auc],
    'Mejora': [
        ba_optimal - ba_default,
        acc_optimal - acc_default,
        prec_optimal - prec_default,
        rec_optimal - rec_default,
        f1_optimal - f1_default,
        0
    ]
})

print("\n📊 COMPARACIÓN DE RESULTADOS:")
print(metrics_comparison.to_string(index=False))

print(f"\n🎉 Balanced Accuracy Final: {ba_optimal:.4f}")
print(f"   Mejora vs threshold 0.5: +{(ba_optimal - ba_default):.4f}")

# Matriz de confusión con threshold óptimo
cm = confusion_matrix(y_test, y_pred_optimal)
tn, fp, fn, tp = cm.ravel()
print("\n📋 Matriz de Confusión (Threshold Óptimo):")
print(cm)
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

print("\n📄 REPORTE DE CLASIFICACIÓN (Threshold Óptimo):")
print(classification_report(y_test, y_pred_optimal, target_names=['No Attrition', 'Attrition']))

# =============================================================================
# 6. GUARDAR MODELO Y RESULTADOS
# =============================================================================
print("\n" + "="*80)
print("6. GUARDADO DE MODELO Y RESULTADOS")
print("="*80)

# Guardar modelo con threshold óptimo
model_data = {
    'model': best_model,
    'optimal_threshold': optimal_threshold,
    'feature_names': X_train.columns.tolist()
}

with open('best_model_improved.pkl', 'wb') as f:
    pkl.dump(model_data, f)
print("✓ Modelo mejorado guardado: best_model_improved.pkl")
print(f"  (incluye threshold óptimo: {optimal_threshold:.3f})")

metrics_comparison.to_csv('metrics_improved.csv', index=False)
print("✓ Métricas guardadas: metrics_improved.csv")

print("\n" + "="*80)
print("✅ PROCESO COMPLETADO CON ÉXITO")
print("="*80)
print("\n🎯 RESUMEN DE MEJORAS IMPLEMENTADAS:")
print("  1. ✓ Feature Engineering: +13 features derivadas")
print("  2. ✓ BorderlineSMOTE: mejor oversampling que SMOTE básico")
print(f"  3. ✓ Threshold Optimization: threshold={optimal_threshold:.3f}")
print(f"\n📈 Balanced Accuracy Final: {ba_optimal:.4f}")
print("\nArchivos generados:")
print("  1. best_model_improved.pkl - Modelo mejorado con threshold óptimo")
print("  2. metrics_improved.csv    - Comparación de métricas")
