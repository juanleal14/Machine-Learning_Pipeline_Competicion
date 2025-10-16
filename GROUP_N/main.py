# main.py
import argparse
import os
import sys
import pickle as pkl
import numpy as np
import pandas as pd

#from Mejoras import FeatureEngineer  # Asegura que la clase esté disponible
LABELS = {0: "No", 1: "Yes"}

def load_model():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = ["best_model.pkl", "best_model_final.pkl"]
    #candidates = ["best_model_improved.pkl"]
    for name in candidates:
        path = os.path.join(here, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return pkl.load(f)
    # Si no se encuentra, error por stderr (stdout debe quedar limpio)
    print("Model file not found (best_model.pkl / best_model_final.pkl).", file=sys.stderr)
    sys.exit(1)

def to_labels(y_pred, y_proba=None):
    """
    Convierte a 'Yes'/'No' con umbral 0.5 si hay proba; si no, usa predict ya binario/strings.
    """
    if y_proba is not None:
        # y_proba shape: (n_samples, 2) o (n_samples,)
        if y_proba.ndim == 2:
            pos = y_proba[:, 1]
        else:
            pos = y_proba
        y_bin = (pos >= 0.5).astype(int)
        return np.array([LABELS[int(v)] for v in y_bin])
    # Sin proba: puede venir como 0/1 o 'Yes'/'No'
    if np.issubdtype(np.array(y_pred).dtype, np.number):
        return np.array([LABELS[int(v)] for v in y_pred])
    # ya son strings
    return np.array(y_pred, dtype=str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the input CSV (in.csv)")
    args = parser.parse_args()

    # Carga de datos
    try:
        df = pd.read_csv(args.file)
    except Exception:
        # No imprimir nada en stdout
        print("Error: the input file does not have a valid format.", file=sys.stderr)
        sys.exit(1)

    # Validaciones mínimas
    if "ID" not in df.columns:
        print("Error: missing 'ID' column in input.", file=sys.stderr)
        sys.exit(1)
    # Asegura que no venga la columna objetivo (en competición no debe venir)
    if "Attrition" in df.columns:
        df = df.drop(columns=["Attrition"])

    # Carga del modelo (Pipeline completo con preprocesado+SMOTE+clasificador)
    model = load_model()

    # Prepara X tal y como el modelo fue entrenado (ID no es feature)
    X = df.drop(columns=["ID"], errors="ignore")

    # Predicción
    # Preferimos predict_proba si existe; si no, predict
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
        except Exception:
            y_proba = None
    y_pred = model.predict(X)

    # A etiquetas 'Yes'/'No'
    y_labels = to_labels(y_pred, y_proba=y_proba)

    # Salida EXACTA por stdout
    # (No imprimir nada más; usar print simple línea a línea)
    print("ID,Attrition")
    # Mantener el orden del input
    for _id, lab in zip(df["ID"].values, y_labels):
        print(f"{_id},{lab}")

if __name__ == "__main__":
    main()
