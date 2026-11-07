"""Experimentos con Titanic. 

Ejecutar desde el directorio discoverR: python validar_modelos.py
Los archivos de salida se generan en resultados_lab02/.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay

from supervisados import Supervisados

RAIZ = Path(__file__).resolve().parent
ARCHIVO = RAIZ / "titanic4_2026.csv"
SALIDA = RAIZ / "resultados_lab02"

# Variantes compactas para que el experimento sea manejable en una laptop.
# Incluye configuraciones basicas y alternativas para cada algoritmo.
GRILLAS = {
    "KNN": {"clasificador__n_neighbors": [3, 5, 7, 9, 11],
            "clasificador__weights": ["uniform", "distance"]},
    "Arbol de Decision": {"clasificador__max_depth": [3, 5, 10, None],
                          "clasificador__min_samples_leaf": [1, 5]},
    "Random Forest": {"clasificador__n_estimators": [100, 200],
                      "clasificador__max_depth": [5, None],
                      "clasificador__min_samples_leaf": [1, 3]},
    "XGBoost": {"clasificador__n_estimators": [100, 200],
                "clasificador__max_depth": [3, 5],
                "clasificador__learning_rate": [0.05, 0.1]},
    "AdaBoost": {"clasificador__n_estimators": [50, 100],
                 "clasificador__learning_rate": [0.5, 1.0]},
}


def main():
    if not ARCHIVO.exists():
        raise FileNotFoundError(f"No se encontro el CSV: {ARCHIVO}")
    SALIDA.mkdir(exist_ok=True)
    df = pd.read_csv(ARCHIVO)
    laboratorio = Supervisados(df, objetivo="survived", semilla=42)
    X_train, X_test, y_train, y_test = laboratorio.dividir_datos()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resumen = []
    todas_las_variantes = []

    print(f"Datos: {len(df)} filas, {len(df.columns)} columnas")
    print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

    for nombre, grilla in GRILLAS.items():
        print(f"\n===== {nombre} =====", flush=True)
        modelo = laboratorio.crear_modelo(nombre, X_train)
        busqueda = GridSearchCV(
            estimator=modelo, param_grid=grilla, scoring="f1_macro",
            cv=cv, n_jobs=-1 if nombre != "XGBoost" else 1,
            refit=True, return_train_score=False, error_score="raise"
        )
        busqueda.fit(X_train, y_train)
        pred = busqueda.predict(X_test)
        metricas = laboratorio.evaluar(y_test, pred)
        print(f"Mejores parametros: {busqueda.best_params_}")
        print(f"F1 macro CV: {busqueda.best_score_:.4f}")
        print(f"Accuracy prueba: {metricas['accuracy']:.4f}")
        print(f"F1 macro prueba: {metricas['f1_macro']:.4f}")

        detalle = pd.DataFrame(busqueda.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].sort_values("rank_test_score")
        detalle.insert(0, "modelo", nombre)
        detalle["params"] = detalle["params"].apply(lambda v: json.dumps(v, ensure_ascii=False))
        todas_las_variantes.append(detalle)

        fila = {"modelo": nombre, "f1_macro_cv": busqueda.best_score_,
                "accuracy_prueba": metricas["accuracy"],
                "precision_macro_prueba": metricas["precision_macro"],
                "recall_macro_prueba": metricas["recall_macro"],
                "f1_macro_prueba": metricas["f1_macro"],
                "mejores_parametros": json.dumps(busqueda.best_params_, ensure_ascii=False)}
        resumen.append(fila)

        ConfusionMatrixDisplay.from_predictions(y_test, pred, values_format="d")
        plt.title(f"Matriz de confusion - {nombre}")
        plt.tight_layout()
        archivo_grafico = nombre.lower().replace(" ", "_")
        plt.savefig(SALIDA / f"matriz_{archivo_grafico}.png", dpi=160)
        plt.close()

    tabla = pd.DataFrame(resumen).sort_values("f1_macro_cv", ascending=False)
    tabla.to_csv(SALIDA / "comparacion_final.csv", index=False, encoding="utf-8-sig")
    pd.concat(todas_las_variantes, ignore_index=True).to_csv(
        SALIDA / "variantes_validacion_cruzada.csv", index=False, encoding="utf-8-sig"
    )
    print("\n===== RESUMEN FINAL =====")
    print(tabla.to_string(index=False))
    print(f"\nResultados guardados en: {SALIDA}")


if __name__ == "__main__":
    main()
