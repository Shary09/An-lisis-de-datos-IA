"""Exportación automática de experimentos de clasificación supervisada.

Cada ejecución crea un directorio independiente para evitar sobrescrituras.
"""
from datetime import datetime
from pathlib import Path
import json
import re
import unicodedata
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analisis_supervisados import analizar_global


def _seguro(nombre):
    texto = unicodedata.normalize("NFKD", str(nombre))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^A-Za-z0-9_-]+", "_", texto).strip("_-")
    return (texto[:75] or "dataset")


def guardar_comparacion(resultados, datos, objetivo, nombre_csv, raiz):
    """Guarda métricas, variantes, matrices y análisis; devuelve la ruta creada.

    Los resultados se guardan en raiz/<nombre_csv>/<fecha-hora>_<id>/.
    """
    if not resultados:
        raise ValueError("No hay resultados de comparación para guardar.")
    raiz = Path(raiz)
    nombre = _seguro(Path(nombre_csv or "dataset.csv").stem)
    marca = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    carpeta = raiz / nombre / f"{marca}_{uuid.uuid4().hex[:6]}"
    carpeta.mkdir(parents=True, exist_ok=False)

    tabla = []
    variantes = []
    for modelo, resultado in resultados.items():
        metricas = resultado["metricas"]
        tabla.append({
            "modelo": modelo,
            "f1_macro_cv": resultado["f1_macro_cv"],
            "accuracy_prueba": metricas["accuracy"],
            "precision_macro_prueba": metricas["precision_macro"],
            "recall_macro_prueba": metricas["recall_macro"],
            "f1_macro_prueba": metricas["f1_macro"],
            "mejores_parametros": json.dumps(resultado["mejores_parametros"], ensure_ascii=False),
            "f1_macro_cv_estandar": resultado.get("es_baseline", {}).get("f1_macro_cv"),
        })
        for _, fila in resultado["variaciones"].iterrows():
            variantes.append({
                "modelo": modelo,
                "params": fila["Parametros"],
                "mean_test_score": fila["F1 macro CV"],
                "std_test_score": fila["Desviacion CV"],
                "rank_test_score": fila["Posicion"],
            })
        fig, ax = plt.subplots(figsize=(6.5, 5))
        try:
            sns.heatmap(
                metricas["matriz_confusion"], annot=True, fmt="d", cmap="Blues",
                xticklabels=resultado["clases"], yticklabels=resultado["clases"], ax=ax,
            )
            ax.set_title(f"Matriz de confusión - {modelo}")
            ax.set_xlabel("Clase predicha")
            ax.set_ylabel("Clase real")
            fig.tight_layout()
            fig.savefig(carpeta / f"matriz_{_seguro(modelo).lower()}.png", dpi=150)
        finally:
            plt.close(fig)

    tabla_df = pd.DataFrame(tabla).sort_values("f1_macro_cv", ascending=False)
    tabla_df.to_csv(carpeta / "comparacion_final.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(variantes).to_csv(
        carpeta / "variantes_validacion_cruzada.csv", index=False, encoding="utf-8-sig"
    )
    (carpeta / "analisis_completo.txt").write_text(
        analizar_global(resultados, datos, objetivo), encoding="utf-8"
    )
    (carpeta / "informacion_ejecucion.json").write_text(
        json.dumps({
            "archivo_csv": nombre_csv,
            "variable_objetivo": objetivo,
            "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
            "filas": len(datos),
            "columnas": list(map(str, datos.columns)),
            "semilla": 42,
            "particion": "80 % entrenamiento / 20 % prueba",
            "validacion": "StratifiedKFold, 5 particiones; F1 macro",
            "algoritmos": list(resultados),
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return carpeta
