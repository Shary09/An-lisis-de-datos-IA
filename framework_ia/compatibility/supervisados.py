"""Algoritmos de clasificacion del Laboratorio 2 - EIF420O."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Supervisados:
    def __init__(self, df, objetivo="survived", semilla=42):
        self.df = df.copy()
        self.objetivo = objetivo
        self.semilla = semilla
        if objetivo not in self.df.columns:
            raise ValueError(f"No existe la columna objetivo: {objetivo}")

    def validar_dataset(self):
        """Valida problemas que impiden una clasificación binaria reproducible."""
        if self.df.empty:
            raise ValueError("El CSV no contiene filas para analizar.")
        if len(self.df.columns) < 2:
            raise ValueError("Se necesita al menos una columna predictora y una columna objetivo.")
        if not self.df.columns.is_unique:
            raise ValueError("El CSV tiene nombres de columnas repetidos. Renómbralos antes de entrenar.")
        datos = self.df.dropna(subset=[self.objetivo]).copy()
        if datos.empty:
            raise ValueError("No hay registros con valor conocido en la variable objetivo.")
        if datos[self.objetivo].nunique(dropna=True) != 2:
            raise ValueError("El laboratorio admite clasificación binaria: el objetivo debe tener exactamente dos clases.")
        X = datos.drop(columns=[self.objetivo])
        if X.shape[1] == 0:
            raise ValueError("Se requiere al menos una variable predictora.")
        # Convertir infinitos numéricos en ausentes antes de procesar el CSV.
        columnas_num = X.select_dtypes(include="number").columns
        if len(columnas_num):
            X.loc[:, columnas_num] = X[columnas_num].replace([np.inf, -np.inf], np.nan)
        if X.isna().all().all():
            raise ValueError("Todas las variables predictoras están vacías.")
        # La clase minoritaria debe permitir la división estratificada y las 5 particiones CV.
        cuentas = datos[self.objetivo].value_counts()
        if cuentas.min() < 7:
            raise ValueError(
                "Cada clase necesita al menos 7 registros para usar 80/20 y validación cruzada de 5 particiones."
            )
        return datos

    def dividir_datos(self, test_size=0.20):
        datos = self.validar_dataset()
        X = datos.drop(columns=[self.objetivo]).copy()
        y = datos[self.objetivo]
        # Limpiar valores numéricos no finitos y datos mixtos antes de separar.
        numericas = X.select_dtypes(include="number").columns
        if len(numericas):
            X.loc[:, numericas] = X[numericas].replace([np.inf, -np.inf], np.nan)
        for columna in X.columns.difference(numericas, sort=False):
            # Asegura categorías homogéneas aunque el CSV mezcle cadenas y números.
            X[columna] = X[columna].map(
                lambda v: np.nan if pd.isna(v) else str(v)
            ).astype(object)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.semilla, stratify=y
        )
        # Eliminar columnas vacías según SOLO entrenamiento; mismos cambios en prueba.
        vacias_entrenamiento = X_train.columns[X_train.isna().all()].tolist()
        if vacias_entrenamiento:
            X_train = X_train.drop(columns=vacias_entrenamiento)
            X_test = X_test.drop(columns=vacias_entrenamiento)
        if X_train.shape[1] == 0:
            raise ValueError("Después de separar los datos, todas las variables predictoras están vacías en entrenamiento.")
        return X_train, X_test, y_train, y_test

    @staticmethod
    def preparar_columnas(X):
        numericas = X.select_dtypes(include="number").columns.tolist()
        categoricas = X.select_dtypes(exclude="number").columns.tolist()
        transformaciones = []
        if numericas:
            transformaciones.append((
                "numericas",
                Pipeline([
                    ("imputar", SimpleImputer(strategy="median")),
                    ("escalar", StandardScaler()),
                ]),
                numericas,
            ))
        if categoricas:
            transformaciones.append((
                "categoricas",
                Pipeline([
                    ("imputar", SimpleImputer(strategy="most_frequent")),
                    ("codificar", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categoricas,
            ))
        return ColumnTransformer(transformaciones)

    def crear_modelo(self, nombre, X, **parametros):
        algoritmos = {
            "KNN": KNeighborsClassifier,
            "Arbol de Decision": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier,
            "XGBoost": XGBClassifier,
            "AdaBoost": AdaBoostClassifier,
        }
        if nombre not in algoritmos:
            raise ValueError(f"Algoritmo desconocido: {nombre}")
        if nombre in ("Arbol de Decision", "Random Forest", "XGBoost", "AdaBoost"):
            parametros.setdefault("random_state", self.semilla)
        if nombre == "XGBoost":
            parametros.setdefault("eval_metric", "logloss")
            parametros.setdefault("n_jobs", 1)
        return Pipeline([
            ("preprocesamiento", self.preparar_columnas(X)),
            ("clasificador", algoritmos[nombre](**parametros)),
        ])

    @staticmethod
    def evaluar(y_real, predicho):
        return {
            "accuracy": accuracy_score(y_real, predicho),
            "precision_macro": precision_score(y_real, predicho, average="macro", zero_division=0),
            "recall_macro": recall_score(y_real, predicho, average="macro", zero_division=0),
            "f1_macro": f1_score(y_real, predicho, average="macro", zero_division=0),
            "matriz_confusion": confusion_matrix(y_real, predicho),
        }

    def _evaluar_configuracion_estandar(self, algoritmo, X_train, y_train):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.semilla)
        modelo = self.crear_modelo(algoritmo, X_train)
        puntajes = cross_val_score(
            modelo, X_train, y_train,
            scoring="f1_macro",
            cv=cv,
            n_jobs=1 if algoritmo == "XGBoost" else -1,
            error_score="raise",
        )
        return {
            "parametros": "Parámetros estándar del algoritmo",
            "f1_macro_cv": float(puntajes.mean()),
            "desviacion_cv": float(puntajes.std()),
        }

    def comparar_modelo(self, objetivo=None, algoritmo="KNN"):
        """Busca la mejor configuración usando solo entrenamiento y evalúa en prueba."""
        if objetivo is not None and objetivo != self.objetivo:
            laboratorio = Supervisados(self.df, objetivo=objetivo, semilla=self.semilla)
            return laboratorio.comparar_modelo(algoritmo=algoritmo)

        grillas = {
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
        if algoritmo not in grillas:
            raise ValueError(f"Algoritmo desconocido: {algoritmo}")
        X_train, X_test, y_train, y_test = self.dividir_datos()
        clases_originales = list(pd.Index(y_train.unique()).sort_values())
        baseline = None
        if algoritmo == "XGBoost":
            codificador = LabelEncoder().fit(y_train)
            clases_originales = list(codificador.classes_)
            y_train = codificador.transform(y_train)
            y_test = codificador.transform(y_test)
        if pd.Series(y_train).value_counts().min() < 5:
            raise ValueError("Cada clase necesita al menos 5 registros de entrenamiento para validación cruzada de 5 particiones. Usa un CSV con más casos por clase.")
        baseline = self._evaluar_configuracion_estandar(algoritmo, X_train, y_train)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.semilla)
        busqueda = GridSearchCV(
            self.crear_modelo(algoritmo, X_train),
            param_grid=grillas[algoritmo], scoring="f1_macro",
            cv=cv, refit=True,
            n_jobs=1 if algoritmo == "XGBoost" else -1,
            error_score="raise",
        )
        busqueda.fit(X_train, y_train)
        predicciones = busqueda.predict(X_test)
        detalle = pd.DataFrame(busqueda.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].sort_values(["rank_test_score", "mean_test_score"], ascending=[True, False]).reset_index(drop=True)
        detalle.columns = ["Parametros", "F1 macro CV", "Desviacion CV", "Posicion"]
        detalle["Parametros"] = detalle["Parametros"].astype(str)
        return {
            "algoritmo": algoritmo,
            "objetivo": self.objetivo,
            "mejores_parametros": busqueda.best_params_,
            "f1_macro_cv": float(busqueda.best_score_),
            "metricas": self.evaluar(y_test, predicciones),
            "variaciones": detalle,
            "clases": clases_originales,
            "n_entrenamiento": len(X_train),
            "n_prueba": len(X_test),
            "es_baseline": baseline,
        }
