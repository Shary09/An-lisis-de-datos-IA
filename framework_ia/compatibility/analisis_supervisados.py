"""Interpretaciones descriptivas y verificables de la clasificación binaria.

No infiere causalidad ni atribuye significados desconocidos a columnas.
"""
import numpy as np
import pandas as pd


def _pct(valor):
    return f"{100 * float(valor):.1f} %"


def _nombre(clase):
    return str(clase)


def _etiquetas_clase(df, objetivo, clases):
    """Devuelve etiquetas más naturales si el objetivo es reconocible.

    Solo traduce algunos casos evidentes; en otros mantiene las clases originales.
    """
    lower_obj = str(objetivo).strip().lower()
    clases_list = list(clases)
    etiquetas = {_nombre(c): _nombre(c) for c in clases_list}

    if len(clases_list) != 2:
        return etiquetas

    if lower_obj in {"survived", "supervivencia", "survive", "sobrevivio", "sobrevivió"}:
        valores = {str(c) for c in clases_list}
        if valores == {"0", "1"}:
            etiquetas["0"] = "no sobrevivió"
            etiquetas["1"] = "sobrevivió"
            return etiquetas

    valores = {str(c) for c in clases_list}
    if valores == {"0", "1"}:
        etiquetas["0"] = "clase 0"
        etiquetas["1"] = "clase 1"
    return etiquetas


def introduccion_datos(df, objetivo):
    datos = df.dropna(subset=[objetivo])
    clases = datos[objetivo].value_counts(dropna=False)
    predictores = [c for c in df.columns if c != objetivo]
    lines = [
        f"El conjunto cargado contiene {len(df)} registros y {len(df.columns)} columnas. "
        f"Se busca predecir '{objetivo}' utilizando {len(predictores)} variables predictoras.",
        f"Hay {len(datos)} registros con valor conocido en la variable objetivo; "
        f"{len(df)-len(datos)} registros sin objetivo se excluyen del entrenamiento y la evaluación.",
        "Distribución de la variable objetivo: " + "; ".join(
            f"{_nombre(k)}: {int(v)} casos ({100*v/len(datos):.1f} %)" for k, v in clases.items()
        ) + "." if len(datos) else "No existen registros con objetivo conocido.",
    ]
    if len(clases) == 2 and len(datos):
        mayor = clases.max() / len(datos)
        if mayor >= 0.70:
            lines.append(
                "Las clases no aparecen en proporciones similares; por eso conviene prestar especial atención al F1 macro y a los errores por clase, no solo al accuracy."
            )
    lines.append(
        "Los nombres de las variables se conservan tal como aparecen en el CSV. Su significado real depende de la documentación del conjunto de datos."
    )
    return "\n\n".join(lines)


def describir_metricas_una_vez():
    return (
        "Las métricas reportadas se interpretan así: accuracy representa la proporción total de aciertos; "
        "precisión macro indica qué proporción de las predicciones de cada clase fue correcta; recall macro "
        "muestra qué proporción de los casos reales de cada clase fue reconocida; y F1 macro resume el equilibrio "
        "entre precisión y recall dando el mismo peso a ambas clases."
    )


def describir_matriz(resultado, df=None, objetivo=None):
    clases = resultado["clases"]
    m = np.asarray(resultado["metricas"]["matriz_confusion"])
    if m.shape != (2, 2):
        return "La matriz requiere dos clases para esta interpretación."
    etiquetas = _etiquetas_clase(df, objetivo, clases)
    total = int(m.sum())
    texto = [
        f"En la matriz de confusión de {resultado['algoritmo']}, las filas representan las clases reales y las columnas las clases predichas. "
        f"De {total} casos de prueba, el modelo acertó {int(np.trace(m))} y se equivocó {int(total - np.trace(m))}."
    ]
    for i, clase in enumerate(clases):
        cantidad = int(m[i].sum())
        correctos = int(m[i, i])
        incorrectos = cantidad - correctos
        tasa = correctos / cantidad if cantidad else 0
        clase_txt = etiquetas[_nombre(clase)]
        destino = etiquetas[_nombre(clases[1 - i])]
        if cantidad:
            texto.append(
                f"Para {clase_txt}, el modelo acertó {correctos} de {cantidad} casos reales ({_pct(tasa)}) y confundió {incorrectos} con {destino}."
            )
        else:
            texto.append(f"{clase_txt.capitalize()} no tuvo casos en el conjunto de prueba.")
    if m[0].sum() and m[1].sum():
        recalls = [m[i, i] / m[i].sum() for i in range(2)]
        peor = int(np.argmin(recalls))
        if abs(recalls[0] - recalls[1]) >= 0.05:
            texto.append(
                f"Los aciertos no se distribuyen por igual: {etiquetas[_nombre(clases[peor])]} fue la clase más difícil de reconocer para este modelo."
            )
        else:
            texto.append(
                "El porcentaje de aciertos por clase fue relativamente parecido en este conjunto de prueba."
            )
    return "\n\n".join(texto)


def analizar_configuraciones(resultado):
    variaciones = resultado["variaciones"].copy()
    params = resultado["mejores_parametros"]
    texto = [
        f"Se compararon {len(variaciones)} configuraciones de {resultado['algoritmo']} mediante validación cruzada estratificada. La mejor alcanzó un F1 macro promedio de {_pct(resultado['f1_macro_cv'])} en entrenamiento."
    ]

    limpio = [f"{k.replace('clasificador__', '')}={v}" for k, v in params.items()]
    texto.append("La configuración seleccionada fue: " + ", ".join(limpio) + ".")

    if "es_baseline" in resultado:
        base = resultado["es_baseline"]
        texto.append(
            f"Como referencia, la configuración estándar del algoritmo obtuvo un F1 macro promedio de {_pct(base['f1_macro_cv'])}."
        )
        diferencia = (resultado["f1_macro_cv"] - base["f1_macro_cv"]) * 100
        if abs(diferencia) < 0.2:
            texto.append(
                "La diferencia entre la configuración estándar y la mejor configuración fue mínima dentro de las pruebas realizadas."
            )
        else:
            texto.append(
                f"La mejor configuración superó a la estándar por {diferencia:.2f} puntos porcentuales en F1 macro de validación cruzada."
            )

    if len(variaciones) > 1:
        minimo = float(variaciones["F1 macro CV"].min())
        maximo = float(variaciones["F1 macro CV"].max())
        texto.append(
            f"Entre las configuraciones probadas, el F1 macro promedio varió de {_pct(minimo)} a {_pct(maximo)}. Esto muestra cuánto influyó la elección de parámetros dentro de las opciones evaluadas, sin demostrar que sea la mejor configuración posible de todas."
        )
        if maximo - minimo < 0.01:
            texto.append(
                "Las diferencias entre configuraciones fueron pequeñas, por lo que no se puede atribuir una ventaja importante a estos parámetros solo con esta comparación."
            )

        empatadas = variaciones[np.isclose(variaciones["F1 macro CV"], maximo)]
        if len(empatadas) > 1:
            texto.append(
                f"Se observó un empate en el mejor resultado de validación cruzada: {len(empatadas)} configuraciones alcanzaron el mismo F1 macro promedio máximo."
            )
            for _, fila in empatadas.iterrows():
                texto.append(f"- {fila['Parametros']}")

    desviacion = float(variaciones.iloc[0]["Desviacion CV"]) if len(variaciones) else 0
    texto.append(
        f"La desviación estándar entre particiones de la mejor configuración fue de {desviacion:.3f}; valores más altos indican resultados más variables entre particiones."
    )
    return "\n\n".join(texto)


def analizar_modelo(resultado, df=None, objetivo=None):
    m = resultado["metricas"]
    cv = float(resultado["f1_macro_cv"])
    prueba = float(m["f1_macro"])
    dif = prueba - cv
    texto = [
        f"{resultado['algoritmo']} obtuvo en prueba un accuracy de {_pct(m['accuracy'])}, una precisión macro de {_pct(m['precision_macro'])}, un recall macro de {_pct(m['recall_macro'])} y un F1 macro de {_pct(prueba)}.",
        f"El F1 macro en prueba fue {abs(dif)*100:.1f} puntos porcentuales {'mayor' if dif >= 0 else 'menor'} que el promedio de validación cruzada. Esta diferencia puede reflejar variaciones entre particiones; por sí sola no permite confirmar ni descartar sobreajuste.",
        analizar_configuraciones(resultado),
        describir_matriz(resultado, df=df, objetivo=objetivo),
    ]
    return "\n\n".join(texto)


def analizar_global(resultados, df, objetivo):
    if not resultados:
        return "No hay resultados para analizar."
    ordenados = sorted(resultados.items(), key=lambda item: float(item[1]["f1_macro_cv"]), reverse=True)
    nombre_ganador, ganador = ordenados[0]
    lineas = [
        "ANÁLISIS DE CLASIFICACIÓN SUPERVISADA",
        "",
        "1. Descripción de los datos",
        introduccion_datos(df, objetivo),
        "",
        "2. Metodología",
        "Se utilizó una partición estratificada de 80 % para entrenamiento y 20 % para prueba, con semilla 42. La selección de hiperparámetros se realizó mediante validación cruzada estratificada de cinco particiones, usando F1 macro. Los resultados finales del conjunto de prueba se reservaron para evaluar los modelos ya seleccionados. Se aplicaron imputación y preprocesamiento dentro del pipeline para evitar utilizar información de prueba al entrenar.",
        describir_metricas_una_vez(),
        "",
        "3. Comparación global",
    ]
    lineas.append(
        f"El modelo con el mayor F1 macro promedio en validación cruzada fue {nombre_ganador} ({_pct(ganador['f1_macro_cv'])}). Se selecciona con base en entrenamiento, no por haber obtenido el mejor resultado en prueba."
    )
    if len(ordenados) > 1:
        segundo_n, segundo = ordenados[1]
        dif = (ganador["f1_macro_cv"] - segundo["f1_macro_cv"]) * 100
        lineas.append(
            f"El siguiente fue {segundo_n} ({_pct(segundo['f1_macro_cv'])}), con una diferencia de {dif:.2f} puntos porcentuales. "
            + (
                "La distancia es pequeña y no permite asegurar una superioridad estadística."
                if dif < 2
                else "La diferencia observada favorece al primero dentro de estas pruebas, pero no demuestra superioridad en cualquier otro conjunto de datos."
            )
        )
    for idx, (nombre, res) in enumerate(ordenados, start=1):
        lineas.extend(["", f"4.{idx}. {nombre} — evaluación y configuraciones", analizar_modelo(res, df=df, objetivo=objetivo)])
    lineas.extend([
        "",
        "5. Conclusión",
        f"Dentro de las configuraciones evaluadas y para este conjunto de datos, {nombre_ganador} es la elección inicial según F1 macro de validación cruzada. Antes de usarlo en un escenario real, deben revisarse los errores de su matriz de confusión, la importancia de cada clase y la posibilidad de evaluar nuevos datos. Los resultados no establecen relaciones causales entre las columnas y la variable objetivo.",
    ])
    return "\n\n".join(lineas)
