# app_hr_logreg_es.py
# Aplicación GUI en CustomTkinter para entrenar y usar un modelo de regresión logística
# Comentarios en español explicando cada línea o bloque.

# --- Imports del sistema y librerías ---
import os  # operaciones con el sistema de archivos (comprobar existencia, nombres)
import threading  # para ejecutar el entrenamiento en un hilo y no bloquear la UI
import tkinter as tk  # tkinter base (se usa para diálogos nativos si hace falta)
from tkinter import filedialog, messagebox  # diálogos para abrir archivos y mostrar mensajes

import customtkinter as ctk  # librería CustomTkinter para una UI moderna
import joblib  # para guardar y cargar modelos/pipelines (serialización)
import matplotlib  # matplotlib para graficar
import matplotlib.pyplot as plt  # interfaz pyplot para crear figuras
import numpy as np  # utilidades numéricas
import pandas as pd  # manipulación de datos con DataFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # incrustar figuras en Tkinter

# Importaciones de scikit-learn para preprocesado, modelo y métricas
from sklearn.compose import ColumnTransformer  # combinar transformaciones por columnas
from sklearn.impute import SimpleImputer  # imputación de valores faltantes
from sklearn.linear_model import LogisticRegression  # modelo de regresión logística
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             average_precision_score, classification_report,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve)  # métricas y utilidades
from sklearn.model_selection import GridSearchCV, train_test_split  # búsqueda de hiperparámetros y split
from sklearn.pipeline import Pipeline  # pipeline de scikit-learn
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # codificación y escalado
from sklearn.cluster import KMeans  # clustering para silhouette (Se utilizo previo a la eleccion del modelo, ignorar)
from sklearn.metrics import silhouette_score  # métrica de silhouette(Ignorar, tambien era para ottro modelo)

# Forzar backend compatible con Tkinter
matplotlib.use("TkAgg")

# Configuración visual de CustomTkinter
ctk.set_appearance_mode("dark")  # modo oscuro por defecto
ctk.set_default_color_theme("blue")  # tema de color

# Constantes y sincronización
MODEL_FILENAME = "mejor_modelo_logreg.joblib"  # nombre de archivo por defecto para guardar el pipeline
PROCESS_THREAD_LOCK = threading.Lock()  # lock para evitar entrenamientos concurrentes

# Clase principal de la aplicación
class AplicacionHR(ctk.CTk):
    def __init__(self):
        # Constructor: inicializa la ventana principal y todos los widgets
        super().__init__()  # inicializa la clase base CTk
        self.title("Deserción de Empleados - Regresión Logística")  # título de la ventana
        self.geometry("1100x700")  # tamaño inicial de la ventana (ancho x alto)

        # Variables internas para almacenar datos y estado
        self.df = None  # DataFrame completo cargado desde CSV
        self.X = None  # DataFrame con features (sin la columna objetivo)
        self.y = None  # Serie con la columna objetivo
        self.feature_names = []  # lista ordenada de nombres de features
        self.pipeline = None  # pipeline entrenado (preprocesado + modelo)
        self.best_estimator_ = None  # referencia al mejor estimador tras GridSearch
        self.categorical_features = []  # lista de columnas categóricas
        self.numerical_features = []  # lista de columnas numéricas

        # Barra superior con botones de control
        top_frame = ctk.CTkFrame(self)  # frame contenedor superior
        top_frame.pack(side="top", fill="x", padx=12, pady=8)  # empaquetado en la parte superior

        # Botón para cargar CSV (llama a self.cargar_csv)
        self.load_btn = ctk.CTkButton(top_frame, text="Cargar CSV", command=self.cargar_csv)
        self.load_btn.pack(side="left", padx=(0, 8))

        # Botón para iniciar entrenamiento (se ejecuta en hilo)
        self.train_btn = ctk.CTkButton(top_frame, text="Entrenar modelo", command=self.iniciar_entrenamiento_thread)
        self.train_btn.pack(side="left", padx=(0, 8))

        # Botón para guardar el modelo entrenado en disco
        self.save_btn = ctk.CTkButton(top_frame, text="Guardar modelo", command=self.guardar_modelo)
        self.save_btn.pack(side="left", padx=(0, 8))

        # Botón para cargar un modelo desde archivo
        self.load_model_btn = ctk.CTkButton(top_frame, text="Cargar modelo (archivo)", command=self.cargar_modelo_archivo)
        self.load_model_btn.pack(side="left", padx=(0, 8))

        # Etiqueta de estado a la derecha
        self.status_label = ctk.CTkLabel(top_frame, text="Estado: listo")
        self.status_label.pack(side="right")

        # Vista principal con pestañas
        self.tab_view = ctk.CTkTabview(self, width=1000)  # contenedor de pestañas
        self.tab_view.pack(expand=True, fill="both", padx=12, pady=12)

        # Añadir pestañas: "Explorar y Gráficas" y "Predecir"
        self.tab_view.add("Explorar y Gráficas")
    

        # Pestaña Explorar y Gráficas
        self.explore_frame = self.tab_view.tab("Explorar y Gráficas")  # frame de la pestaña
        left_explore = ctk.CTkFrame(self.explore_frame, width=350)  # panel izquierdo
        left_explore.pack(side="left", fill="y", padx=8, pady=8)

        # Caja de texto para mostrar información del dataset y reportes
        self.df_info_text = ctk.CTkTextbox(left_explore, width=330, height=300)
        self.df_info_text.pack(padx=6, pady=6)

        # Botón para generar gráficas (usa resultados del último entrenamiento)
        self.plot_btn = ctk.CTkButton(left_explore, text="Generar gráficas (requiere modelo entrenado)", command=self.generar_graficas)
        self.plot_btn.pack(padx=6, pady=(12, 6))



        # Panel derecho para las figuras
        right_explore = ctk.CTkFrame(self.explore_frame)
        right_explore.pack(side="right", expand=True, fill="both", padx=8, pady=8)

        # Figura matplotlib que se incrustará en la UI
        self.fig = plt.Figure(figsize=(7, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_explore)  # canvas para Tkinter
        self.canvas.get_tk_widget().pack(expand=True, fill="both")  # empaquetar el widget del canvas

        # Cargar modelo automáticamente si existe el archivo
        if os.path.exists(MODEL_FILENAME):
            try:
                self.cargar_modelo(MODEL_FILENAME)  # intenta cargar el pipeline guardado
                self.estado("Modelo cargado automáticamente desde archivo.")
            except Exception:
                # Si falla la carga, se muestra estado informativo
                self.estado("Modelo guardado pero no pudo cargarse automáticamente.")

    
    def estado(self, texto):
        # Actualiza la etiqueta de estado en la UI para que se entienda el proceso
        self.status_label.configure(text=f"Estado: {texto}")

    # Cargar CSV
    def cargar_csv(self):
        # Abre diálogo para seleccionar archivo CSV
        ruta = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")])
        if not ruta:
            return  # si el usuario cancela, salir

        try:
            # Lee el CSV en un DataFrame de pandas
            self.df = pd.read_csv(ruta)
        except Exception as e:
            # Si ocurre un error al leer, mostrar mensaje y salir
            messagebox.showerror("Error", f"No se pudo leer el CSV: {e}")
            return

        # Mostrar información básica del DataFrame en la caja de texto
        self.df_info_text.delete("0.0", "end")  # limpiar caja de texto
        info = []
        info.append(f"Archivo: {os.path.basename(ruta)}")  # nombre del archivo
        info.append(f"Filas: {self.df.shape[0]}  Columnas: {self.df.shape[1]}")  # dimensiones
        info.append("\nColumnas y tipos:")
        for c, t in zip(self.df.columns, self.df.dtypes):
            info.append(f"- {c}: {t}")  # listar columnas y tipos
        info_text = "\n".join(info)
        self.df_info_text.insert("0.0", info_text)  # insertar texto en la caja

        # Intentar inferir la columna objetivo por nombres comunes
        posibles = [c for c in self.df.columns if c.lower() in ("attrition", "desertion", "desert", "is_left", "target", "label", "deserto", "desercion", "abandono")]
        if not posibles:
            objetivo = self.df.columns[-1]  # si no encuentra, usar la última columna
        else:
            objetivo = posibles[0]  # si encuentra, usar la primera coincidencia

        # Actualizar estado indicando la columna objetivo elegida
        self.estado(f"CSV cargado. Usando objetivo: {objetivo}")

        # Separar X (features) e y (target)
        self.y = self.df[objetivo]
        self.X = self.df.drop(columns=[objetivo])

        # Detectar columnas categóricas y numéricas automáticamente ( implementado previo a la version final, no conservamos categoricos tras el procesamiento)
        self.categorical_features = list(self.X.select_dtypes(include=["object", "category"]).columns)
        self.numerical_features = list(self.X.select_dtypes(include=[np.number]).columns)

        # Guardar orden de features (numéricas primero, luego categóricas)
        self.feature_names = self.numerical_features + self.categorical_features

        # Construir el formulario de predicción con widgets para cada feature
        self.construir_formulario_prediccion()

        # Actualizar estado final
        self.estado("CSV listo. Puedes entrenar el modelo.")

    # Construir formulario de predicción (Seccion no implementada, hay que saltarla)
    def construir_formulario_prediccion(self):
        # Buscar el frame de formulario dentro de la pestaña Predecir
        form_frame = None
        for child in self.predict_frame.winfo_children():
            if isinstance(child, ctk.CTkFrame):
                # heurística sencilla: tomar el primer CTkFrame como form_frame
                form_frame = child
                break
        if form_frame is None:
            # si no existe, crear uno nuevo
            form_frame = ctk.CTkFrame(self.predict_frame)
            form_frame.pack(side="left", fill="y", padx=12, pady=12)

        # Limpiar widgets previos en el formulario
        for w in form_frame.winfo_children():
            w.destroy()

        # Reiniciar diccionario de entradas
        self.input_entries = {}

        # Título del formulario
        lbl = ctk.CTkLabel(form_frame, text="Ingrese datos para predecir", font=ctk.CTkFont(size=16, weight="bold"))
        lbl.pack(pady=(4, 8))

        # Para cada feature crear un widget apropiado (Entry para numéricas, OptionMenu para categóricas)
        for feat in self.feature_names:
            frame = ctk.CTkFrame(form_frame)  # frame horizontal para etiqueta + widget
            frame.pack(fill="x", pady=4, padx=6)
            label = ctk.CTkLabel(frame, text=f"{feat}:", width=160, anchor="w")  # etiqueta con nombre de feature
            label.pack(side="left", padx=4)

            if feat in self.categorical_features:
                # Si es categórica, crear un OptionMenu con valores únicos del dataset
                unique_vals = list(self.X[feat].dropna().unique().astype(str))
                if not unique_vals:
                    unique_vals = [""]  # si no hay valores, dejar una opción vacía
                opt = ctk.CTkOptionMenu(frame, values=unique_vals)
                opt.set(unique_vals[0])  # seleccionar el primer valor por defecto
                opt.pack(side="right", padx=4)
                self.input_entries[feat] = opt  # guardar widget en el diccionario
            else:
                # Si es numérica, crear una entrada de texto y prellenarla con la media si es posible
                entry = ctk.CTkEntry(frame)
                try:
                    meanv = float(self.X[feat].dropna().astype(float).mean())
                    entry.insert(0, f"{meanv:.3f}")  # insertar media como valor por defecto
                except Exception:
                    # si no se puede calcular la media (por ejemplo strings), no hacer nada
                    pass
                entry.pack(side="right", fill="x", expand=True, padx=4)
                self.input_entries[feat] = entry  # guardar widget

    # Entrenamiento en hilo (esto es para no bloquear la UI, funciona mejor :) )
    def iniciar_entrenamiento_thread(self):
        # Verificar que haya un DataFrame cargado
        if self.df is None:
            messagebox.showwarning("Aviso", "Primero cargue un archivo CSV.")
            return
        # Evitar iniciar múltiples entrenamientos simultáneos
        if PROCESS_THREAD_LOCK.locked():
            messagebox.showinfo("Información", "Ya hay un entrenamiento en proceso.")
            return
        # Crear y arrancar el hilo que ejecutará self.entrenar_modelo
        thread = threading.Thread(target=self.entrenar_modelo, daemon=True)
        thread.start()

    def entrenar_modelo(self):
        # Ejecutar el entrenamiento dentro del lock para evitar concurrencia (se bloqueo para evitar errores)
        with PROCESS_THREAD_LOCK:
            try:
                self.estado("Entrenando...")  # actualizar estado en la UI

                # Dividir datos en entrenamiento y prueba (20% test), manteniendo proporciones de clases
                X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)


                #(De nuevo esta parte fue previa al procesamiento pero no es necesaria, ignorar)
                # Preprocesado numérico: imputación y escalado
                numeric_transformer = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),  # imputar con la mediana
                        ("scaler", StandardScaler()),  # estandarizar (media 0, varianza 1)
                    ]
                )

                # Preprocesado categórico: imputación y one-hot encoding
                categorical_transformer = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),  # imputar con la moda
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),  # codificar categorías
                    ]
                )

                # ColumnTransformer que aplica transformaciones por tipo de columna
                preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer, self.numerical_features),
                        ("cat", categorical_transformer, self.categorical_features),
                    ],
                    remainder="drop",  # descartar columnas no listadas
                )





                # Instanciar el clasificador: Regresión Logística
                logreg = LogisticRegression(max_iter=1000, solver="liblinear")

                # Pipeline completo: preprocesado + clasificador
                pipe = Pipeline(steps=[("pre", preprocessor), ("clf", logreg)])

                # Grid de hiperparámetros para buscar la mejor C y penalización
                param_grid = {
                    "clf__C": [0.01, 0.1, 1, 10],
                    "clf__penalty": ["l1", "l2"],
                }

                # GridSearchCV para buscar la mejor combinación (investigando esto optimizaba F1)
                grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
                grid.fit(X_train, y_train)  # ejecutar búsqueda y entrenamiento

                # Guardar el mejor pipeline encontrado
                self.pipeline = grid.best_estimator_
                self.best_estimator_ = grid.best_estimator_

                # Evaluación en el conjunto de prueba
                y_pred = self.pipeline.predict(X_test)  # predicciones discretas
                # si el pipeline soporta predict_proba, pues que se obtenga la probabilidades de la clase positiva
                y_proba = self.pipeline.predict_proba(X_test)[:, 1] if hasattr(self.pipeline, "predict_proba") else None

                # Calcular métricas principales (Todo lo posible)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
                avg_prec = average_precision_score(y_test, y_proba) if y_proba is not None else None

                # Se guardan los resultados solo para mostrar
                self.last_eval = {
                    "X_test": X_test,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                    "acc": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "roc_auc": roc_auc,
                    "avg_precision": avg_prec,
                }

                #(Ignorar bloque)
                #(Sigue siendo silhouette)
                # Este bloque intenta calcular un score de silhouette como indicio de estructura de clusters
                if len(self.numerical_features) >= 2:
                    # Preparar matriz numérica: imputar y escalar
                    X_cluster = self.X[self.numerical_features].fillna(self.X[self.numerical_features].median())
                    X_cluster = StandardScaler().fit_transform(X_cluster)
                    sil_score = None
                    best_k = None
                    try:
                        # Probar k de 2 hasta 10 (o hasta N-1)
                        for k in range(2, min(10, X_cluster.shape[0] - 1) + 1):
                            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cluster)
                            s = silhouette_score(X_cluster, km.labels_)
                            if sil_score is None or s > sil_score:
                                sil_score = s
                                best_k = k
                        self.last_eval["silhouette"] = {"score": sil_score, "best_k": best_k}
                    except Exception:
                        # Si falla el cálculo, guardar None
                        self.last_eval["silhouette"] = {"score": None, "best_k": None}
                else:
                    # Si no hay suficientes features numéricas, no calcular silhouette
                    self.last_eval["silhouette"] = {"score": None, "best_k": None}

                # Notificacion de resultados (Se presenta la penalizacion y C resultantes)
                self.estado(f"Entrenado. Acc={acc:.3f} F1={f1:.3f}")
                messagebox.showinfo("Entrenamiento completado", f"Mejor modelo: {grid.best_params_}\nAccuracy: {acc:.3f}\nF1: {f1:.3f}")
            except Exception as e:
                # Manejo de errores: mostrar mensaje y actualizar estado
                messagebox.showerror("Error de entrenamiento", str(e))
                self.estado("Error en entrenamiento")

    #  Generar gráficas 
    def generar_graficas(self):
        # Verificar que exista evaluación previa (entrenamiento)
        if not hasattr(self, "last_eval"):
            messagebox.showwarning("Aviso", "Entrene el modelo primero.")
            return

        # Limpiar la figura actual
        self.fig.clf()
        # Crear una cuadrícula 2x2 de ejes
        axs = self.fig.subplots(2, 2)
        X_test = self.last_eval["X_test"]
        y_test = self.last_eval["y_test"]
        y_pred = self.last_eval["y_pred"]
        y_proba = self.last_eval["y_proba"]

        # Matriz de confusión
        ax = axs[0, 0]
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap=plt.cm.Blues, display_labels=["No", "Sí"])
        ax.set_title("Matriz de confusión")

        # Curva ROC 
        ax = axs[0, 1]
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax.plot(fpr, tpr, label=f"AUC = {self.last_eval['roc_auc']:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("Tasa de Falsos Positivos")
            ax.set_ylabel("Tasa de Verdaderos Positivos")
            ax.set_title("Curva ROC")
            ax.legend()
        else:
            # Si no hay probabilidades, mostrar texto informativo ( obvio que va a haber probabilidades)
            ax.text(0.5, 0.5, "Probabilidades no disponibles", ha="center", va="center")
            ax.set_axis_off()

        # Curva Precision-Recall 
        ax = axs[1, 0]
        if y_proba is not None:
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            ax.plot(rec, prec, label=f"AP={self.last_eval['avg_precision']:.3f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Curva Precision-Recall")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Probabilidades no disponibles", ha="center", va="center")
            ax.set_axis_off()

        # Barras con métricas principales
        ax = axs[1, 1]
        metrics = {
            "accuracy": self.last_eval["acc"],
            "precision": self.last_eval["precision"],
            "recall": self.last_eval["recall"],
            "f1": self.last_eval["f1"],
        }
        names = list(metrics.keys())
        vals = [metrics[n] for n in names]
        ax.bar(names, vals, color=["#2b83ba", "#abdda4", "#fdae61", "#d7191c"])
        ax.set_ylim(0, 1)
        ax.set_title("Métricas principales")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center")  # ponerle valores encima de las barras

        # Ajustar layout y dibujar en el canvas de la UI
        self.fig.tight_layout()
        self.canvas.draw()

        # Mostrar clasificación en la caja de texto
        report = classification_report(y_test, y_pred, zero_division=0)
        self.df_info_text.delete("0.0", "end")
        header = f"Reporte de clasificación (test)\nAccuracy: {self.last_eval['acc']:.3f}  F1: {self.last_eval['f1']:.3f}\n\n"
        self.df_info_text.insert("0.0", header + report)

        # (Ignorar nuevamente, es lo mismo que se implemento de sillhouette)
        sil_info = self.last_eval.get("silhouette", None)
        if sil_info:
            sil_text = f"\nSilhouette score (en features numéricas): {sil_info.get('score')}  mejor k: {sil_info.get('best_k')}"
            self.df_info_text.insert("end", sil_text)

    # (Ignorar tambien, es no implementado por valores normalizados)
    def predecir_desde_inputs(self):
        # Verificar que exista un pipeline cargado
        if self.pipeline is None:
            messagebox.showwarning("Aviso", "Necesita cargar o entrenar un modelo primero.")
            return

        # Construir un diccionario con los valores ingresados por el usuario
        row = {}
        for feat, widget in self.input_entries.items():
            if feat in self.categorical_features:
                # Para categóricas, obtener el valor seleccionado
                val = widget.get()
                row[feat] = val
            else:
                # Para numéricas, intentar convertir a float; si falla, usar NaN
                txt = widget.get()
                try:
                    val = float(txt)
                except Exception:
                    val = np.nan
                row[feat] = val

        # Crear DataFrame con una sola fila y asegurar el orden de columnas
        X_new = pd.DataFrame([row])[self.feature_names]

        try:
            # Obtener probabilidad y predicción (si el pipeline lo soporta)
            proba = self.pipeline.predict_proba(X_new)[:, 1][0] if hasattr(self.pipeline, "predict_proba") else None
            pred = self.pipeline.predict(X_new)[0]
            # Actualizar etiquetas de resultado y probabilidad en la UI
            self.result_label.configure(text=f"Resultado: {'Desertará' if pred in (1, True, '1') else 'No desertará'}")
            if proba is not None:
                self.prob_label.configure(text=f"Probabilidad de desertar: {proba:.3f}")
            else:
                self.prob_label.configure(text="Probabilidad no disponible")
        except Exception as e:
            # Si ocurre error durante la predicción, mostrar mensaje
            messagebox.showerror("Error en predicción", str(e))

    # Guardar modelo en archivos 
    def guardar_modelo(self):
        # Verificar que exista pipeline para guardar
        if self.pipeline is None:
            messagebox.showwarning("Aviso", "No hay modelo para guardar.")
            return
        try:
            # Serializar pipeline completo (incluye preprocesado) con joblib
            joblib.dump(self.pipeline, MODEL_FILENAME)
            self.estado("Modelo guardado.")
            messagebox.showinfo("Guardado", f"Modelo guardado en {MODEL_FILENAME}")
        except Exception as e:
            # Manejo de errores al guardar
            messagebox.showerror("Error al guardar", str(e))

    #  Cargar modelo desde archivo seleccionado por el usuario
    def cargar_modelo_archivo(self):
        ruta = filedialog.askopenfilename(filetypes=[("Joblib", "*.joblib *.pkl *.joblib"), ("Todos", "*.*")])
        if not ruta:
            return
        try:
            self.cargar_modelo(ruta)  # delegar en el método cargar_modelo
            messagebox.showinfo("Cargado", "Modelo cargado desde archivo.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar: {e}")

    # Cargar modelo desde ruta dada 
    def cargar_modelo(self, ruta):
        mdl = joblib.load(ruta)  # cargar objeto serializado
        if not hasattr(mdl, "predict"):
            # Validación mínima: el objeto debe tener método predict
            raise ValueError("El archivo no parece contener un modelo válido.")
        self.pipeline = mdl  # asignar pipeline cargado
        self.estado("Modelo cargado")
        # Si ya hay un DataFrame cargado, reconstruir el formulario para mantener consistencia
        if self.df is not None:
            self.construir_formulario_prediccion()

# Bloque principal: crear y ejecutar la aplicación 
if __name__ == "__main__":
    app = AplicacionHR()  # crear instancia de la aplicación
    app.mainloop()  # iniciar el loop principal de la GUI