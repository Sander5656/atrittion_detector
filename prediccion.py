import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             average_precision_score, classification_report,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

matplotlib.use("TkAgg")
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

MODEL_FILENAME = "mejor_modelo_logreg.joblib"
PROCESS_THREAD_LOCK = threading.Lock()


class AplicacionHR(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Deserción de Empleados - Regresión Logística")
        self.geometry("1100x700")

        # Estado interno
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = []
        self.pipeline = None
        self.best_estimator_ = None
        self.categorical_features = []
        self.numerical_features = []

        # Barra superior con controles
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(side="top", fill="x", padx=12, pady=8)

        self.load_btn = ctk.CTkButton(top_frame, text="Cargar CSV", command=self.cargar_csv)
        self.load_btn.pack(side="left", padx=(0, 8))

        self.train_btn = ctk.CTkButton(top_frame, text="Entrenar modelo", command=self.iniciar_entrenamiento_thread)
        self.train_btn.pack(side="left", padx=(0, 8))

        self.save_btn = ctk.CTkButton(top_frame, text="Guardar modelo", command=self.guardar_modelo)
        self.save_btn.pack(side="left", padx=(0, 8))

        self.load_model_btn = ctk.CTkButton(top_frame, text="Cargar modelo (archivo)", command=self.cargar_modelo_archivo)
        self.load_model_btn.pack(side="left", padx=(0, 8))

        self.status_label = ctk.CTkLabel(top_frame, text="Estado: listo")
        self.status_label.pack(side="right")

        # Vista principal con pestañas
        self.tab_view = ctk.CTkTabview(self, width=1000)
        self.tab_view.pack(expand=True, fill="both", padx=12, pady=12)

        self.tab_view.add("Explorar y Gráficas")
        self.tab_view.add("Predecir")

        # Pestaña Explorar
        self.explore_frame = self.tab_view.tab("Explorar y Gráficas")
        left_explore = ctk.CTkFrame(self.explore_frame, width=350)
        left_explore.pack(side="left", fill="y", padx=8, pady=8)

        self.df_info_text = ctk.CTkTextbox(left_explore, width=330, height=300)
        self.df_info_text.pack(padx=6, pady=6)

        self.plot_btn = ctk.CTkButton(left_explore, text="Generar gráficas (requiere modelo entrenado)", command=self.generar_graficas)
        self.plot_btn.pack(padx=6, pady=(12, 6))

        right_explore = ctk.CTkFrame(self.explore_frame)
        right_explore.pack(side="right", expand=True, fill="both", padx=8, pady=8)

        self.fig = plt.Figure(figsize=(7, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_explore)
        self.canvas.get_tk_widget().pack(expand=True, fill="both")

        # Pestaña Predecir
        self.predict_frame = self.tab_view.tab("Predecir")
        top_predict = ctk.CTkFrame(self.predict_frame)
        top_predict.pack(side="top", fill="x", padx=12, pady=12)

        form_frame = ctk.CTkFrame(self.predict_frame)
        form_frame.pack(side="left", fill="y", padx=12, pady=12)

        self.input_entries = {}

        # Panel derecho para resultados
        result_frame = ctk.CTkFrame(self.predict_frame)
        result_frame.pack(side="right", expand=True, fill="both", padx=12, pady=12)

        self.result_label = ctk.CTkLabel(result_frame, text="Resultado: modelo no cargado", font=ctk.CTkFont(size=18, weight="bold"))
        self.result_label.pack(pady=12)

        self.prob_label = ctk.CTkLabel(result_frame, text="Probabilidad: N/A")
        self.prob_label.pack(pady=6)

        self.predict_btn = ctk.CTkButton(result_frame, text="Predecir", command=self.predecir_desde_inputs)
        self.predict_btn.pack(pady=12)

        # Cargar modelo si existe
        if os.path.exists(MODEL_FILENAME):
            try:
                self.cargar_modelo(MODEL_FILENAME)
                self.estado("Modelo cargado automáticamente desde archivo.")
            except Exception:
                self.estado("Modelo guardado pero no pudo cargarse automáticamente.")

    # Utilidades de UI
    def estado(self, texto):
        self.status_label.configure(text=f"Estado: {texto}")

    def cargar_csv(self):
        ruta = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")])
        if not ruta:
            return
        try:
            self.df = pd.read_csv(ruta)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el CSV: {e}")
            return

        # Mostrar info básica
        self.df_info_text.delete("0.0", "end")
        info = []
        info.append(f"Archivo: {os.path.basename(ruta)}")
        info.append(f"Filas: {self.df.shape[0]}  Columnas: {self.df.shape[1]}")
        info.append("\nColumnas y tipos:")
        for c, t in zip(self.df.columns, self.df.dtypes):
            info.append(f"- {c}: {t}")
        info_text = "\n".join(info)
        self.df_info_text.insert("0.0", info_text)

        # Inferir columna objetivo por nombres comunes
        posibles = [c for c in self.df.columns if c.lower() in ("left", "left_company", "attrition", "desertion", "desert", "is_left", "target", "label", "deserto", "desercion", "abandono")]
        if not posibles:
            objetivo = self.df.columns[-1]
        else:
            objetivo = posibles[0]

        self.estado(f"CSV cargado. Usando objetivo: {objetivo}")

        self.y = self.df[objetivo]
        self.X = self.df.drop(columns=[objetivo])
        self.categorical_features = list(self.X.select_dtypes(include=["object", "category"]).columns)
        self.numerical_features = list(self.X.select_dtypes(include=[np.number]).columns)
        self.feature_names = self.numerical_features + self.categorical_features

        self.construir_formulario_prediccion()
        self.estado("CSV listo. Puedes entrenar el modelo.")

    def construir_formulario_prediccion(self):
        # Encontrar frame de formulario izquierdo y limpiarlo
        form_frame = None
        for child in self.predict_frame.winfo_children():
            if isinstance(child, ctk.CTkFrame):
                # heurística sencilla para escoger el frame izquierdo
                form_frame = child
                break
        if form_frame is None:
            form_frame = ctk.CTkFrame(self.predict_frame)
            form_frame.pack(side="left", fill="y", padx=12, pady=12)

        for w in form_frame.winfo_children():
            w.destroy()

        self.input_entries = {}
        lbl = ctk.CTkLabel(form_frame, text="Ingrese datos para predecir", font=ctk.CTkFont(size=16, weight="bold"))
        lbl.pack(pady=(4, 8))

        for feat in self.feature_names:
            frame = ctk.CTkFrame(form_frame)
            frame.pack(fill="x", pady=4, padx=6)
            label = ctk.CTkLabel(frame, text=f"{feat}:", width=160, anchor="w")
            label.pack(side="left", padx=4)

            if feat in self.categorical_features:
                unique_vals = list(self.X[feat].dropna().unique().astype(str))
                if not unique_vals:
                    unique_vals = [""]
                opt = ctk.CTkOptionMenu(frame, values=unique_vals)
                opt.set(unique_vals[0])
                opt.pack(side="right", padx=4)
                self.input_entries[feat] = opt
            else:
                entry = ctk.CTkEntry(frame)
                try:
                    meanv = float(self.X[feat].dropna().astype(float).mean())
                    entry.insert(0, f"{meanv:.3f}")
                except Exception:
                    pass
                entry.pack(side="right", fill="x", expand=True, padx=4)
                self.input_entries[feat] = entry

    # Entrenamiento (thread seguro)
    def iniciar_entrenamiento_thread(self):
        if self.df is None:
            messagebox.showwarning("Aviso", "Primero cargue un archivo CSV.")
            return
        if PROCESS_THREAD_LOCK.locked():
            messagebox.showinfo("Información", "Ya hay un entrenamiento en proceso.")
            return
        thread = threading.Thread(target=self.entrenar_modelo, daemon=True)
        thread.start()

    def entrenar_modelo(self):
        with PROCESS_THREAD_LOCK:
            try:
                self.estado("Entrenando...")
                X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

                numeric_transformer = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                )
                categorical_transformer = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False)
),
                    ]
                )
                preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer, self.numerical_features),
                        ("cat", categorical_transformer, self.categorical_features),
                    ],
                    remainder="drop",
                )

                logreg = LogisticRegression(max_iter=1000, solver="liblinear")
                pipe = Pipeline(steps=[("pre", preprocessor), ("clf", logreg)])

                param_grid = {
                    "clf__C": [0.01, 0.1, 1, 10],
                    "clf__penalty": ["l1", "l2"],
                }
                grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0)
                grid.fit(X_train, y_train)

                self.pipeline = grid.best_estimator_
                self.best_estimator_ = grid.best_estimator_

                y_pred = self.pipeline.predict(X_test)
                y_proba = self.pipeline.predict_proba(X_test)[:, 1] if hasattr(self.pipeline, "predict_proba") else None

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
                avg_prec = average_precision_score(y_test, y_proba) if y_proba is not None else None

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

                # Cálculo de silhouette en features numéricas si es posible
               

                self.estado(f"Entrenado. Acc={acc:.3f} F1={f1:.3f}")
                messagebox.showinfo("Entrenamiento completado", f"Mejor modelo: {grid.best_params_}\nAccuracy: {acc:.3f}\nF1: {f1:.3f}")
            except Exception as e:
                messagebox.showerror("Error de entrenamiento", str(e))
                self.estado("Error en entrenamiento")

    def generar_graficas(self):
        if not hasattr(self, "last_eval"):
            messagebox.showwarning("Aviso", "Entrene el modelo primero.")
            return

        self.fig.clf()
        axs = self.fig.subplots(2, 2)
        X_test = self.last_eval["X_test"]
        y_test = self.last_eval["y_test"]
        y_pred = self.last_eval["y_pred"]
        y_proba = self.last_eval["y_proba"]

        # Matriz de confusión
        ax = axs[0, 0]
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap=plt.cm.Blues, display_labels=["No", "Sí"])
        ax.set_title("Matriz de confusión")

        # ROC
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
            ax.text(0.5, 0.5, "Probabilidades no disponibles", ha="center", va="center")
            ax.set_axis_off()

        # Precision-Recall
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

        # Barras de métricas
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
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

        self.fig.tight_layout()
        self.canvas.draw()

        reporte = classification_report(y_test, y_pred, zero_division=0)
        self.df_info_text.delete("0.0", "end")
        header = f"Reporte de clasificación (test)\nAccuracy: {self.last_eval['acc']:.3f}  F1: {self.last_eval['f1']:.3f}\n\n"
        self.df_info_text.insert("0.0", header + reporte)

        sil_info = self.last_eval.get("silhouette", None)
        if sil_info:
            sil_text = f"\nSilhouette score (en features numéricas): {sil_info.get('score')}  mejor k: {sil_info.get('best_k')}"
            self.df_info_text.insert("end", sil_text)

    def predecir_desde_inputs(self):
        if self.pipeline is None:
            messagebox.showwarning("Aviso", "Necesita cargar o entrenar un modelo primero.")
            return
        row = {}
        for feat, widget in self.input_entries.items():
            if feat in self.categorical_features:
                val = widget.get()
                row[feat] = val
            else:
                txt = widget.get()
                try:
                    val = float(txt)
                except Exception:
                    val = np.nan
                row[feat] = val
        X_new = pd.DataFrame([row])[self.feature_names]

        try:
            proba = self.pipeline.predict_proba(X_new)[:, 1][0] if hasattr(self.pipeline, "predict_proba") else None
            pred = self.pipeline.predict(X_new)[0]
            self.result_label.configure(text=f"Resultado: {'Desertará' if pred in (1, True, '1') else 'No desertará'}")
            if proba is not None:
                self.prob_label.configure(text=f"Probabilidad de desertar: {proba:.3f}")
            else:
                self.prob_label.configure(text="Probabilidad no disponible")
        except Exception as e:
            messagebox.showerror("Error en predicción", str(e))

    def guardar_modelo(self):
        if self.pipeline is None:
            messagebox.showwarning("Aviso", "No hay modelo para guardar.")
            return
        try:
            joblib.dump(self.pipeline, MODEL_FILENAME)
            self.estado("Modelo guardado.")
            messagebox.showinfo("Guardado", f"Modelo guardado en {MODEL_FILENAME}")
        except Exception as e:
            messagebox.showerror("Error al guardar", str(e))

    def cargar_modelo_archivo(self):
        ruta = filedialog.askopenfilename(filetypes=[("Joblib", "*.joblib *.pkl *.joblib"), ("Todos", "*.*")])
        if not ruta:
            return
        try:
            self.cargar_modelo(ruta)
            messagebox.showinfo("Cargado", "Modelo cargado desde archivo.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar: {e}")

    def cargar_modelo(self, ruta):
        mdl = joblib.load(ruta)
        if not hasattr(mdl, "predict"):
            raise ValueError("El archivo no parece contener un modelo válido.")
        self.pipeline = mdl
        self.estado("Modelo cargado")
        if self.df is not None:
            self.construir_formulario_prediccion()


if __name__ == "__main__":
    app = AplicacionHR()
    app.mainloop()