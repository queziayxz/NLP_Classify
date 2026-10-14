import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from load import Load
from app.preprocess import PreProcess
from app.features import Features
from comparar_modelos import montar_T, comparar_todos, conjuntos_de_entrada, avaliar_no_teste

n_words = 150

preprocess = PreProcess()
features = Features()
docs = preprocess.loadDataFrame()
x_train, x_test, x_val = preprocess.split_docs(docs=docs)

x_train = preprocess.create_phrases(x_train, n_words=n_words)
x_test = preprocess.create_phrases(x_test, n_words=n_words)
x_val = preprocess.create_phrases(x_val, n_words=n_words)

F_train = features.create_features(x_train)
F_test = features.create_features(x_test)
F_val = features.create_features(x_val)

F_train["text"] = x_train["text"].values
F_test["text"] = x_test["text"].values
F_val["text"] = x_val["text"].values

T = montar_T(F_train, x_train, F_val, x_val)

cols_all = [col for col in F_train.columns if col != "text"]
cols_canon = ["n_canonicas", "n_caninicas_type"]
# tabela = comparar_todos(T, cols_all, cols_canon, n_jobs=1)
# print(tabela.round(3).to_string())

entradas = conjuntos_de_entrada(cols_all, cols_canon)

finalistas = [
    ("Baseline (classe mais frequente)", "features"),
    ("Random Forest",                    "features"),
    ("Regressão Logística",              "tfidf+features"),
    ("SVM linear",                       "tfidf+features"),
    ("LightGBM",                         "tfidf+features"),
    ("Regressão Logística",              "tfidf+features sem canonicidade"),
]

model = avaliar_no_teste(nome_modelo="Regressão Logística", usar_tfidf=True, cols=cols_all, T_treino=T)
path_model = "model.joblib"
joblib.dump(model, path_model)
print(f"Modelo salvo com sucesso em: {path_model}")
# resultados = []
# for nome, entrada in finalistas:
#     usar_tfidf, cols = entradas[entrada]
#     print(f"nome: {nome}")
#     print(f"entrada: {entrada}")
#     print(f"usar_tfidf: {usar_tfidf}")
#     print(f"cols: {cols}")
#     res, _ = avaliar_no_teste(nome, usar_tfidf, cols, T_treino=T, F_test=F_test,
#                               y_test=x_test["classe"], rotulo=f"{nome} | {entrada}")
#     resultados.append(res)

# teste = pd.DataFrame(resultados).sort_values("macro_F1", ascending=False)
# teste.to_csv("resultados_teste.csv", index=False)
# print(teste.round(3).to_string())

