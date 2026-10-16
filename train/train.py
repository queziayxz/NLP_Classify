import os
import sys

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.dirname(diretorio_atual)
sys.path.append(diretorio_pai)
from app.preprocess import PreProcess
from app.features import Features

import joblib
import pandas as pd
from pathlib import Path
from comparar_modelos import montar_T, conjuntos_de_entrada, avaliar_no_teste, comparar_todos

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO = RAIZ / "model" / "model.joblib"

n_words = 150

preprocess = PreProcess()
features = Features()
docs = preprocess.loadDataFrame()

print("Dividindo arquivos em conjuntos de Treino, Validação e Teste")
x_train, x_test, x_val = preprocess.split_docs(docs=docs)

print("Dividindo conteúdo dos arquivos em textos menores")
x_train = preprocess.create_phrases(x_train, n_words=n_words)
x_test = preprocess.create_phrases(x_test, n_words=n_words)
x_val = preprocess.create_phrases(x_val, n_words=n_words)

print("Criando conjuntos com as features avaliadas no treinamento")
F_train = features.create_features(x_train)
F_test = features.create_features(x_test)
F_val = features.create_features(x_val)

F_train["text"] = x_train["text"].values
F_test["text"] = x_test["text"].values
F_val["text"] = x_val["text"].values

T = montar_T(F_train, x_train, F_val, x_val)

cols_all = [col for col in F_train.columns if col != "text"]
cols_canon = ["n_canonicas", "n_caninicas_type"]

entradas = conjuntos_de_entrada(cols_all, cols_canon)

finalistas = [
    ("Baseline (classe mais frequente)", "features"),
    ("Random Forest",                    "features"),
    ("Regressão Logística",              "features"),
    ("Regressão Logística",              "tfidf"),
    ("Regressão Logística",              "tfidf+features"),
    ("SVM linear",                       "tfidf+features"),
    ("SVM linear",                       "tfidf"),
    ("LightGBM",                         "tfidf+features"),
    ("Regressão Logística",              "tfidf+features sem canonicidade"),
]

# nome_modelo, usar_tfidf, cols, T_treino, F_test=None, y_test=None, rotulo=None, salvar_png=True
print("Treinando Modelo")
# T, cols_all, cols_canon, n_splits=5, salvar_csv="resultados_cv.csv", n_jobs=1
# tabela = comparar_todos(T, cols_all, cols_canon)
# print(tabela.round(3).to_string())
resultados = []
for nome, entrada in finalistas:
    usar_tfidf, cols = entradas[entrada]
    res, model = avaliar_no_teste(nome, usar_tfidf, cols, T_treino=T, F_test=F_test,
                              y_test=x_test["classe"], rotulo=f"{nome} | {entrada}")
    resultados.append(res)
# model = avaliar_no_teste("Regressão Logística", False, cols_all, T_treino=T, F_test=F_test, 
#                               y_test=x_test["classe"], rotulo=f"{"features"}")
# model = avaliar_no_teste("Regressão Logística", True, [], T_treino=T, F_test=F_test, 
#                               y_test=x_test["classe"], rotulo=f"{"tfidf"}")
# model = avaliar_no_teste("Regressão Logística", True, cols_all, T_treino=T, F_test=F_test, 
                            #   y_test=x_test["classe"], rotulo=f"{"tfidf+features"}")
# res, model = avaliar_no_teste(nome_modelo="Regressão Logística", usar_tfidf=True, cols=cols_all, T_treino=T)
teste = pd.DataFrame(resultados).sort_values("macro_F1", ascending=False)
teste.to_csv(RAIZ / "resultados/resultados_teste.csv", index=False)
print(teste.round(3).to_string())
# path_model = RAIZ / "model/model.joblib"
# joblib.dump(model, path_model)
# print(f"Modelo salvo com sucesso em: {path_model}")

