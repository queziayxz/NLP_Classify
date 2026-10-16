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
from comparar_modelos import montar_T, avaliar_no_teste

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
F_train = features.create_features_train(x_train)
F_test = features.create_features_train(x_test)
F_val = features.create_features_train(x_val)

F_train["text"] = x_train["text"].values
F_test["text"] = x_test["text"].values
F_val["text"] = x_val["text"].values

T = montar_T(F_train, x_train, F_val, x_val)

cols_all = [col for col in F_train.columns if col != "text"]
cols_canon = ["n_canonicas", "n_caninicas_type"]

print("Treinando Modelo")
res, model = avaliar_no_teste(nome_modelo="Regressão Logística", usar_tfidf=True, cols=cols_all, 
                              F_test=F_test, y_test=x_test["classe"], T_treino=T)
path_model = RAIZ / "model/model.joblib"
joblib.dump(model, path_model)
print(f"Modelo salvo com sucesso em: {path_model}")

