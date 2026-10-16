import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, make_scorer, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO = RAIZ / "model" / "model.joblib"

try:
    from lightgbm import LGBMClassifier
except ImportError:  # segue sem LightGBM
    LGBMClassifier = None

SEED = 42
CLASSES = ["Fundamental 1", "Fundamental 2", "Médio", "Superior"]
ORDEM = {c: i for i, c in enumerate(CLASSES)}


# ----------------------------------------------------------------------
# Metricas ordinais (as classes têm uma ordem)
# ----------------------------------------------------------------------
def _idx(y):
    return pd.Series(np.asarray(y)).map(ORDEM).values


def mae_ordinal(y_true, y_pred):
    return float(np.abs(_idx(y_true) - _idx(y_pred)).mean())


def acc_mais_menos_1(y_true, y_pred):
    return float((np.abs(_idx(y_true) - _idx(y_pred)) <= 1).mean())


SCORING = {
    "f1_macro": "f1_macro",
    "prec": make_scorer(precision_score, average="macro", zero_division=0),   # nova
    "rec": make_scorer(recall_score, average="macro", zero_division=0),       # nova
    "acc": "accuracy",
    "acc_pm1": make_scorer(acc_mais_menos_1),
    "mae": make_scorer(mae_ordinal, greater_is_better=False),
}

# ----------------------------------------------------------------------
# Modelos candidatos
# ----------------------------------------------------------------------
MODELOS = {
    "Baseline (classe mais frequente)": lambda: DummyClassifier(strategy="most_frequent"),
    "Regressão Logística": lambda: LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0),
    "SVM linear": lambda: LinearSVC(class_weight="balanced", C=0.5, max_iter=5000),
    "Naive Bayes (Complement)": lambda: ComplementNB(),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
}
if LGBMClassifier is not None:
    MODELOS["LightGBM"] = lambda: LGBMClassifier(
        n_estimators=300, learning_rate=0.05, class_weight="balanced",
        random_state=SEED, verbose=-1)

# modelos lineares sofrem com a diferença de escala entre TF-IDF e features -> usam peso
LINEARES = {"Regressão Logística", "SVM linear"}

# em quais conjuntos de entrada cada modelo faz sentido
APENAS = {
    "Baseline (classe mais frequente)": {"features"},
    "Naive Bayes (Complement)": {"tfidf"},  # exige valores não negativos (sem features padronizadas)
}


def conjuntos_de_entrada(cols_all, cols_canon):
    """nome -> (usa_tfidf, colunas numericas). Inclui a ablação da canonicidade."""
    cols_sem = [c for c in cols_all if c not in cols_canon]
    return {
        "features": (False, cols_all),
        "features sem canonicidade": (False, cols_sem),
        "tfidf": (True, []),
        "tfidf+features": (True, cols_all),
        "tfidf+features sem canonicidade": (True, cols_sem),
    }


def construir(nome_modelo, usar_tfidf, cols, w_feats=0.05, max_features=2500):
    """Pipeline: [TF-IDF do texto | features padronizadas] -> classificador."""
    blocos, pesos = [], {}
    if usar_tfidf:
        blocos.append(("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                                min_df=5, sublinear_tf=True), "text"))
        pesos["tfidf"] = 1.0
    if cols:
        blocos.append(("num", StandardScaler(), list(cols)))
        # o peso só importa para modelos lineares combinando os dois blocos
        pesos["num"] = w_feats if (nome_modelo in LINEARES and usar_tfidf) else 1.0
    return make_pipeline(ColumnTransformer(blocos, transformer_weights=pesos), MODELOS[nome_modelo]())


def montar_T(F_a, x_a, F_b, x_b):
    """Junta treino e validação numa tabela só (colunas: text, features, doc_id, classe)."""
    partes = []
    for F, x in [(F_a, x_a), (F_b, x_b)]:
        partes.append(F.assign(doc_id=x["doc_id"].values, classe=x["classe"].values))
    return pd.concat(partes, ignore_index=True)


# ----------------------------------------------------------------------
# 1) Comparação por validação cruzada agrupada (sem tocar no teste)
# ----------------------------------------------------------------------
def comparar_todos(T, cols_all, cols_canon, n_splits=5, salvar_csv="resultados_cv.csv", n_jobs=1):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)  # mesmas dobras p/ todos
    linhas = []
    for conj_nome, (usar_tfidf, cols) in conjuntos_de_entrada(cols_all, cols_canon).items():
        for nome in MODELOS:
            if nome in APENAS and conj_nome not in APENAS[nome]:
                continue
            t0 = time.time()
            r = cross_validate(construir(nome, usar_tfidf, cols), T, T["classe"], cv=cv,
                               groups=T["doc_id"], scoring=SCORING, n_jobs=n_jobs)
            linhas.append({
                "modelo": nome, "entrada": conj_nome,
                "macro_F1": r["test_f1_macro"].mean(), "macro_F1_dp": r["test_f1_macro"].std(),
                "precisao_macro": r["test_prec"].mean(), "precisao_macro_dp": r["test_prec"].std(),
                "recall_macro": r["test_rec"].mean(), "recall_macro_dp": r["test_rec"].std(),
                "acuracia": r["test_acc"].mean(), "acuracia_dp": r["test_acc"].std(),
                "acerto_±1_nivel": r["test_acc_pm1"].mean(),
                "MAE_niveis": abs(r["test_mae"].mean()),
                "seg": round(time.time() - t0, 1),
            })
            u = linhas[-1]
            print(f"{nome:34s} {conj_nome:34s} macro-F1={u['macro_F1']:.3f} ± {u['macro_F1_dp']:.3f}  ({u['seg']}s)")
    tabela = pd.DataFrame(linhas).sort_values("macro_F1", ascending=False).reset_index(drop=True)
    if salvar_csv:
        tabela.to_csv(salvar_csv, index=False)
    return tabela


# ----------------------------------------------------------------------
# 2) Avaliação final no teste (usar UMA vez, com modelos já escolhidos)
# ----------------------------------------------------------------------
def avaliar_no_teste(nome_modelo, usar_tfidf, cols, T_treino, F_test, y_test=None, rotulo=None, salvar_png=True):
    modelo = construir(nome_modelo, usar_tfidf, cols)
    modelo.fit(T_treino, T_treino["classe"])
    y_pred = modelo.predict(F_test)

    rotulo = rotulo or f"{nome_modelo} | {'TF-IDF' if usar_tfidf else ''}{'+' if usar_tfidf and cols else ''}{'features' if cols else ''}"
    print("=" * 70)
    print(rotulo)
    print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0))
    res = {
        "modelo": rotulo,
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "macro_F1": f1_score(y_test, y_pred, average="macro"),
        "acuracia": accuracy_score(y_test, y_pred),
        "acerto_±1_nivel": acc_mais_menos_1(y_test, y_pred),
        "MAE_niveis": mae_ordinal(y_test, y_pred),
    }
    print({k: round(v, 3) for k, v in res.items() if k != "modelo"})

    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Matriz de confusão (teste) - {rotulo}", fontsize=10)
    plt.ylabel("Classe real")
    plt.xlabel("Classe prevista")
    plt.tight_layout()
    # if salvar_png:
    #     arq = "cm_" + "".join(ch if ch.isalnum() else "_" for ch in rotulo)[:60] + ".png"
    #     plt.savefig(RAIZ / "resultados" / arq, dpi=150)
    # plt.show()
    return res, modelo