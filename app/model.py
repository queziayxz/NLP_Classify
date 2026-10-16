import joblib
from pathlib import Path
import pandas as pd

from .preprocess import PreProcess
from .features import Features

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO = RAIZ / "model" / "model.joblib"

preprocess = PreProcess()
features = Features()

class Model():
    def __init__(self):
        pass

    def train():
        pass

    def load_model(self) -> dict:
        art = joblib.load(CAMINHO)
        return art

    def predict(self, text: str):
        model = self.load_model()
        fragments = preprocess.split_phrase(text=text, n_word=150)
        rows = []
        for frag in fragments:
            rows.append(features.create(fragment=frag))
        df = pd.DataFrame(rows)
        df["text"] = text
        predict = model.predict(df)

        return {
            "metricas": {
                "n_canonicas": df["n_canonicas"].iat[0],
                "n_canonicas_type": df["n_canonicas_type"].iat[0],
                "size_mean_words": df["size_mean_words"].iat[0],
                "size_mean_phrase": df["size_mean_words"].iat[0],
                "virgulas_por_frase": df["virgulas_por_frase"].iat[0],
                "diversity": df["diversity"].iat[0],
                "flesch_reading_ease": df["flesch_reading_ease"].iat[0],
                "syllables_per_word": df["syllables_per_word"].iat[0],
                "prop_sconj": df["prop_sconj"].iat[0],
                "prop_cconj": df["prop_cconj"].iat[0],
                "prop_adj": df["prop_adj"].iat[0],
                "prop_adv": df["prop_adv"].iat[0],
                "prop_verb": df["prop_verb"].iat[0],
                "prop_adp": df["prop_adp"].iat[0],
            },
            "level": predict[0]
        }