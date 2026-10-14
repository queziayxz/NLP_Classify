import os
import joblib
from pathlib import Path

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
        df = features.create(fragments=fragments)
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

# model = Model()
# # model.predic("O Pedro tem um cachorro. O nome do cachorro é Bolinha. Ele é pequeno e muito brincalhão. Todo dia, o Pedro leva o Bolinha para passear na praça. Lá, o cachorro corre atrás da bola. Depois, os dois bebem água na sombra de uma árvore. A mãe do Pedro diz que o Bolinha é da família. À noite, o cachorro dorme perto da cama do menino. Quando o Pedro acorda, o Bolinha já está de pé, esperando mais um dia de brincadeira. O Pedro sorri e dá um abraço no amigo. Os dois são muito felizes juntos.")
# result = model.predic("A heterogeneidade dos resultados observados nos diferentes estratos socioeconômicos sugere que a eficácia das políticas educacionais não pode ser dissociada das condições estruturais em que são implementadas. Embora a literatura aponte correlações significativas entre investimento público e desempenho escolar, tais associações dependem, em grande medida, de mediações institucionais que escapam à mensuração convencional. Nesse sentido, a análise comparativa exige cautela metodológica, uma vez que a inferência causal pressupõe a identificação de variáveis de confusão frequentemente negligenciadas. Ademais, a ausência de dados longitudinais compromete a possibilidade de distinguir efeitos transitórios de transformações duradouras.")
# print(result)