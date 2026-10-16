import nltk
import pandas as pd
import os
import re
import spacy
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
# nltk.download("stopwords")
# nltk.download("punkt") 
# nltk.download("punkt_tab")

FUND_1 = "1_Ensino_Fundamental_I"
FUND_2 = "2_Ensino_Fundamental_II"
MEDIO = "3_Ensino_Medio"
SUPERIOR = "4_Ensino_Superior"

DIR_DATA = [FUND_1,FUND_2,MEDIO,SUPERIOR]

DATASET = "./dataset_caed"

STOP = set(stopwords.words("portuguese"))
nlp = spacy.load("pt_core_news_sm")

class Load():

    def __init__(self):
        pass
  
    def loadDataFrame(self):
        """
            Funcao responsavel por ler os arquivos de dados e criar um dataframe com colunas 
            "doc_id", "file_name", "class" e "text"
            Return: docs [DataFrame]
        """
        files = []
        mapa = {f"{FUND_1}": "Fundamental 1", f"{FUND_2}": "Fundamental 2", f"{MEDIO}": "Médio", f"{SUPERIOR}": "Superior"}
        for dir in DIR_DATA:
            path_dir = f"{DATASET}/{dir}"
            for file_name in os.listdir(path_dir):
                path = os.path.join(path_dir, file_name)
                if os.path.isfile(path):
                    files.append({
                        "doc_id": path,
                        "file_name": file_name,
                        "classe": dir,
                        "text": self.read(path)
                    })

        docs = pd.DataFrame(files)
        docs["classe"] = docs["classe"].map(mapa)

        return docs

    def split_docs(self, docs):
        """
            Funcao responsavel por separar o dataframe em blocos de treino, teste e validacao
            Args: 
                docs [DataFrame] -> "doc_id", "file_name", "class", "text"
            Return: 
                x_train, x_test, x_val [DataFrame]
            """
        docs_trval, x_test = train_test_split(
            docs, test_size=0.15, stratify=docs["classe"], random_state=42
        )
        x_train, x_val = train_test_split(
            docs_trval, test_size=0.15 / 0.85, stratify=docs_trval["classe"], random_state=42
        )

        print(type(x_train))

        assert not set(x_train["doc_id"]) & set(x_test["doc_id"])
        assert not set(x_train["doc_id"]) & set(x_val["doc_id"])
        assert not set(x_val["doc_id"]) & set(x_test["doc_id"])

        return x_train, x_test, x_val

    def split_phrase(self, text: str, n_word: int) -> list:
        """
            Funcao responsavel por dividir as frases em tamanhos menores para termos mais dados

            Args: 
                text: str -> texto a ser dividido
                n_words: int -> numero maximo de palavras que uma frase tera
            
            Return: 
                fragment: list -> lista contendo todas as novas frases separadas em menor quantidade
        """
        phrases = sent_tokenize(text, language="portuguese")
        cont , current, fragment = 0, [], []
        for phrase in phrases:
            current.append(phrase)
            words = phrase.split()
            cont += len(words)
            if(cont >= n_word):
                fragment.append(" ".join(current))
                current, cont = [], 0
        if current and (cont >= n_word // 2 or not fragment):  # resto muito curto eh descartado
            fragment.append(" ".join(current))

        return fragment

    def create_features(self, docs):
        """
            Funcao que cria features para serem analisadas pelo modelo no seu treinamento
            Features:
                - n_canonicas: quantidade media de palavras canonicas na frase
                - n_caninicas_type: quantidade media de palavras canonicas por tipo
                - size_mean_words: tamanho medio das palavras
                - size_mean_phrase: tamanho medio das frases
                - virgulas_por_frase: quantidade de virgulas por frase,
                - razao_type_token: razao entre palavras distintas e o total de palavras. mede a variedade de palavras na frase
            Args:
                docs [DataFrame] -> conjunto de treino, teste ou validacao com o texto
            Return:
                df [DataFrame] -> conjunto novo com as features
        """
        rows = []
        for _, row in docs.iterrows():
            phrase = sent_tokenize(row["text"], language="portuguese")
            
            # separa as frases em palavras e pontuacao
            # transforma tudo para minusculo descartando numeros e pontuacao com o isalpha. retira tambem palavras com hifen (guarda-chuva)
            words = [t for t in word_tokenize(row["text"], language="portuguese") if t.isalpha()]
            # filter_words = [p for p in words if p not in STOP]
            n_words = max(len(words), 1)  # quantidade de palavras da frase toda
            n_phrases = max(len(phrase), 1) # quantidade de frases do paragrafo
            types = set(words) # guarda todas as palavras mas sem repeticao

            # --- CÁLCULOS PARA FLESCH ---
            total_syllables = sum(self.count_syllables_pt(w) for w in words) # calcula o total de silabas das palavras
            size_mean_phrase =  n_words / n_phrases
            syllables_mean_word = total_syllables / n_words # media de silabas por palavra

            # Formula Indice de Flesch adaptada para o Portugues (Martins, 1993)
            # Quanto maior, mais facil (Ensino Fundamental). Quanto menor, mais dificil (Superior).
            flesch_index = 248.835 - (1.015 * size_mean_phrase) - (84.6 * syllables_mean_word)

            # --- METRICAS SINTATICAS (spaCy) ---
            # Passa o texto inteiro no spaCy para analise sintatica
            doc = nlp(row["text"])
            # Conta a frequencia de cada classe gramatical (POS)
            pos_counts = Counter([token.pos_ for token in doc if token.is_alpha])

            disc = {
                "n_canonicas": sum(self.is_canonica(p) for p in words) / max(len(words), 1), # media de palavras canonicas na frase
                "n_canonicas_type": sum(self.is_canonica(p) for p in types) / max(len(types), 1), # media de palavras canonicas por tipo
                "size_mean_words": sum(len(w) for w in words) / n_words,
                "size_mean_phrase": n_words / n_phrases,
                "virgulas_por_frase": row["text"].count(",") / n_phrases,
                "diversity": len(types) / max(len(words), 1),
                "flesch_reading_ease": flesch_index,
                "syllables_per_word": syllables_mean_word,
                
                # Metricas Sintaticas (Proporcao em relacao ao total de palavras)
                # SCONJ = Conjuncoes Subordinativas (ex: embora, porque, que). Forte indicativo de Ensino Medio/Superior.
                "prop_sconj": pos_counts.get("SCONJ", 0) / n_words,
                
                # CCONJ = Conjuncoes Coordenativas (ex: e, ou, mas). Mais comuns no Ensino Fundamental.
                "prop_cconj": pos_counts.get("CCONJ", 0) / n_words,
                
                # ADJ = Adjetivos. Textos academicos e do Ensino Medio tendem a ter vocabulario mais descritivo.
                "prop_adj": pos_counts.get("ADJ", 0) / n_words,
                
                # ADV = Adverbios.
                "prop_adv": pos_counts.get("ADV", 0) / n_words,
                
                # VERB = Verbos. 
                "prop_verb": pos_counts.get("VERB", 0) / n_words,
                
                # ADP = Preposicoes (de, em, para). Maior densidade de preposicoes indica frases mais longas e conectadas.
                "prop_adp": pos_counts.get("ADP", 0) / n_words
            }
            rows.append(disc)
        df = pd.DataFrame(rows)
        return df
        

    def create_phrases(self, docs, n_words:int = 200):
        """
            Funcao responsavel por dividir o campo texto do dataFrame em frases menores
            Args: 
                docs: [DataFrame] -> conjunto de dados (treino, teste, validacao)
                n_words: [int] -> numero maximo de palavras que uma frase tera
            
            Return: 
                df: [DataFrame] -> um novo data frame com as frases reduzidas
        """
        rows = []
        for _, row in docs.iterrows():
            fragment = self.split_phrase(row["text"], n_words)
            for index, frag in enumerate(fragment):
                rows.append({"doc_id": row["doc_id"], "fragment_id": index, "text": frag, "classe": row["classe"]})

        df = pd.DataFrame(rows)
        return df

    def clear_text(self,text: str):
        """
            Funcao responsavel por limpar o texto lida do arquivo

            Args: text [str] -> texto do arquivo
            Return text [str] -> texto limpo 
        """
        text = text.strip()
        text = " ".join(text.split())
        return text

    def read(self, path_file):
        # utf-8-sig retira o BOM do inicio do arquivo
        with open(f'{path_file}', 'r', encoding='utf-8-sig') as f:
            text = f.read()
            text = self.clear_text(text=text)
            # text = sent_tokenize(text, language="portuguese")

        # frases = sent_tokenize("O Sr. Silva chegou. Ele sorriu.", language="portuguese")
        return text

    def is_canonica(self, word: str) -> bool:
        """
            Funcao responsavel por verificar se uma palavra eh caninica ou nao
            Decisoes:
                - Ditongos (pai, muito, mao) viram duas vogais seguidas, entao a palavra eh nao canonica.
                - Palavras de uma vogal so ("a", "o", "e") sao nao canonicas
                - ss e rr contam como uma unica consoante
                - x eh tratado como consoante simples
            Args: 
                text [str] -> palavra a ser verificada
            
            Return:
                bool -> verdadeiro ou falso, se a palavra for canonica ou nao
        """
        VOGAIS = "aeiouáéíóúâêôãõàü"
        C = r"(?:ch|lh|nh|rr|ss|qu(?=[eéêií])|gu(?=[eéêií])|[b-df-hj-np-tv-zç])"
        V = f"[{VOGAIS}]"
        CANONICA = re.compile(f"^(?:{C}{V})+$")

        p = word.lower()
        return p.isalpha() and bool(CANONICA.match(p))

    def count_syllables_pt(self, word: str):
        """
            Estimativa rapida de silabas em portugues contando grupos de vogais.
            Para uma precisao perfeita, considere usar a biblioteca 'pyphen'.
        """
        word = word.lower()
        # Encontra grupos de vogais juntas (ex: 'ei', 'a', 'ão')
        vowel_groups = re.findall(r'[aeiouáéíóúâêîôûãõ]+', word)
        return max(1, len(vowel_groups))