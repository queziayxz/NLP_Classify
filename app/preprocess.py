import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
# nltk.download("stopwords")
# nltk.download("punkt") 
# nltk.download("punkt_tab")

FUND_1 = "1_Ensino_Fundamental_I"
FUND_2 = "2_Ensino_Fundamental_II"
MEDIO = "3_Ensino_Medio"
SUPERIOR = "4_Ensino_Superior"

DIR_DATA = [FUND_1,FUND_2,MEDIO,SUPERIOR]

DIRETORIO_SCRIPT = Path(__file__).resolve().parent

# sobe uma hierarquia
DATASET = DIRETORIO_SCRIPT.parent / "dataset_caed"

class PreProcess():
    def __init__(self):
        pass

    def loadDataFrame(self):
        """
            Funcao responsavel por ler os arquivos de dados e criar um dataframe
            Columns:
                - "doc_id" [str] -> identificador unico da linha - sera o caminho absoluto do arquivo
                - "file_name" [str] -> nome do arquivo
                - "class" [str] -> classe de dificuldade do arquivo (Fundamental 1 ou 2, Médio, Superior)
                - "text" [str] -> texto bruto do arquivo
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
            Funcao responsavel por separar o dataframe em blocos de treino (70%), teste (15%) e validacao (15%)
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

        assert not set(x_train["doc_id"]) & set(x_test["doc_id"])
        assert not set(x_train["doc_id"]) & set(x_val["doc_id"])
        assert not set(x_val["doc_id"]) & set(x_test["doc_id"])

        return x_train, x_test, x_val

    def split_phrase(self, text: str, n_word: int) -> list:
        """
            Funcao responsavel por dividir um paragrafo em frases menores
            Args: 
                text: [str] -> texto a ser dividido
                n_words: [int] -> numero maximo de palavras que uma frase tera
            
            Return: 
                fragment: [list] -> lista contendo todas as novas frases separadas em menor quantidade
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
            Funcao responsavel por limpar o texto lido do arquivo
            Args: 
                - text [str] -> texto do arquivo
            Return:
                - text [str] -> texto limpo 
        """
        text = text.strip()
        text = " ".join(text.split())
        return text

    def read(self, path_file: str) -> str:
        """
            Funcao responsavel por ler o texto do arquivo informado
            Args: 
                - path_file [str] -> caminho do arquivo
            Return:
                - text [str] -> conteudo do arquivo 
        """
        # utf-8-sig retira o BOM do inicio do arquivo
        with open(f'{path_file}', 'r', encoding='utf-8-sig') as f:
            text = f.read()
            text = self.clear_text(text=text)
        return text