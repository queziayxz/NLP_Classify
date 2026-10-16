import pandas as pd
import pyphen
import spacy
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize

nlp = spacy.load("pt_core_news_sm")
dic = pyphen.Pyphen(lang='pt_BR')

class Features():
    def __init__(self):
        pass

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

    def create(self, fragments: list) -> pd.DataFrame:
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
        for frag in fragments:
            phrase = sent_tokenize(frag, language="portuguese")
            
            # separa as frases em palavras e pontuacao
            # transforma tudo para minusculo descartando numeros e pontuacao com o isalpha. retira tambem palavras com hifen (guarda-chuva)
            words = [t for t in word_tokenize(frag, language="portuguese") if t.isalpha()]
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
            doc = nlp(frag)
            # Conta a frequencia de cada classe gramatical (POS)
            pos_counts = Counter([token.pos_ for token in doc if token.is_alpha])

            disc = {
                "n_canonicas": sum(self.is_canonica(p) for p in words) / max(len(words), 1), # media de palavras canonicas na frase
                "n_canonicas_type": sum(self.is_canonica(p) for p in types) / max(len(types), 1), # media de palavras canonicas por tipo
                "size_mean_words": sum(len(w) for w in words) / n_words,
                "size_mean_phrase": n_words / n_phrases,
                "virgulas_por_frase": frag.count(",") / n_phrases,
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

    def is_canonica(self, word: str) -> bool:
        """
            Funcao responsavel por verificar se uma palavra eh caninica ou nao
            Decisoes:
                - Ditongos (pai, muito, mao) viram duas vogais seguidas, entao a palavra eh nao canonica.
                - Palavras de uma vogal so ("a", "o", "e") sao nao canonicas
            Args: 
                text [str] -> palavra a ser verificada
            
            Return:
                bool -> verdadeiro ou falso, se a palavra for canonica ou nao
        """
        word = word.lower()
        silabas = dic.inserted(word).split('-')
        
        vogais = "aeiouáéíóúâêôãõüà"
        
        for silaba in silabas:
            # Cria o padrão da sílaba (ex: "ca" vira "CV", "trans" vira "CCVCC")
            padrao = "".join(["V" if letra in vogais else "C" for letra in silaba if letra.isalpha()])
            
            # Se alguma sílaba NÃO for rigorosamente Consoante + Vogal (CV), a palavra não é canônica
            if padrao != "CV":
                return False
        return True

    def count_syllables_pt(self, word: str):
        """
            Estimativa rapida de silabas em portugues contando grupos de vogais.
            Para uma precisao perfeita, considere usar a biblioteca 'pyphen'
            Args:
                - word [str] -> palavra a ser contada quantas silabas possui
            Return:
                - quant [int] -> quantidade de silabas da palavra
        """
        word = word.lower()
        word_hifen = dic.inserted(word)
        return len(word_hifen.split('-'))

# teste = Features()
# print(teste.is_canonica("c"))