import random
import spacy
from spacy.symbols import PROPN, CCONJ, SCONJ, nsubj
# nltk.download('wordnet')
# nltk.download('omw')
from nltk.corpus import wordnet as wn
from functools import lru_cache

# Limite minimo de similaridade
# Palavras contextualmente corretas tendem a pontuar mais alto
LIMITE_SIMILARIDADE = 0.40

class MutationPhrase():
    def __init__(self):
        pass

    @lru_cache(maxsize=1)
    def get_nlp(self):
        import spacy
        return spacy.load("pt_core_news_lg")

    def tokens_validate(self, phrase: str) -> list:
        """
            Funcao responsavel por gerar tokens validos para buscar sinonimos
            Tipos Validos:
                - Substantivos
                - Adjetivos
            Args:
                - phrase [str] -> frase original para extrair os tokens validos
            Return:
                - tokens [list] -> lista de tokens validos encotrados
        """
        nlp = self.get_nlp()
        doc = nlp(phrase)
        tokens = [token for token in doc if token.pos_ in ["NOUN", "ADJ", "ADV"] and not token.is_punct]

        return tokens

    def get_synonyms(self, word: str, lang: str = "por") -> list:
        """
            Busca sinonimos para uma palavra no idioma informado usando o WordNet
            Args:
                - word [str] -> palavra na qual sera buscada sinonimos
                - lang [str] -> idioma da palavra. padrao eh portugues
            Return:
                - synonyms [list] -> lista dos sinonimos encontrados
        """
        synonyms = set()
        for synset in wn.synsets(word, lang=lang):
            for lemma in synset.lemmas(lang=lang):
                subs = lemma.name().replace("_", " ")
                if subs.lower() != word.lower():
                    synonyms.add(subs)
        return list(synonyms)

    def filter_synonyms_context(self, phrase: str, word: str, synonyms: list) -> list:
        """
            Filtra sinonimos que nao combinam semanticamente com o contexto da frase original
            por um percentual de simularidade
            Args:
                - phrase [str] -> frase original para comparar o contexto
                - word [str] -> palavra original dos sinonimos encontrados
                - synonymds [list] -> sinonimos encontrados para word
            Return:
                - syn_filter [list] -> lista com os sinonimos filtrados por contexto
        """
        nlp = self.get_nlp()
        doc_phrase = nlp(phrase)
        synonyms_valid = []

        for syn in synonyms:
            doc_syn = nlp(syn)

            # o spaCy calcula a similaridade media do termo com a frase inteira
            score = doc_phrase.similarity(doc_syn)

            # Opcional: print para ver os scores rodando no terminal
            # print(f"Candidato: {syn} -> Score: {score:.4f}")

            if score >= LIMITE_SIMILARIDADE:
                synonyms_valid.append((syn, score))

        # Ordena do sinonimo mais contextual para o menos contextual
        synonyms_valid.sort(key=lambda x: x[1], reverse=True)

        # Retorna apenas o texto dos sinonimos filtrados
        syn_filter = [s[0] for s in synonyms_valid]
        return syn_filter

    def swap_synonyms(self, phrase: str) -> str:
        tokens = self.tokens_validate(phrase)
        synonyms_filter = []
        if not tokens:
            return "Sem tokens validos na frase informada!"

        random.shuffle(tokens)
        for token in tokens:
            word_choice = token
            synonyms = self.get_synonyms(word_choice.text)
            synonyms_filter = self.filter_synonyms_context(phrase=phrase, word=word_choice.text, synonyms=synonyms)
            if synonyms_filter:
                break

        if not synonyms_filter:
            return "Sem sinonimos validos para a frase escolhida"

        synonym_choice = synonyms_filter[0]
        new_phrase = []
        nlp = self.get_nlp()
        doc = nlp(phrase)
        for tok in doc:
            if tok.i == word_choice.i:
                new_phrase.append(synonym_choice + tok.whitespace_)
            else:
                new_phrase.append(tok.text + tok.whitespace_)

        return "".join(new_phrase).strip()

    def parafra_suj(self, phrase: str) -> str:
        """
            Funcao responsavel por parafrasear uma frase que possui sujeito
            Args:
                - phrase [str] -> frase que sera parafraseada
            Return:
                - new_phrase [str] -> nova frase depois de alterada
        """
        nlp = self.get_nlp()
        phrase = phrase.strip()
        doc = nlp(phrase)

        verb_main = None
        sujeito_tokens = []
        aux_tokens = []
        objs = []
        
        # 1. Identificar o Sujeito e o Verbo Principal
        for token in doc:
            if token.dep_ in ["nsubj", "nsubj:pass"]:
                verb_main = token.head
                sujeito_tokens = list(token.subtree)
                break
        
        # Se não encontrou sujeito ou verbo principal, retorna a frase original sem quebrar
        # arrumar para quando nao possui somente sujeito
        if not verb_main or not sujeito_tokens:
            return phrase

        # 2. Identificar todos os Verbos Auxiliares e advérbios entre o auxiliar e o principal
        for token in doc:
            if token.i < verb_main.i and "aux" in token.dep_ and token not in sujeito_tokens:
                # if token not in sujeito_tokens:
                    aux_tokens.append(token)

        # 3. Identificar todos os Objetos Diretos e Indiretos da frase
        for token in doc:
            # Verifica se é Objeto Direto (obj) ou Objeto Indireto (iobj)
            if token.dep_ in ["obj", "iobj"] and token.head == verb_main:
                objs.extend(list(token.subtree))

        # 4. Montar as partes da nova frase
        # Bloco do Verbo (Auxiliares + Verbo Principal)
        if not len(aux_tokens):
            bloco_verbo = [verb_main] + [t for t in objs]
        else:
            bloco_verbo = [t for t in aux_tokens] + [verb_main]
        bloco_verbo_str = " ".join([t.text for t in bloco_verbo])
        bloco_verbo_str = bloco_verbo_str[0].upper() + bloco_verbo_str[1:].lower()

        # Bloco do Sujeito (Preserva Nomes Próprios em maiúsculo)
        bloco_sujeito = [
            tok.text if tok.pos_ == "PROPN" else tok.text.lower()
            for tok in sujeito_tokens
        ]
        bloco_sujeito_str = " ".join(bloco_sujeito)

        # Bloco Restante (Tudo após o verbo principal e antes do objeto)
        tokens_utilizados = set(sujeito_tokens + bloco_verbo + objs)
        restante_tokens = [
            tok.text
            for tok in doc
            if not tok.is_punct and tok not in tokens_utilizados
        ]
        restante_str = " ".join(restante_tokens)

        # Captura a pontuação final original (se houver)
        pontuacao_final = doc[-1].text if doc[-1].is_punct else "."

        # 4. Unir tudo
        nova_frase = f"{bloco_verbo_str} {bloco_sujeito_str}"
        if restante_str:
            nova_frase += f" {restante_str}"

        return nova_frase + pontuacao_final
        
    def parafrasear(self, phrase: str):
        """
            Funcao responsavel por parafrasear uma frase informada
            Args:
                - phrase [str] -> frase original que devera ser parafraseada
            Return:
                - new_phrase [str] -> frase nova depois de ser parafraseada
        """
        new_phrase = self.parafra_suj(phrase)
        return new_phrase

# mut = MutationPhrase()
# result = mut.parafrasear("O projeto foi aprovado por Maria.")
# result = mut.parafrasear("O projeto inovador foi aprovado, mas a equipe antiga saiu.")
# result = mut.parafrasear("A taxa de aprovação subiu porque os alunos estudaram mais.")
# result = mut.parafrasear("Há duas semanas atrás estava chovendo")
# result = mut.parafrasear("O projeto foi aprovado por Maria da Silva em São Paulo.")
# result = mut.parafrasear("O cantor Roberto Carlos fez aniversário em São Paulo")
# result = mut.parafrasear("O aluno cuidadosamente fez o trabalho")
# result = mut.parafrasear("O texto foi lido pelo aluno")
# new_phrase = mut.mutation("O texto foi lido pelo aluno")
# print(result)