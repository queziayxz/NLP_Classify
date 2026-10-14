import random
import spacy
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

    def parafrasear(self, phrase: str):
        """
            Funcao responsavel por parafrasear uma frase informada
            Args:
                - phrase [str] -> frase original que devera ser parafraseada
            Return:
                - new_phrase [str] -> frase nova depois de ser parafraseada
        """
        nlp = self.get_nlp()
        phrase = phrase.strip()
        doc = nlp(phrase)

        for token in doc:
            # CCONJ = Conjunção Coordenativa | SCONJ = Conjunção Subordinativa
            if token.pos_ in ["CCONJ", "SCONJ"]:
                print(token)
                synonyms = self.get_synonyms(token.text)
                filter_sy = self.filter_synonyms_context(phrase=phrase, word=token.text, synonyms=synonyms)
                print(f"novo token: {len(filter_sy) > 0 : filter_sy[0] ? token}")
                # print(f"{token.text:<15} | {token.pos_:<25} | {spacy.explain(token.pos_)}")
        # print(phrase)

mut = MutationPhrase()
mut.parafrasear("A taxa de aprovação subiu porque os alunos estudaram mais")
# new_phrase = mut.mutation("O texto foi lido pelo aluno")
# print(f"nova frase: {new_phrase}")