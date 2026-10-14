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

    def parafra_cconj_sconj(self, phrase: str) -> str:
        """
            Funcao responsavel por parafrasear uma frase que possui conjuncao
            Args:
                - doc -> documento nlp carregado
                - phrase [str] -> frase que sera parafraseada
                - token [str] -> conjuncao encontrada na funcao principal
                - index [int] -> indice do token conjuncao encontrada no texto
            Return:
                - new_phrase [str] -> nova frase depois de alterada
        """
        nlp = self.get_nlp()
        phrase = phrase.strip()
        doc = nlp(phrase)
        index = -1
        exists = False
        new_phrase = []
        
        for token in doc:
            index += 1
            # CCONJ = Conjunção Coordenativa | SCONJ = Conjunção Subordinativa
            if token.pos in [CCONJ, SCONJ]:
                synonyms = self.get_synonyms(token.text)
                filter_sy = self.filter_synonyms_context(phrase=phrase, word=token.text, synonyms=synonyms)
                new_tok = filter_sy[0] if len(filter_sy) > 0 else token.text
                exists = True
                break
  
        if not exists: return ""
        
        new_phrase.append(new_tok[0].upper() + new_tok[1:] + " ")
        for tok in doc[(index+1):]:
            if not tok.is_punct:
                new_phrase.append(tok.text + tok.whitespace_ if tok.pos == PROPN else tok.text.lower() + tok.whitespace_)
        new_phrase.append(", ")
        for tok in doc[:index]:
            if not tok.is_punct:
                new_phrase.append(tok.text + tok.whitespace_ if tok.pos_ == PROPN else tok.text.lower() + tok.whitespace_)
        
        return "".join(new_phrase).strip()

    def parafra_propn(self, phrase: str) -> str:
        """
            Funcao responsavel por parafrasear uma frase que possui pronome proprio
            Args:
                - doc -> documento nlp carregado
                - phrase [str] -> frase que sera parafraseada
                - token [str] -> conjuncao encontrada na funcao principal
                - index [int] -> indice do token conjuncao encontrada no texto
            Return:
                - new_phrase [str] -> nova frase depois de alterada
        """
        nlp = self.get_nlp()
        phrase = phrase.strip()
        doc = nlp(phrase)
        index = -1
        exists = False
        new_phrase = []
        new_tok = []
        
        for token in doc:
            index += 1
            # CCONJ = Conjunção Coordenativa | SCONJ = Conjunção Subordinativa
            if token.ent_type_: # Se fizer parte de um nome próprio/entidade mapeada
                # Pega todo o bloco da entidade à qual esse token pertence
                nome_proprio_completo = token.doc[token.left_edge.i : token.right_edge.i + 1].ents[0].text
                print(nome_proprio_completo)
                new_tok = nome_proprio_completo
                exists = True
                break
                
        if not exists: return ""
        new_phrase.append(new_tok + ", ")
        for tok in doc[:index]:
            if not tok.is_punct:
                new_phrase.append(tok.text + tok.whitespace_ if tok.pos_ == PROPN else tok.text.lower() + tok.whitespace_)
        new_phrase.append(", ")
        for tok in doc[(index+1):]:
            if not tok.is_punct:
                new_phrase.append(tok.text + tok.whitespace_ if tok.pos == PROPN else tok.text.lower() + tok.whitespace_)

        return "".join(new_phrase).strip()

    def parafra_suj(self, phrase: str) -> str:
        """
            Funcao responsavel por parafrasear uma frase que possui sujeito
            Args:
                - doc -> documento nlp carregado
                - phrase [str] -> frase que sera parafraseada
                - token [str] -> conjuncao encontrada na funcao principal
                - index [int] -> indice do token conjuncao encontrada no texto
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
            if token.i < verb_main.i and (
                "aux" in token.dep_ or token.head == verb_main
            ):
                if token not in sujeito_tokens:
                    aux_tokens.append(token)

        # 3. Identificar todos os Objetos Diretos e Indiretos da frase
        for token in doc:
            # Verifica se é Objeto Direto (obj) ou Objeto Indireto (iobj)
            if token.dep_ in ["obj", "iobj"]:
                objs = list(token.subtree)

        # 4. Montar as partes da nova frase
        # Bloco do Verbo (Auxiliares + Verbo Principal)
        if not len(aux_tokens):
            bloco_verbo = [verb_main.text] + [t.text for t in objs]
        else:
            bloco_verbo = [t.text for t in aux_tokens] + [verb_main.text]
        bloco_verbo_str = " ".join(bloco_verbo)
        bloco_verbo_str = bloco_verbo_str[0].upper() + bloco_verbo_str[1:].lower()

        # Bloco do Sujeito (Preserva Nomes Próprios em maiúsculo)
        bloco_sujeito = [
            tok.text if tok.pos_ == "PROPN" else tok.text.lower()
            for tok in sujeito_tokens
        ]
        bloco_sujeito_str = " ".join(bloco_sujeito)

        # Bloco Restante (Tudo após o verbo principal e antes do objeto)
        restante_tokens = [
            tok.text
            for tok in doc[verb_main.i + 1 :]
            if not tok.is_punct and tok not in (sujeito_tokens and objs)
        ]
        restante_str = " ".join(restante_tokens)

        # Bloco Objetos
        # bloco_obj = [
        #     tok.text if tok.pos_ == "PROPN" else tok.text.lower()
        #     for tok in objs
        # ]
        # bloco_obj_str = " ".join(bloco_obj)

        # Bloco Restante (Tudo após o objeto)
        # restante_tokens_end = [
        #     tok.text
        #     for tok in doc[.i + 1 :]
        #     if not tok.is_punct and tok not in sujeito_tokens
        # ]
        # restante_str_end = " ".join(restante_tokens_end)

        # Captura a pontuação final original (se houver)
        pontuacao_final = doc[-1].text if doc[-1].is_punct else "."

        # 4. Unir tudo
        nova_frase = f"{bloco_verbo_str} {bloco_sujeito_str}"
        if restante_str:
            nova_frase += f" {restante_str}"

        return nova_frase + pontuacao_final

        # new_phrase.append(aux_main[0].text[0].upper() + aux_main[0].text[1:])
        # new_phrase.append(aux_main[1].text.lower() if not aux_main[1].pos == PROPN else aux_main[1].text)
        # new_phrase.append(" ".join([tok.text.lower() for tok in sujeito_completo if not tok.pos == PROPN]))
        # new_phrase.append(" ".join([tok.text for tok in doc[(verb_main.i+1):] if not tok.is_punct]))

        # return " ".join(new_phrase).strip()
        
    def parafrasear(self, phrase: str):
        """
            Funcao responsavel por parafrasear uma frase informada
            Args:
                - phrase [str] -> frase original que devera ser parafraseada
            Return:
                - new_phrase [str] -> frase nova depois de ser parafraseada
        """
        # new_phrase = self.parafra_cconj_sconj(phrase)
        # if not len(new_phrase):
        new_phrase = self.parafra_suj(phrase)
        # if not len(new_phrase):
        #     new_phrase = self.parafra_propn(phrase)
        #     print(new_phrase)
        return new_phrase

mut = MutationPhrase()
# result = mut.parafrasear("O projeto foi aprovado por Maria.")
# result = mut.parafrasear("O projeto inovador foi aprovado, mas a equipe antiga saiu.")
# result = mut.parafrasear("A taxa de aprovação subiu porque os alunos estudaram mais.")
# result = mut.parafrasear("Há duas semanas atrás estava chovendo")
# result = mut.parafrasear("O projeto foi aprovado por Maria da Silva em São Paulo.")
# result = mut.parafrasear("O cantor Roberto Carlos fez aniversário em São Paulo")
result = mut.parafrasear("O texto foi lido pelo aluno")
# new_phrase = mut.mutation("O texto foi lido pelo aluno")
print(result)