import random
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
        tokens = [token for token in doc if token.pos_ in ["NOUN", "ADJ"] and not token.is_punct]

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

    def encaixa(self, doc, alvo, cand, razao=0.9):
        vizinhos = [t for t in doc if t.i != alvo.i and t.pos_ in {"NOUN", "ADJ"} and t.has_vector]
        if not vizinhos or not cand.has_vector:
            return False
        media = lambda w: sum(w.similarity(v) for v in vizinhos) / len(vizinhos)
        return media(cand) >= razao * media(alvo)

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

        alvo = next((t for t in doc_phrase if t.text == word), None)

        for syn in synonyms:
            syn_limpo = syn.strip()
        
            if " " in syn_limpo or "_" in syn_limpo:
                continue

            doc_syn = nlp(syn)

            if not doc_syn.has_vector or doc_syn.vector_norm == 0 or alvo is None or not alvo.has_vector:
                continue 

            if (not doc_syn[0].morph.get("Gender") == alvo.morph.get("Gender")) or (
                not doc_syn[0].morph.get("Number") == alvo.morph.get("Number")):
                continue

            # o spaCy calcula a similaridade media do termo com a frase inteira
            score = doc_phrase.similarity(doc_syn)

            if self.encaixa(doc_phrase, alvo, doc_syn):
                # if score >= LIMITE_SIMILARIDADE:
                synonyms_valid.append((syn, score))
                # synonyms_valid.append((syn, score))

            # Opcional: print para ver os scores rodando no terminal
            # print(f"Candidato: {syn} -> Score: {score:.4f}")


        # Ordena do sinonimo mais contextual para o menos contextual
        synonyms_valid.sort(key=lambda x: x[1], reverse=True)

        # Retorna apenas o texto dos sinonimos filtrados
        syn_filter = [s[0] for s in synonyms_valid]
        return syn_filter

    def swap_synonyms(self, phrase: str) -> str:
        tokens = self.tokens_validate(phrase)
        synonyms_filter = []
        if not tokens:
            raise Exception("Sem tokens válidos na frase. Informe um texto que contenha Substantivos e Adjetivos")

        random.seed(None)
        random.shuffle(tokens)
        for token in tokens:
            word_choice = token
            synonyms = self.get_synonyms(word_choice.text)
            synonyms_filter = self.filter_synonyms_context(phrase=phrase, word=word_choice.text, synonyms=synonyms)
            if synonyms_filter:
                break

        if not synonyms_filter:
            raise Exception("Sinonimos encontrados porém com baixa similaridade no contexto")

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

    def parafra_without_suj(self, phrase: str) -> str: 
        """
            Funcao responsavel por parafrasear uma frase que nao possui sujeito
            Args:
                - phrase [str] -> frase que sera parafraseada
            Return:
                - new_phrase [str] -> nova frase depois de alterada
        """
        nlp = self.get_nlp()
        phrase = phrase.strip()
        doc = nlp(phrase)
        verb_main = None

        # 1. Localiza o verbo principal (ROOT)
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"]:
                verb_main = token
                break

        if not verb_main:
            return phrase
        
        pre_tokens = []
        negacoes = []

        # 2. Separa os termos que estão ANTES do verbo
        for token in doc[:verb_main.i]:
            # Negações devem ficar ligadas ao verbo - ver se tem como pegar de forma automatica com spacy
            if token.dep_ == "advmod" and token.text.lower() in ["não", "nunca", "jamais"]:
                negacoes.append(token.text)
            else:
                pre_tokens.append(token)

        # Se não houver termos antes do verbo para mover, retorna a frase original
        if not pre_tokens:
            return phrase

        # 3. Termos APÓS o verbo principal (sem pontuação final)
        post_tokens = [tok for tok in doc[verb_main.i + 1:] if not tok.is_punct]

        # 4. Formatação dos Blocos
        # Bloco do Verbo (preserva negações como "Não há")
        str_negacao = " ".join(negacoes) + " " if negacoes else ""
        str_verbo = str_negacao + verb_main.text
        str_verbo = str_verbo[0].upper() + str_verbo[1:].lower()

        # Bloco Pós-Verbal
        str_post = " ".join([t.text for t in post_tokens])

        # Bloco Pré-Verbal movido para o final (ajusta maiúsculas/minúsculas)
        str_pre = " ".join([
            t.text if t.pos_ == "PROPN" else t.text.lower() 
            for t in pre_tokens
        ])

        # Captura a pontuação final
        pontuacao = doc[-1].text if doc[-1].is_punct else "."

        # 5. União: Verbo + Pós-Verbal + Pré-Verbal
        nova_frase = f"{str_verbo}"
        if str_post:
            nova_frase += f" {str_post}"
        if str_pre:
            nova_frase += f" {str_pre}"

        return nova_frase.strip() + pontuacao

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
            return self.parafra_without_suj(phrase)

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
            bloco_verbo = [verb_main] + [t for t in aux_tokens]
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

    def mutation(self, phrase: str) -> str:
        syn = phrase
        try:
            syn = self.swap_synonyms(phrase)
        except Exception as error:
            raise Exception(error["message"])
        finally:
            parafrase = self.parafrasear(syn)
            return parafrase
            
    
CASOS = [
    # --- básico -------------------------------------------------------
    ("B01", "basico", "O carro é muito grande."),
    ("B02", "basico", "A casa é bonita."),
    ("B03", "basico", "O menino ficou feliz com o presente."),
    ("B04", "basico", "A estrada era longa e estreita."),
    ("B05", "basico", "O professor fez uma pergunta difícil."),
    ("B06", "basico", "A festa foi um sucesso."),
    ("B07", "basico", "Ele tem medo de altura."),
    ("B08", "basico", "O filme tem um final triste."),
    ("B09", "basico", "A ideia parece perigosa."),
    ("B10", "basico", "O trabalho exigiu muita coragem."),
    # --- flexão -------------------------------------------------------
    ("F01", "flexao", "As casas bonitas ficam na rua principal."),
    ("F02", "flexao", "Os meninos felizes brincavam no parque."),
    ("F03", "flexao", "Duas estradas largas cruzam a cidade."),
    ("F04", "flexao", "Ela comprou roupas novas."),
    ("F05", "flexao", "Os alunos fizeram perguntas difíceis."),
    ("F06", "flexao", "As crianças ficaram cansadas."),
    ("F07", "flexao", "Ele usou palavras simples."),
    ("F08", "flexao", "Elas contaram histórias engraçadas."),
    # --- gênero do substantivo ------------------------------------------
    ("G01", "genero", "A casa fica longe da escola."),
    ("G02", "genero", "A pergunta ficou sem resposta."),
    ("G03", "genero", "A viagem foi longa."),
    ("G04", "genero", "O medo tomou conta dela."),
    ("G05", "genero", "A coragem dele impressionou todos."),
    ("G06", "genero", "O erro estava na primeira linha."),
    # --- polissemia ---------------------------------------------------
    ("P01", "polissemia", "Sentei no banco da praça."),
    ("P02", "polissemia", "A manga estava doce."),
    ("P03", "polissemia", "O avião pousou na pista."),
    ("P04", "polissemia", "Ele guardou a chave na gaveta."),
    ("P05", "polissemia", "A banda tocou no festival."),
    ("P06", "polissemia", "Meu pai trabalha no banco."),
    ("P07", "polissemia", "O gato subiu no telhado."),
    # --- fixas e dêiticos (não devem mudar) -------------------------------
    ("X01", "fixas", "Na semana passada fez muito frio."),
    ("X02", "fixas", "No próximo ano vamos viajar."),
    ("X03", "fixas", "Feliz Ano Novo para todos!"),
    ("X04", "fixas", "Ele chegou no último dia do curso."),
    ("X05", "fixas", "O mesmo aluno voltou à sala.", ["mesmo"], "preservar", ""),
    ("X06", "fixas", "Boa noite, pessoal.", ["boa", "noite"], "preservar", "cumprimento"),
    # --- nomes próprios e siglas (não devem mudar) -----------------------
    ("N01", "nomes", "Maria visitou o Museu Nacional em Brasília.", ["maria", "museu", "nacional", "brasília"], "preservar", ""),
    ("N02", "nomes", "O Banco do Brasil abriu uma nova agência.", ["banco", "brasil"], "preservar", "nome de instituição"),
    ("N03", "nomes", "Roberto Carlos é um cantor famoso.", ["roberto", "carlos"], "preservar", ""),
    ("N04", "nomes", "Em 2020, a ONU publicou um relatório importante.", ["onu", "2020"], "preservar", "sigla e número"),
    ("N05", "nomes", "Ela mora no Rio de Janeiro.", ["rio", "janeiro"], "preservar", "rio é substantivo comum, mas aqui é nome"),
    # --- sem tokens válidos ---------------------------------------------
    ("S01", "sem_validos", "Ele disse que sim.", [], "sem_troca", ""),
    ("S02", "sem_validos", "Eu fui e voltei.", [], "sem_troca", ""),
    ("S03", "sem_validos", "Ela está aqui agora.", [], "sem_troca", "só advérbios"),
    ("S04", "sem_validos", "Não sei.", [], "sem_troca", ""),
    # --- colocações -------------------------------------------------------
    ("C01", "colocacao", "Tomamos café da manhã juntos.", ["café", "manhã"], "preservar", "café da manhã"),
    ("C02", "colocacao", "Vamos dar uma volta no parque.", [], "trocar", "dar uma volta"),
    ("C03", "colocacao", "Ela tem olhos azuis."),
    ("C04", "colocacao", "O gato pulou o muro alto."),
    ("C05", "colocacao", "Fiz um bolo de chocolate."),
    # --- frases complexas -------------------------------------------------
    ("K01", "complexas", "Embora estivesse cansado, o professor corrigiu todas as provas antes de ir para casa."),
    ("K02", "complexas", "A menina que mora na casa azul ganhou um prêmio importante na escola.", [], "trocar", "oração relativa"),
    ("K03", "complexas", "O prefeito anunciou que a nova ponte será inaugurada no próximo mês.", ["próximo"], "preservar", ""),
    ("K04", "complexas", "Os estudantes que chegaram cedo conseguiram lugares melhores na biblioteca."),
    # --- ruído ------------------------------------------------------------
    ("R01", "ruido", "o carro novo quebrou ontem", [], "trocar", "sem maiúscula e sem ponto"),
    ("R02", "ruido", "ela comprou uma casa grande", [], "trocar", "tudo em minúsculas"),
    ("R03", "ruido", "A CASA É GRANDE.", [], "trocar", "caixa alta"),
    ("R04", "ruido", "A casa  é   bonita .", [], "trocar", "espaços irregulares"),
    # --- verbos auxiliares e locuções verbais -----------------------------
    # Simples (2 verbos: gerúndio, particípio ou infinitivo)
    ("V01", "auxiliares", "O aluno está lendo um livro interessante."),
    ("V02", "auxiliares", "Nós vamos viajar no próximo feriado."),
    ("V03", "auxiliares", "Ela tem trabalhado muito nos últimos dias."),
    ("V04", "auxiliares", "O prêmio principal foi entregue pelo diretor."),
    ("V05", "auxiliares", "Eles devem terminar o relatório ainda hoje."),
    
    # Intermediárias (Voz passiva e combinações modais)
    ("V06", "auxiliares", "O cachorro estava tentando abrir a porta pesada da cozinha."),
    ("V07", "auxiliares", "A casa já tinha sido vendida quando nós chegamos.", [], "trocar", "voz passiva composta"),
    ("V08", "auxiliares", "Ninguém conseguiu entender o que estava acontecendo ali."),
    
    # Complexas (Cadeias de 3 ou mais verbos e tempos compostos)
    ("V09", "auxiliares", "O problema técnico poderia ter sido resolvido muito mais rápido.", [], "trocar", "cadeia de 3 verbos"),
    ("V10", "auxiliares", "Se ela tivesse prestado atenção, não teria cometido o mesmo erro novamente."),
    ("V11", "auxiliares", "Os documentos que vinham sendo analisados pela equipe devem ser enviados amanhã de manhã."),
    ("V12", "auxiliares", "Mesmo que estivesse chovendo, o jogo não iria ser cancelado pelos organizadores."),
]

mut = MutationPhrase()
# # result = mut.parafrasear("O projeto foi aprovado por Maria.")
# # result = mut.parafrasear("O projeto inovador foi aprovado, mas a equipe antiga saiu.")
# # result = mut.parafrasear("A taxa de aprovação subiu porque os alunos estudaram mais.")
# # result = mut.parafrasear("Há duas semanas atrás estava chovendo")
# # result = mut.parafrasear("O projeto foi aprovado por Maria da Silva em São Paulo.")
# # result = mut.parafrasear("O cantor Roberto Carlos fez aniversário em São Paulo")
# # result = mut.parafrasear("O aluno cuidadosamente fez o trabalho")
# # result = mut.parafrasear("O texto foi lido pelo aluno")
# # result = mut.mutation("O texto foi lido pelo aluno")
# # result = mut.mutation("A taxa de aprovação subiu porque os alunos estudaram mais")
# # result = mut.mutation("A casa é bonita.")
for case in CASOS:
    print(f"{case[1]} - {case[2]}")
    result = mut.parafrasear(case[2])
    print(f"{result}")
    print("")