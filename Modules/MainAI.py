from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import spacy
import fasttext
import base

def InicializarAI():
	ft_model = fasttext.load_model('cc.pt.50.bin')
	spacy_nlp = spacy.load("pt_core_news_sm")

	return [spacy_nlp, ft_model]



#rode a função inicializar AI antes disso
def AvaliarPergunta(pergunta_cliente_atual, lista_lista_perguntas_respostas_conhecidas, spacy_nlp, ft_model): 
	lista_perguntas_repetidas = []

	spacy_doc1 = spacy_nlp(pergunta_cliente_atual)

	vetor_pergunta_cliente_atual = ft_model.get_sentence_vector(pergunta_cliente_atual)

	for lista_perguntas_respostas_conhecidas in lista_lista_perguntas_respostas_conhecidas:
		for pergunta_resposta_conhecida_atual in lista_perguntas_respostas_conhecidas:
			pergunta_conhecida_atual = pergunta_resposta_conhecida_atual['pergunta']
			numero_fuzzy = fuzz.partial_ratio(pergunta_cliente_atual, pergunta_conhecida_atual)
			spacy_doc2 = spacy_nlp(pergunta_conhecida_atual)
			numero_spacy = spacy_doc1.similarity(spacy_doc2)
			vetor_pergunta_conhecida_atual = ft_model.get_sentence_vector(pergunta_conhecida_atual)
			similaridade = rede_neural(vetor_pergunta_cliente_atual, vetor_pergunta_conhecida_atual, numero_fuzzy, numero_spacy)
			if similaridade > 0.5:
				lista_perguntas_repetidas.append(pergunta_resposta_conhecida_atual)
	
	return lista_perguntas_repetidas

def rede_neural(vetor_pergunta_cliente_atual, vetor_pergunta_conhecida_atual, numero_fuzzy, numero_spacy):
	return 1