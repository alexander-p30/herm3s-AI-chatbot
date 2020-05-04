from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import spacy
import fasttext

def InicializarAI():
	ft_model = fasttext.load_model('cc.pt.50.bin')
	spacy_nlp = spacy.load("pt_core_news_sm")

	return [spacy_nlp, ft_model]

def Treinamento(spacy_nlp, ft_model):
	i = 1
	embed1 = []
	embed2 = []
	fuzzylist = []
	spacy = []
	y = []
	par_pergunta_resultado = {}
	lista_pares_perguntas_resultado = []

	arq = open('base_de_treino.txt', 'r')

	while(i < 50):
		par_pergunta_resultado['pergunta1'] = arq.readline()
		par_pergunta_resultado['pergunta1'] = par_pergunta_resultado['pergunta1'].rstrip('\n')

		par_pergunta_resultado['pergunta2'] = arq.readline()
		par_pergunta_resultado['pergunta2'] = par_pergunta_resultado['pergunta2'].rstrip('\n')

		par_pergunta_resultado['resultado'] = arq.readline()
		par_pergunta_resultado['resultado'] = par_pergunta_resultado['resultado'].rstrip('\n')

		lista_pares_perguntas_resultado.append({'pergunta1' : par_pergunta_resultado['pergunta1'], 'pergunta2' : par_pergunta_resultado['pergunta2'], 'resultado' : par_pergunta_resultado['resultado']})

		i = i + 1

	arq.close()

	for par_pergunta_resultado in lista_pares_perguntas_resultado:

		pergunta1 = par_pergunta_resultado['pergunta1']
		pergunta2 = par_pergunta_resultado['pergunta2']
		resultado = par_pergunta_resultado['resultado']

		y.append(resultado)

		numero_fuzzy = fuzz.partial_ratio(pergunta1, pergunta2)

		fuzzylist.append(numero_fuzzy)

		spacy_doc1 = spacy_nlp(pergunta1)
		spacy_doc2 = spacy_nlp(pergunta2)
		numero_spacy = spacy_doc1.similarity(spacy_doc2)

		spacy.append(numero_spacy)

		vetor_pergunta1 = ft_model.get_sentence_vector(pergunta1)
		embed1.append(vetor_pergunta1)

		vetor_pergunta2 = ft_model.get_sentence_vector(pergunta2)
		embed2.append(vetor_pergunta2)

	return [embed1, embed2, fuzzylist, spacy, y]
