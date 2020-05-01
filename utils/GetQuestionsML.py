import bs4

html = open('arquivos/mercado_livre.html')
soup = bs4.BeautifulSoup(html, 'html.parser')

lista = []
perguntas = soup.findAll("p", {"class": "ui-pdp-color--BLACK ui-pdp-size--SMALL ui-pdp-family--REGULAR ui-pdp-qadb__questions-list__question__label"})
respostas = soup.findAll("p", {"class": "ui-pdp-color--BLACK ui-pdp-size--SMALL ui-pdp-family--REGULAR ui-pdp-qadb__questions-list__answer-item__answer"})
for i in range(0, len(perguntas)):
    lista.append({"pergunta": perguntas[i].text, "respostas": respostas[i].text})

print(lista)