import bs4

html = open('../arquivos/amazon.html')
soup = bs4.BeautifulSoup(html, 'html.parser')

lista = []
testes = soup.findAll('div',{'class':'a-fixed-left-grid a-spacing-small'})
print(len(testes))
for teste in testes:
    pergunta = teste.find("span", 'a-declarative')
    print(pergunta.text)

testes2 = soup.findAll('div', {'class': 'a-fixed-left-grid-inner'})
print(len(testes2))
for teste in testes2:
    pergunta = teste.find("span", '')
    if pergunta:
        print(pergunta.text)

#perguntas = soup.findNext('div',{'class':'a-fixed-left-grid a-spacing-small'}).findAll("span", 'a-declarative')
# perguntas = soup.findAll("span", {"class": ""})
# respostas = soup.findAll("span", {"class": "ui-pdp-color--BLACK ui-pdp-size--SMALL ui-pdp-family--REGULAR ui-pdp-qadb__questions-list__answer-item__answer"})
#or pergunta in perguntas:
#    print(pergunta.text)
#for i in range(0, len(perguntas)):
#    lista.append({"pergunta": perguntas[i].text, "respostas": respostas[i].text})

print(lista)