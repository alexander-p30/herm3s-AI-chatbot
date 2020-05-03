import bs4

html = open('../arquivos/amazon.html')
soup = bs4.BeautifulSoup(html, 'html.parser')

lista = []
respostas = []
perguntas = []

# Obtem as perguntas
question_items = soup.findAll('div', {'class': 'a-fixed-left-grid a-spacing-small'})
for question_item in question_items:
    if question_item:
        perguntas.append(question_item.find("span", 'a-declarative'))

# Obtem as respostas
answer_items = soup.findAll('div', {'class': 'a-fixed-left-grid-col a-col-right'})
for answer_item in answer_items:
    resposta = answer_item.find("span", '')
    if resposta and not(resposta in respostas):
        respostas.append(resposta)

for i in range(0, len(perguntas)):
    lista.append({"pergunta": perguntas[i].text.strip(), "respostas": respostas[i].text})

print(lista)
