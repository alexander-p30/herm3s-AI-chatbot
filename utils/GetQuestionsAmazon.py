import bs4

def get_faq(soup):
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
        if resposta and not (resposta in respostas):
            respostas.append(resposta)

    for i in range(0, len(perguntas)):
        lista.append(
            {
             "pergunta": perguntas[i].text.strip(),
             "respostas": respostas[i].text if i < len(respostas) else ""
            }
        )

    return lista


if __name__ == '__main__':
    from os import walk
    directory = '../arquivos/'
    faq_list = []
    cont = 0
    for (dirpath, dirnames, filenames) in walk(directory):
        for filename in filenames:
            html = open(directory + filename)
            soup = bs4.BeautifulSoup(html, 'html.parser')
            faq = get_faq(soup)
            faq_list.append(faq)
    for faq_li in faq_list:
        cont += len(faq_li)
    print(faq_list)
    print(cont)