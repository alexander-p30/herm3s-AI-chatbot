import scrapy


# Criar um diretório com nome 'arquivos' a partir do diretório do projeto
# Dentro do diretório 'arquivos' criar um diretório chamado 'B001E5MO5E'
# Rodar no terminar 'scrapy runspider spiders/Amazon.py'
class HermesSpiderAmazon(scrapy.Spider):
    name = "HermesSpiderAmazon"
    start_urls = ["https://www.amazon.com.br/ask/questions/asin/B07R5PSTC9"]
    download_delay = 2.0
    cont = 0

    def parse(self, response):
        next_page = []
        filename = 'arquivos/B07R5PSTC9/amazon-%s.html' % self.cont
        self.cont += 1
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)

        # Seleciona todos os link que tem oferta de dados no nome
        NEXT_PAGE_SELECTOR_A = 'a::attr(href)'
        NEXT_PAGE_SELECTOR = '//li[has-class("a-last")]'

        response_href = response.xpath(NEXT_PAGE_SELECTOR).css(NEXT_PAGE_SELECTOR_A).extract()

        print(response_href)
        # Adicionando a url inicial nos links obtidos
        next_page.append("https://www.amazon.com.br/" + response_href[0])

        # Para todas as urls
        for page in next_page:
            yield scrapy.http.Request(page, callback=self.parse)
