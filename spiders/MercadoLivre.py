import scrapy


class HermesSpiderML(scrapy.Spider):
    name = "HermesSpiderML"
    start_urls = ["https://www.mercadolivre.com.br/fritadeira-sem-oleo-multilaser-air-fryer-4-l-gourmet-vermelha-110v/p/MLB15064648?source=search#searchVariation=MLB15064648&position=1&type=product&tracking_id=0f6f270c-3bb8-4e5e-8c8e-70c07d3dd4d2"]
    download_delay = 2.0

    def parse(self, response):
        filename = 'arquivos/mercado_livre.html'

        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)
