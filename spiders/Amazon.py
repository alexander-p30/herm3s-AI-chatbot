import scrapy

class HermesSpiderAmazon(scrapy.Spider):
    name = "HermesSpiderAmazon"
    start_urls = ["https://www.amazon.com.br/ask/questions/asin/B001E5MO5E"]
    download_delay = 2.0

    def parse(self, response):
        filename = 'arquivos/amazon.html'

        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)