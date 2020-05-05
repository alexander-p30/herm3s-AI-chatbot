from django.db import models


class Faq(models.Model):
    question = models.CharField(max_length=200)
    answer = models.CharField(max_length=200)

    class Meta:
        verbose_name = 'Faq de perguntas e respostas'


class Product(models.Model):
    name = models.CharField(max_length=200)
    link_image = models.CharField(max_length=400)
    id_amazon = models.CharField(max_length=100)
    link_amazon = models.CharField(max_length=300)
    desc = models.CharField(max_length=300)