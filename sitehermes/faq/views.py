from django.shortcuts import render
from .models import Product
from .forms import FormsFaq
from utils.compare_questions import compare_question_amazon_product
# Create your views here.


def main(request):
    template_name = "main.html"

    context = {
        "Products": Product.objects.all(),
    }
    return render(request, template_name, context)


def show(request, **kwargs):
    template_name = 'product_show.html'
    form = FormsFaq(request.POST or None)
    context = {
        "Product": Product.objects.filter(id=int(kwargs['pk'])).get(),
    }
    if form.is_valid():
        # Coleta a pergunta do formulário
        pergunta = form.cleaned_data.get('pergunta')
        reposta = compare_question_amazon_product(pergunta)
        context["resposta"] = reposta[1]["respostas"]
        context["porcentagem"] = reposta[0]
        context["pergunta"] = reposta[1]["pergunta"]
    context["form"] =form
    return render(request, template_name, context)

