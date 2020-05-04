from django.shortcuts import render
from .models import Product
from .forms import FormsFaq
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
        reposta = "Resposta Padrão"
        context["resposta"] = reposta
    context["form"] =form
    return render(request, template_name, context)

