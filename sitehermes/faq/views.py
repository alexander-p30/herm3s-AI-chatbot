from django.shortcuts import render

# Create your views here.


def main(request):
    template_name = "main.html"

    context = {
        "Conteudo": "Abacate",
    }
    return render(request, template_name, context)
