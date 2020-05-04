from django import forms

class FormsFaq(forms.Form):
    # Campo acera do assunto
    pergunta = forms.CharField(label='Faça uma pergunta para o Hermes, sobre o produto', widget=forms.TextInput(attrs={'class': 'form-control'}))