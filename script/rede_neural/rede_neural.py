from .base import *
from ::Modules import PreprocessamentoTreinamentoAI
# MODELO
#   Rede neural em 3 camadas: 604 - 200 - 1. 

# CARREGAR OS EXEMPLOS AQUI (vetores de m linhas), remover essas funções aleatorias
# entrada
Init = PreprocessamentoTreinamentoAI.InicializarAI()
spacy = Init[0]
ft = Init[1]
Init = PreprocessamentoTreinamentoAI.Treinamento(spacy, ft)

embed1 = Init[0]
embed2 = Init[1]
fw = Init[2]
spy = Init[3]
# saída
y = Init[4]

# Configurações
# numero de exemplos
# graus maximo de polinomio para fw e spy
grau = 2
# elementos na camada escondida
hidden_camada_tamanho = 20
# coeficiente de regularização
lmbd = 0.5
# máximo de iterações para a função de otimização
max = 150
# define as fronteiras entre previsão 1 e 0
fronteira = 0.5
# proporções dos conjuntos de exemplos (treino, cv, teste)
fracoes = (6, 2, 2)

# Configurações automáticas (não alterar)
# número de exemplos
m = embed1.shape[0]
# número de elementos no embed
elementos_embed = embed1.shape[1]
# elementos na camada inicial, de input
input_camada_tamanho = 2 * elementos_embed + 2*grau + sum(range(grau))

# Gerar X
X = np.concatenate((embed1, embed2, obterCombinacoes(fw, spy, grau)), axis=1)

# print('Tamanho de X:', X.shape, 'Tamanho de y:', y.shape)
# normalizar as entradas (desnecessário para o método de otimização atual)
# X = normalizar(X)[0]

# retorna uma tupla: (previsão (gerada com base na fronteira), probabilidade (resultado da rede neural))
def PrevisaoPara(embed1t, embed2t, fwt, spyt):
    # recuperar os thetas
    pesos = RecuperarPesos()
    theta1 = np.reshape(pesos[:hidden_camada_tamanho*(input_camada_tamanho+1)], (hidden_camada_tamanho, input_camada_tamanho+1))
    theta2 = np.reshape(pesos[hidden_camada_tamanho*(input_camada_tamanho+1):], (1, hidden_camada_tamanho+1))

    # for. propagation
    a1 = np.concatenate((np.ones((m, 1)), X), axis=1)
    z2 = a1.dot(theta1.T)
    a2 = sp.expit(z2)   # sigmoid
    a2 = np.concatenate((np.ones((m, 1)), a2), axis=1)
    z3 = a2.dot(theta2.T)
    h = sp.expit(z3)

    # aplicar a fronteira
    previsao = (h >= fronteira).astype(int)
    return previsao, h

def GerarParametros(output=False):
    # popular os thetas com valores iniciais aleatórios: theta1(hidden_camada_tamanho x input_camada_tamanho+1), theta2(1 x hidden_camada_tamanho+1)
    # inicializamos os pesos entre -margem e +margem
    margem = 0.12
    theta1 = (np.random.rand(hidden_camada_tamanho, input_camada_tamanho+1) * (2 * margem)) - margem
    theta2 = (np.random.rand(1, hidden_camada_tamanho+1) * (2 * margem)) - margem
    # Otimização precisa de um vetor
    nn_params = np.array([np.concatenate((theta1,theta2), axis=None)]).T
    # print('Thetas formados:', theta1, theta2)
    # print(nn_params)

    # Otimização
    otimizacao = otimizar(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, max)
    # print(otimizacao)
    if output:
        (J, _) = funcaoCusto(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)
        (Jo, _) = funcaoCusto(otimizacao, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)
        print('Custo inicial:', J, 'Custo final:', Jo)
    
    # registrar pesos novos
    with open('pesos.txt', 'w') as f:
        otimizacao.tofile(f, sep=",")
    
def GradientChecking():
    pesos = RecuperarPesos()
    (_, grad) = funcaoCusto(pesos, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)

    # Verificação de gradientes iniciais:
    D1 = np.reshape(grad[:hidden_camada_tamanho*(input_camada_tamanho+1)], (hidden_camada_tamanho, input_camada_tamanho+1))
    D2 = np.reshape(grad[hidden_camada_tamanho*(input_camada_tamanho+1):], (1, hidden_camada_tamanho+1))

    # gradient checking
    numgrad = gradientesNumericos(lambda t : funcaoCusto(t, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)[0], pesos)
    numericalD1 = np.reshape(numgrad[:hidden_camada_tamanho*(input_camada_tamanho+1)], (hidden_camada_tamanho, input_camada_tamanho+1))
    numericalD2 = np.reshape(numgrad[hidden_camada_tamanho*(input_camada_tamanho+1):], (1, hidden_camada_tamanho+1))
    # print('Gradientes numéricos:', numericalD1, numericalD2)
    print('Gradient check: média dos erros:', np.mean(np.concatenate((D1-numericalD1, D2-numericalD2), axis=None)))

# retorna uma tupla
# 1 - vetor de custos por grau (grau, custo em treino, custo em cross validation)
# 2 - vetor de custos por lambda (grau, custo em treino, custo em cross validation)
# 3 - grau otimo encontrado
# 4 - lambda otimo encontrado
def AnaliseCustos(n_graus = (1, 5), l_set = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]):
    (custos_grau, custos_lambda) = AnaliseDeCombinacaoELambda(hidden_camada_tamanho, np.concatenate((embed1, embed2), axis=1), fw, spy, y, lmbd, fracoes, max, n_graus, l_set)

    # selecionar o grau e lambda mais promissores
    min_p_i = np.argmin(custos_grau, axis=0)[2]
    grau_o = int(custos_grau[min_p_i, 0])
    min_l_i = np.argmin(custos_lambda, axis=0)[2]
    lmbd_o = custos_lambda[min_l_i, 0]
    return custos_grau, custos_lambda, grau_o, lmbd_o

# gera uma matriz n x 3
# as linhas correspondem a um valor de tamanho de conjunto de exemplos utilizado na análise
# as colunas correspondem a: o tamanho do conjunto utilizado, o custo em treinamento, o custo em cross validation
def AnaliseCurvaAprendizado(n = 20):
    return CurvaDeAprendizado(n, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, fracoes, max)

# retorna uma tupla:
# exatidão (accuracy), precisão (precision), revocação (recall) e medida F (F1, F measure)
def AnaliseDesempenho():
    return AnalisarDesempenho(RecuperarPesos(), input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, fracoes, fronteira)