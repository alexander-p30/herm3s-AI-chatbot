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
for i in range(m):
    polinomio = obterCombinacoes(fw, spy, grau)

    ex = np.concatenate((embed1, embed2, polinomio), axis=1)
    if i==0:
        X = ex
    else:
        X = np.concatenate((X, ex), axis=0)

# print('Tamanho de X:', X.shape, 'Tamanho de y:', y.shape)
# normalizar as entradas (desnecessário para o método de otimização atual)
# X = normalizar(X)[0]

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
    
# O RESTO ESTA EM DESENVOLVIMENTO
    
# def GradientChecking()
    


# # Verificação de gradientes iniciais:
# # D1 = np.reshape(grad[:hidden_camada_tamanho*(input_camada_tamanho+1)], (hidden_camada_tamanho, input_camada_tamanho+1))
# # D2 = np.reshape(grad[hidden_camada_tamanho*(input_camada_tamanho+1):], (1, hidden_camada_tamanho+1))
# # print('Gradientes:', D1, D2)

# # gradient checking
# numgrad = gradientesNumericos(lambda t : funcaoCusto(t, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)[0], nn_params)
# numericalD1 = np.reshape(numgrad[:hidden_camada_tamanho*(input_camada_tamanho+1)], (hidden_camada_tamanho, input_camada_tamanho+1))
# numericalD2 = np.reshape(numgrad[hidden_camada_tamanho*(input_camada_tamanho+1):], (1, hidden_camada_tamanho+1))
# # print('Gradientes numéricos:', numericalD1, numericalD2)
# print('Gradient check', np.mean(np.concatenate((D1-numericalD1, D2-numericalD2), axis=None)))


# input('Pressione enter para continuar')

# # análise de custos por grau e lambda
# pos = elementos_embed*2
# # print('fwd e spy:', X[:, pos+1:pos+2], X[:, pos:pos+1])
# (custos_grau, custos_lambda) = AnaliseDeCombinacaoELambda(hidden_camada_tamanho, X[:, 0:pos], X[:, pos+1:pos+2], X[:, pos:pos+1], y, lmbd, fracoes, max)
# print('Custos por grau:\n', custos_grau, '\nCustos por lambda:\n', custos_lambda)

# # selecionar o grau e lambda mais promissores
# min_p_i = np.argmin(custos_grau, axis=0)[2]
# grau = int(custos_grau[min_p_i, 0])
# min_l_i = np.argmin(custos_lambda, axis=0)[2]
# lmbd = custos_lambda[min_l_i, 0]
# input_camada_tamanho = 2 * elementos_embed + 2*grau + sum(range(grau))
# X = gerarInputAleatorio(m, elementos_embed, grau)
# print('Grau e lambda otimos:', grau, lmbd)

# theta1 = (np.random.rand(hidden_camada_tamanho, input_camada_tamanho+1) * (2 * 0.12)) - 0.12
# theta2 = (np.random.rand(1, hidden_camada_tamanho+1) * (2 * 0.12)) - 0.12
# # print('Thetas formados:', theta1, theta2)

# # # função custo e gradientes
# nn_params = np.array([np.concatenate((theta1,theta2), axis=None)]).T
# nn_params = otimizar(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, max)

# input('Pressione enter para continuar')
# # análise de curva de aprendizado
# curva_aprendizado = CurvaDeAprendizado(20, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, fracoes, max)
# print('Resultados da curva de aprendizado:\n', curva_aprendizado)

# # análise de desempenho
# desempenho = AnalisarDesempenho(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, fracoes, fronteira)
# print('Exatidão:', desempenho[0], 'Precisão:', desempenho[1], 'Revocação:', desempenho[2], 'Medida F:', desempenho[3])