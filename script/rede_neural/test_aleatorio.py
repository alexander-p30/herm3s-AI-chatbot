from base import *
# MODELO
#   Rede neural em 3 camadas: 604 - 200 - 1. 
m = 200
grau = 2
elementos_embed = 10
hidden_camada_tamanho = 20
lmbd = 0.5
max = 150
# define as fronteiras entre previsão 1 e 0
fronteira = 0.5
# proporções dos conjuntos de exemplos (treino, cv, teste)
fracoes = (6, 2, 2)

input_camada_tamanho = 2 * elementos_embed + 2*grau + sum(range(grau))

# obter os exemplos de entrada
X = gerarInputAleatorio(m, elementos_embed, grau)
print('Tamanho de X:', X.shape)
# print('Input bruto:', X[0:2])
# normalizar as entradas
# X = normalizar(X)[0]
# print('Input normalizado:', X[0:2])

# obter os resultados de saída
y = np.random.random((m, 1)).round()
# print(np.shape(y))

# popular os thetas: theta1(200x605), theta2(1x201)
# inicializamos os pesos entre -0.12 e 0.12
theta1 = (np.random.rand(hidden_camada_tamanho, input_camada_tamanho+1) * (2 * 0.12)) - 0.12
theta2 = (np.random.rand(1, hidden_camada_tamanho+1) * (2 * 0.12)) - 0.12
# print('Thetas formados:', theta1, theta2)

# # função custo e gradientes
nn_params = np.array([np.concatenate((theta1,theta2), axis=None)]).T
# print(nn_params)
(J, grad) = funcaoCusto(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)
print('Custo inicial:', J)
D1 = np.reshape(grad[:hidden_camada_tamanho*(input_camada_tamanho+1)], (hidden_camada_tamanho, input_camada_tamanho+1))
D2 = np.reshape(grad[hidden_camada_tamanho*(input_camada_tamanho+1):], (1, hidden_camada_tamanho+1))
# print('Gradientes:', D1, D2)

# gradient checking
numgrad = gradientesNumericos(lambda t : funcaoCusto(t, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)[0], nn_params)
numericalD1 = np.reshape(numgrad[:hidden_camada_tamanho*(input_camada_tamanho+1)], (hidden_camada_tamanho, input_camada_tamanho+1))
numericalD2 = np.reshape(numgrad[hidden_camada_tamanho*(input_camada_tamanho+1):], (1, hidden_camada_tamanho+1))
# print('Gradientes numéricos:', numericalD1, numericalD2)
print('Gradient check', np.mean(np.concatenate((D1-numericalD1, D2-numericalD2), axis=None)))

# otimização
otimizacao = otimizar(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, max)
# print(otimizacao)

input('Pressione enter para continuar')

# análise de custos por grau e lambda
pos = elementos_embed*2
# print('fwd e spy:', X[:, pos+1:pos+2], X[:, pos:pos+1])
(custos_grau, custos_lambda) = AnaliseDeCombinacaoELambda(hidden_camada_tamanho, X[:, 0:pos], X[:, pos+1:pos+2], X[:, pos:pos+1], y, lmbd, fracoes, max)
print('Custos por grau:\n', custos_grau, '\nCustos por lambda:\n', custos_lambda)

# selecionar o grau e lambda mais promissores
min_p_i = np.argmin(custos_grau, axis=0)[2]
grau = int(custos_grau[min_p_i, 0])
min_l_i = np.argmin(custos_lambda, axis=0)[2]
lmbd = custos_lambda[min_l_i, 0]
input_camada_tamanho = 2 * elementos_embed + 2*grau + sum(range(grau))
X = gerarInputAleatorio(m, elementos_embed, grau)
print('Grau e lambda otimos:', grau, lmbd)

theta1 = (np.random.rand(hidden_camada_tamanho, input_camada_tamanho+1) * (2 * 0.12)) - 0.12
theta2 = (np.random.rand(1, hidden_camada_tamanho+1) * (2 * 0.12)) - 0.12
# print('Thetas formados:', theta1, theta2)

# # função custo e gradientes
nn_params = np.array([np.concatenate((theta1,theta2), axis=None)]).T
nn_params = otimizar(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, max)

input('Pressione enter para continuar')
# análise de curva de aprendizado
curva_aprendizado = CurvaDeAprendizado(20, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, fracoes, max)
print('Resultados da curva de aprendizado:\n', curva_aprendizado)

# análise de desempenho
desempenho = AnalisarDesempenho(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, fracoes, fronteira)
print('Exatidão:', desempenho[0], 'Precisão:', desempenho[1], 'Revocação:', desempenho[2], 'Medida F:', desempenho[3])