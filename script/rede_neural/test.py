from base import *
# MODELO
#   Rede neural em 3 camadas: 604 - 200 - 1. 
m = 10
grau = 2
elementos_embed = 3
input_camada_tamanho = 2 * elementos_embed + 2*grau + sum(range(grau))
hidden_camada_tamanho = 2
lmbd = 5

# obter os exemplos de entrada
X = gerarInputAleatorio(m, elementos_embed, grau)
print(X.shape)
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
print('Thetas formados:', theta1, theta2)

# # função custo e gradientes
nn_params = np.array([np.concatenate((theta1,theta2), axis=None)]).T
# print(nn_params)
(J, grad) = funcaoCusto(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)
print('Custo:', J)
D1 = np.reshape(grad[:hidden_camada_tamanho*(input_camada_tamanho+1)], (hidden_camada_tamanho, input_camada_tamanho+1))
D2 = np.reshape(grad[hidden_camada_tamanho*(input_camada_tamanho+1):], (1, hidden_camada_tamanho+1))
print('Gradientes:', D1, D2)

# gradient checking
numgrad = gradientesNumericos(lambda t : funcaoCusto(t, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)[0], nn_params)
numericalD1 = np.reshape(numgrad[:hidden_camada_tamanho*(input_camada_tamanho+1)], (hidden_camada_tamanho, input_camada_tamanho+1))
numericalD2 = np.reshape(numgrad[hidden_camada_tamanho*(input_camada_tamanho+1):], (1, hidden_camada_tamanho+1))
print('Gradientes numéricos:', numericalD1, numericalD2)

# otimização
otimizacao = otimizar(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)
print(otimizacao)