import numpy as np
from scipy import special as sp

def gerarInputAleatorio(m, elements):
    ret = np.ones((1, elements*2+4))
    for i in range(m):
        # fazer 2 vetores de elements floats aleatorios
        embed1 = np.random.rand(1, elements) * 100
        embed2 = np.random.rand(1, elements) * 100

        # simular retornos do fuzzywuzzy e spacy
        fw = np.random.randint(100)
        spy = np.random.rand() * 10
        fw2 = fw**2
        spy2 = spy**2
        case = np.concatenate((embed1, embed2, np.array([[fw]]), np.array([[spy]]), np.array([[fw2]]), np.array([[spy2]])), axis=1)
        ret = np.concatenate((ret, case), axis=0)
    return np.delete(ret, 0, 0)

def funcaoCusto(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd): # lmbd -> lambda
    # adquirir m
    m = X.shape[0]
    # ajustar X
    a1 = np.concatenate((np.ones((m, 1)), X), axis=1)
    # recuperar os thetas
    theta1 = np.reshape(nn_params[:hidden_camada_tamanho*(input_camada_tamanho+1)], (hidden_camada_tamanho, input_camada_tamanho+1))
    theta2 = np.reshape(nn_params[hidden_camada_tamanho*(input_camada_tamanho+1):], (1, hidden_camada_tamanho+1))
    print(theta1[0:2], theta2[0:2])

    # for. propagation
    z2 = a1.dot(theta1.T)
    a2 = sp.expit(z2)   # sigmoid
    a2 = np.concatenate((np.ones((m, 1)), a2), axis=1)

    z3 = a2.dot(theta2.T)
    h = sp.expit(z3)

    # função custo
    # remove os pesos constantes (primeira coluna), pois eles não são penalizados na regularização
    theta1_i = theta1[:, 1:]
    theta2_i = theta2[:, 1:]

    J = (np.sum(-y * (np.log(h)) - (1 - y) * (np.log(1 - h))) + (lmbd/2) * (np.sum(theta1_i*theta1_i) + np.sum(theta2_i*theta2_i))) / m
    return J

def normalizar(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X, axis=0)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma

# MODELO
#   Rede neural em 3 camadas: 604 - 200 - 1. 
m = 10
input_camada_tamanho = 10
hidden_camada_tamanho = 2

# obter os exemplos de entrada
X = gerarInputAleatorio(m, (input_camada_tamanho-4)//2)
# print('Input bruto:', X[0:2])
# normalizar as entradas
X = normalizar(X)[0]
# print('Input normalizado:', X[0:2])

# obter os resultados de saída
y = np.random.random(m).round()
# print(y)

# popular os thetas: theta1(200x605), theta2(1x201)
# inicializamos os pesos entre -0.12 e 0.12
theta1 = (np.random.rand(hidden_camada_tamanho, input_camada_tamanho+1) * (2 * 0.12)) - 0.12
theta2 = (np.random.rand(1, hidden_camada_tamanho+1) * (2 * 0.12)) - 0.12
print(theta1[0:2], theta2[0:2])

# # função custo
nn_params = np.concatenate((theta1,theta2), axis=None)
# print(nn_params)
J = funcaoCusto(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, 0.5)
print(J)

# gradientes

# gradient checking

# otimização