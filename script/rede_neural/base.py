import numpy as np
from scipy import special as sp
from scipy import optimize as opt

def obterCombinacoes(a, b, n):
    ret = np.array([[]])
    for m in (np.array(range(n))+1):
        for i in range(m+1):
            ret = np.concatenate((ret, [[a**i * b**(m-i)]]), axis=1)
    return ret

def gerarInputAleatorio(m, elements, grau):
    ret = [[None]]
    for i in range(m):
        # fazer 2 vetores de elements floats aleatorios
        embed1 = np.random.rand(1, elements) * 100
        embed2 = np.random.rand(1, elements) * 100

        # simular retornos do fuzzywuzzy e spacy
        fw = np.random.randint(100)
        spy = np.random.rand() * 10
        estimacoes = obterCombinacoes(fw, spy, grau)

        case = np.concatenate((embed1, embed2, estimacoes), axis=1)
        if i==0:
            ret = case
        else:
            ret = np.concatenate((ret, case), axis=0)
    return ret

# retorna o custo e um vetor com os gradientes de cada parametro recebido
def funcaoCusto(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd): # lmbd -> lambda
    # adquirir m
    m = X.shape[0]
    # ajustar X
    a1 = np.concatenate((np.ones((m, 1)), X), axis=1)
    # recuperar os thetas
    # print(nn_params.shape)
    theta1 = np.reshape(nn_params[:hidden_camada_tamanho*(input_camada_tamanho+1)], (hidden_camada_tamanho, input_camada_tamanho+1))
    theta2 = np.reshape(nn_params[hidden_camada_tamanho*(input_camada_tamanho+1):], (1, hidden_camada_tamanho+1))
    # print('Thetas recuperados:', theta1[0:2], theta2[0:2])

    # for. propagation
    z2 = a1.dot(theta1.T)
    a2 = sp.expit(z2)   # sigmoid
    a2 = np.concatenate((np.ones((m, 1)), a2), axis=1)

    z3 = a2.dot(theta2.T)
    h = sp.expit(z3)
    # print('Tamanho de h e y:', np.shape(h), np.shape(y))

    # função custo
    # remove os pesos constantes (primeira coluna), pois eles não são penalizados na regularização
    theta1_i = theta1[:, 1:]
    theta2_i = theta2[:, 1:]

    J = (np.sum(-y * (np.log(h)) - (1 - y) * (np.log(1 - h))) + (lmbd/2) * (np.sum(theta1_i*theta1_i) + np.sum(theta2_i*theta2_i))) / m

    # bac. propagation
    d3 = h - y
    aux = sp.expit(z2)
    d2 = d3.dot(theta2_i) * (aux * (1 - aux))   # multiplica pelo gradiente do sigmoid

    # regularização dos gradientes
    r1 = lmbd * np.concatenate((np.zeros((theta1.shape[0], 1)), theta1[:, 1:]), axis=1)
    r2 = lmbd * np.concatenate((np.zeros((theta2.shape[0], 1)), theta2[:, 1:]), axis=1)

    # gradientes
    D1 = (d2.T.dot(a1) + r1) / m
    D2 = (d3.T.dot(a2) + r2) / m

    # vetorizar os gradientes
    grad = np.concatenate((D1, D2), axis=None)
    return J, grad

def normalizar(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X, axis=0)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma

def gradientesNumericos(J, theta):
    n = theta.size
    e = 1e-4
    numgrad = np.zeros((n, 1))
    perturbacao = np.zeros((n, 1))
    for i in range(n):
        # print(i)
        perturbacao[i] = e
        # print(theta.shape, perturbacao.shape)
        perda1 = J(theta - perturbacao)
        perda2 = J(theta + perturbacao)
        numgrad[i] = (perda2 - perda1) / (2 * e)
        perturbacao[i] = 0
    return numgrad

def otimizar(theta, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd):
    return opt.fmin_cg(lambda x : funcaoCusto(x, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)[0], theta, fprime=(lambda x : funcaoCusto(x, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)[1]), maxiter=200)
