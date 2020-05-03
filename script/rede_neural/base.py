import numpy as np

def gerarInputAleatorio(m):
    ret = np.ones((1, 605))
    for i in range(m):
        # fazer 2 vetores de 300 floats aleatorios
        embed1 = np.random.rand(1, 300) * 100
        embed2 = np.random.rand(1, 300) * 100

        # simular retornos do fuzzywuzzy e spacy
        fw = np.random.randint(100)
        spy = np.random.rand() * 10
        fw2 = fw**2
        spy2 = spy**2
        case = np.concatenate((np.array([[1]]), embed1, embed2, np.array([[fw]]), np.array([[spy]]), np.array([[fw2]]), np.array([[spy2]])), axis=1)
        ret = np.concatenate((ret, case), axis=0)
    return np.delete(ret, 0, 0)


# MODELO
#   Rede neural em 3 camadas: 604 - 200 - 1. 
a1 = gerarInputAleatorio(10)
# popular os thetas: theta1(200x605), theta2(1x201)
# inicializamos os pesos entre -0.12 e 0.12
theta1 = (np.random.rand(200, 605) * (2 * 0.12)) - 0.12
theta2 = (np.random.rand(1, 201) * (2 * 0.12)) - 0.12

# for. propagation

# função custo

# bac. propagation

# gradientes