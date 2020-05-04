import numpy as np
from scipy import special as sp
from scipy import optimize as opt

def obterCombinacoes(a, b, n):
    ret = np.array([[]])

    # print(type(a))
    if type(a) != np.ndarray:
        combinar = lambda a,b,i,m : np.array([[a**i * b**(m-i)]])
    else:
        combinar = lambda a,b,i,m : a**i * b**(m-i)
    
    for m in range(1, n+1):
        for i in range(m+1):
            aux = combinar(a, b, i, m)
            if m==1 and i==0:
                ret = aux
            else:
                # print(ret.shape, aux.shape)
                ret = np.concatenate((ret, aux), axis=1)
    return ret

def gerarInputAleatorio(m, elements, grau):
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

# retorna 3 matrizes de exemplos: os exemplos de treino, de cross validation, e de teste
# fractions é uma tupla que determina em quais fracoes deve ser dividido os exemplos (exmplo: (6, 2, 2))
def dividirExemplos(X, fractions):
    total = sum(fractions)
    teste_frac = int(X.shape[0]*fractions[2]/total)
    cv_frac = int(X.shape[0]*fractions[1]/total)
    # print('Total, teste_frac e cv_frac', total, teste_frac, cv_frac)

    X_teste = X[ : teste_frac]
    X_cv = X[teste_frac : teste_frac+cv_frac]
    X_treino = X[teste_frac+cv_frac : ]

    return X_treino, X_cv, X_teste


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
    # if m == 0:
    #     print('OLHA O M:',m)

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
    e = 1e-7
    numgrad = np.zeros((n, 1))
    perturbacao = np.zeros((n, 1))
    for i in range(n):
        if i>0 and i in [n//5, 2*n//5, 3*n//5, 4*n//5]:
            print('Progresso do calculo de gradientes numericos:', 100*i//n+1, 'de 100')
        # print(i)
        perturbacao[i] = e
        # print(theta.shape, perturbacao.shape)
        perda1 = J(theta - perturbacao)
        perda2 = J(theta + perturbacao)
        numgrad[i] = (perda2 - perda1) / (2 * e)
        perturbacao[i] = 0
    return numgrad

def otimizar(theta, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, max):
    ret = opt.fmin_cg(lambda x : funcaoCusto(x, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)[0], theta, fprime=(lambda x : funcaoCusto(x, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)[1]), maxiter=max)
    return ret.reshape(ret.size, 1)

# retorna 2 vetores: o primeiro mede a variacao do custo conforme se aumenta o grau das combinações, o segundo mede conforme se aumenta lambda
# cada linha do vetor representa o resultado do experimento com um valor diferente de grau/lambda, e as colunas são:
#   a primeira é o valor experimentado de grau/lambda, a segunda é o custo observado no conjunto de exemplos de treino e a terceira o custo observado no conjunto de exemplos de CV
def AnaliseDeCombinacaoELambda(hidden_camada_tamanho, embeds, fw, spy, y, lmbd, fractions, max):
    # separa os y's
    (y, y_cv, _) = dividirExemplos(y, (6, 2, 2))
    
    # interface amigavel para o custo
    # lambda 0 pq é para avaliar o custo final, e não para ser utilizado na otimização
    J = lambda theta, size, X, yl : funcaoCusto(theta, size, hidden_camada_tamanho, X, yl, 0)[0]
    
    # custos por grau
    print('Calculando custos por grau...')
    for p in range(1, 6):
        # ajustar entrada
        X = np.concatenate((embeds, obterCombinacoes(fw, spy, p)), axis=1)
        # print('Tamanho total de X:', X.shape)
        (X, X_cv, _) = dividirExemplos(X, fractions)
        # print('Tamanho de X final e de X_cv:', X.shape, X_cv.shape)
        input_camada_tamanho = embeds.shape[1] + 2*p + sum(range(p))
        # print('tamanho de X e da camada de input:', X.shape, input_camada_tamanho)

        # pesos iniciais
        theta1 = (np.random.rand(hidden_camada_tamanho, input_camada_tamanho+1) * (2 * 0.12)) - 0.12
        theta2 = (np.random.rand(1, hidden_camada_tamanho+1) * (2 * 0.12)) - 0.12
        nn_params = np.array([np.concatenate((theta1,theta2), axis=None)]).T
        # print('Tamanhos de theta1, theta2 e nn_params:', theta1.shape, theta2.shape, nn_params.shape)

        # otimizar
        theta = otimizar(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, max)
        # print('Tamanho de theta final e seu tipo:', theta.shape, type(theta))
        # salvar os custos
        if p == 1:
            custos_por_grau = np.array([[p, J(theta, input_camada_tamanho, X, y), J(theta, input_camada_tamanho, X_cv, y_cv)]])
        else:
            custos_por_grau = np.concatenate((custos_por_grau, np.array([[p, J(theta, input_camada_tamanho, X, y), J(theta, input_camada_tamanho, X_cv, y_cv)]])), axis=0)

    # custos por lambda    
    print('Calculando custos por lambda...')
    # ajustar X
    # ja pega o melhor resultado de p do teste anterior
    # print(custos_por_grau)
    min_p_i = np.argmin(custos_por_grau, axis=0)[2]
    # print(np.argmin(custos_por_grau, axis=0), min_p_i, type(min_p_i))
    min_p = int(custos_por_grau[min_p_i, 0])
    # print(min_p)
    X = np.concatenate((embeds, obterCombinacoes(fw, spy, min_p)), axis=1)
    (X, X_cv, _) = dividirExemplos(X, fractions)
    input_camada_tamanho = embeds.shape[1] + 2*min_p + sum(range(min_p))

    # pesos iniciais
    theta1 = (np.random.rand(hidden_camada_tamanho, input_camada_tamanho+1) * (2 * 0.12)) - 0.12
    theta2 = (np.random.rand(1, hidden_camada_tamanho+1) * (2 * 0.12)) - 0.12
    nn_params = np.array([np.concatenate((theta1,theta2), axis=None)]).T

    # interface amigavel para o custo
    # lambda 0 pq é para avaliar o custo final, e não para ser utilizado na otimização
    J = lambda theta, x, yl : funcaoCusto(theta, input_camada_tamanho, hidden_camada_tamanho, x, yl, 0)[0]

    l_set = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for l in l_set:
        # otimizar
        # print(nn_params[0:2], input_camada_tamanho, hidden_camada_tamanho, X[0:2], y[0:2], l, sep='\n')
        theta = otimizar(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, l, max)
        # salvar os custos
        if l == l_set[0]:
            custos_por_lambda = np.array([[l, J(theta, X, y), J(theta, X_cv, y_cv)]])
        else:
            custos_por_lambda = np.concatenate((custos_por_lambda, np.array([[l, J(theta, X, y), J(theta, X_cv, y_cv)]])), axis=0)
    print('Grau selecionado:', min_p)
    return custos_por_grau, custos_por_lambda
# gera uma matriz n x 3
# as linhas correspondem a um valor de tamanho de conjunto de exemplos utilizado na análise
# as colunas correspondem a: o tamanho do conjunto utilizado, o custo em treinamento, o custo em cross validation
def CurvaDeAprendizado(n, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, fractions, max):
    (X, X_cv, _) = dividirExemplos(X, fractions)
    (y, y_cv, _) = dividirExemplos(y, fractions)
    # obter m
    m = X.shape[0]

    # impede o usuário de pedir mais iterações do que existem exemplos
    if n > m:
        n = m
        print('Aviso: A quantidade de iterações solicitada é maior do que a quantidade de exemplos de treinamento obtida. Serão realizadas apenas', n, 'iterações.')

    # interface amigável para o custo
    # lambda 0 pq é para avaliar o custo final, e não para ser utilizado na otimização
    J = lambda theta, Xj, yj : funcaoCusto(theta, input_camada_tamanho, hidden_camada_tamanho, Xj, yj, 0)[0]

    for k in range(1, n+1):
        qtd_exemplos = m * k//n
        # linhas de exemplo a serem selecionadas para essa iteração
        linhas = np.random.permutation(m)[0 : qtd_exemplos]
        X_it = X[linhas]
        y_it = y[linhas]

        # pesos iniciais
        theta1 = (np.random.rand(hidden_camada_tamanho, input_camada_tamanho+1) * (2 * 0.12)) - 0.12
        theta2 = (np.random.rand(1, hidden_camada_tamanho+1) * (2 * 0.12)) - 0.12
        nn_params = np.array([np.concatenate((theta1,theta2), axis=None)]).T

        # otimizar
        theta = otimizar(nn_params, input_camada_tamanho, hidden_camada_tamanho, X_it, y_it, lmbd, max)
        # salvar os custos
        if k == 1:
            ret = np.array([[qtd_exemplos, J(theta, X_it, y_it), J(theta, X_cv, y_cv)]])
        else:
            ret = np.concatenate((ret, np.array([[qtd_exemplos, J(theta, X_it, y_it), J(theta, X_cv, y_cv)]])), axis=0)
    return ret

# retorna uma tupla:
# exatidão (accuracy), precisão (precision), revocação (recall) e medida F (F1, F measure)
# fronteira é um numero entre 0 e 1 tal que se h >= fronteira, h pode ser considerado uma previsão positiva (1)
def AnalisarDesempenho(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd, fractions, fronteira):
    # obter os exemplos de teste
    (_, _, X) = dividirExemplos(X, fractions)
    (_, _, y) = dividirExemplos(y, fractions)
    # custo
    J = funcaoCusto(nn_params, input_camada_tamanho, hidden_camada_tamanho, X, y, lmbd)[0]

    # previsão
    # recuperar os thetas
    # print(nn_params.shape)
    theta1 = np.reshape(nn_params[:hidden_camada_tamanho*(input_camada_tamanho+1)], (hidden_camada_tamanho, input_camada_tamanho+1))
    theta2 = np.reshape(nn_params[hidden_camada_tamanho*(input_camada_tamanho+1):], (1, hidden_camada_tamanho+1))
    # print('Thetas recuperados:', theta1[0:2], theta2[0:2])

    # for. propagation
    m = X.shape[0]
    a1 = np.concatenate((np.ones((m, 1)), X), axis=1)
    z2 = a1.dot(theta1.T)
    a2 = sp.expit(z2)   # sigmoid
    a2 = np.concatenate((np.ones((m, 1)), a2), axis=1)

    z3 = a2.dot(theta2.T)
    h = sp.expit(z3)

    # aplicar a fronteira
    h = h >= fronteira

    # relações:
    resultado = h - y + (h * y * 2)
    # estrutura:
    # h y - resultado
    # 0 1 -   -1
    # 1 0 -    1
    # 1 1 -    2
    # 0 0 -    0
    # qtd de -1: falsos negativos
    # qtd de  1: falsos positivos
    # qtd de  2: verdadeiros positivos
    # qtd de  0: verdadeiros falsos
    f_n = np.count_nonzero(resultado == -1)
    f_p = np.count_nonzero(resultado == 1)
    v_p = np.count_nonzero(resultado == 2)
    v_n = np.count_nonzero(resultado == 0)
    
    print(h, y, resultado, sep='\n------\n')
    
    # exatidão
    exatidao = (v_p + v_n)/resultado.size
    # precisão e revocação
    precisao = v_p / (v_p + f_p)
    revocacao = v_p / (v_p + f_n)
    # medida F
    F1 = 2*revocacao*precisao / (revocacao + precisao)
    return exatidao, precisao, revocacao, F1

def RecuperarPesos():
    with open('pesos.txt', 'r') as f:
        inp = f.read()
    return np.array(np.mat(inp)).T