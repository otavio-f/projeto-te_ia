"""
Classificadores
:author: Otávio Ferreira
"""

import math
from dataclasses import dataclass
from dataclasses import field
import numpy as np
from typing import Callable, Any, Iterable, List


def augment(v: np.ndarray, aug: int|float) -> np.ndarray:
    """
    Aumenta um vetor.
    :param aug: Argumento que irá aumentar o vetor
    :returns: O vetor com o elemento aumentado ao final.
    """
    result = np.concat((v, [aug,]))
    return result


def _get_term(i: int, xk: float, is_indep=False) -> str:
    """
    Retorna uma representação do termo na equação. Ex.: x1, -2x2, x3
    :param i: índice do termo
    :param xk: multiplicador do termo
    :param is_indep: se é um termo independente (sem x)
    """
    if xk == 0:
        return ""
    
    if xk < 0:
        sign = "-"
    elif i==0:
        sign = ""
    else:
        sign = "+"
    
    if abs(xk) == 1 and not is_indep:
        mul = ""
    elif abs(xk - math.floor(xk)) == 0:
        mul = f"{abs(math.floor(xk))}"
    else:
        mul = f"{abs(xk)}"

    if is_indep:
        term = ""
    else:
        term = f"x{i+1}"
    
    return f"{sign}{mul}{term}"


def euclidean_dist(m: np.ndarray) -> [Callable[[float, ...], float], str]:
    """
    Calcula a equação de distância euclideana.
    :param m: O vetor característica.
    :returns: A equação de distância euclideana e uma representação legível dessa equação.
    """
    eq = '√('
    
    # x1² + x2² + ... xn²
    eq += "+".join(f"x{i+1}²" for i, _ in enumerate(m))
    
    # - 2*m1x1 - 2*m2x2 - ... - 2*mnxn
    for i, mi in enumerate(m):
        if mi == 0:
            continue
        mul = -2 * mi
        sign = "-" if mul<0 else "+"
        term = f"x{i+1}" if abs(mul) == 1 else f"{abs(mul)}x{i+1}"
        eq += f"{sign}{term}"
    
    # m1² + m2² + ... + mn²
    eq += f"+{sum(mi*mi for mi in m)})"
    # func = lambda x: math.sqrt((x - m).T * (x - m))
    func = lambda x: math.sqrt((x - m).T.dot(x - m))
    return func, eq


def max_dist(m: np.ndarray) -> [Callable[[float, ...], float], str]:
    """
    Calcula a equação de classificador máximo.
    :param m: O vetor característica.
    :returns: A equação de classificador máximo e uma representação legível dessa equação.
    """
    eq = "".join(_get_term(i, mi) for i,mi in enumerate(m))

    # extra = -1/2 * (m.T * m)
    extra = -1/2 * m.T.dot(m)
    if extra != 0:
        sign = "-" if extra < 0 else '+'
        eq += f'{sign}{abs(extra)}'
    
    # func = lambda x: (x.T * m) - (1/2 * (m.T * m))
    func = lambda x: m.T.dot(x) - (1/2 * (m.dot(m)))
    return func, eq


def dij(mi: np.ndarray, mj: np.ndarray) -> [Callable[[float, ...], float], str]:
    """
    Calcula a equação de superfície de decisão sobre duas classes
    :param mi: Vetor característica da classe i
    :param mj: Vetor característica da classe j
    :return: A equação de superfície de decisão e uma representação legível dessa equação.
    """
    eq = "".join(_get_term(i, di) for i,di in enumerate(mi-mj))
    
    # extra = -1/2 * ((mi - mj).T * (mi + mj))
    extra = -1/2 * (mi - mj).T.dot(mi + mj)
    if extra != 0:
        sign = '-' if extra < 0 else '+'
        eq += f'{sign}{abs(extra)}'
    
    # func = lambda x: ((mi - mj).T * x) - 1/2 * ((mi - mj).T * (mi + mj))
    func = lambda x: (mi - mj).T.dot(x) - 1/2 * (mi - mj).T.dot(mi + mj)
    return func, eq


def perceptron(c1v: [np.ndarray, ...], c2v: [np.ndarray, ...], max_iters=10_000, c=1) -> [Callable[[float, ...], float], str, int]:
    """
    Calcula a equação sobre duas classes pelo método perceptron
    :param c1v: Conjunto de vetores de treinamento da classe 1
    :param c2v: Conjunto de vetores de treinamento da classe 2
    :param max_iters: O número máximo de iterações do algoritmo
    :param c: O fator de randomização, recomendado usar valores entre 0 a 1
    :return: A equação de superfície, representação legível dessa equação e a quantidade de iterações necessárias para chegar nessa iteração.
    """
    # inicializa pesos com vetor de zeros, aumentado de zero [0, 0, ..., 0]
    w = np.zeros(len(c1v[0]) + 1)
    
    # alterna c1, c2, c1, c2
    # aumenta vetores (..., 1)
    train_vectors = [augment(v, 1) for pair in zip(c1v, c2v) for v in pair]

    # condição de parada: número máximo de iterações ou passos sem erro
    # 5 passos sem mudança dos pesos (w) para o algoritmo
    min_steps = 5
    iters = 0
    clean_steps = 0

    # iteração
    while iters < max_iters and clean_steps < min_steps:
        for i, xk in enumerate(train_vectors):
            iters += 1
            # test = w * xk.T
            test = w.dot(xk.T)
            test_ok = (test > 0) if i%2==0 else (test < 0) # c1 (par) >0, c2 (ímpar) <0
            if not test_ok:
                clean_steps = 0
                if i%2 == 0: # xk pertence a c1
                    w = w + (c * xk)
                else: # xk pertence a c2
                    w = w - (c * xk)
            else:
                clean_steps += 1
            if clean_steps > min_steps and iters > len(train_vectors):
                break

    eq = ""
    for i, wi in enumerate(w):
        eq += _get_term(i, wi, i==len(w)-1)
    

    func = lambda x: w.dot(augment(x, 1))
    # func = lambda x: w * augment(x, 1)
    return func, eq, iters


def delta_perceptron(c1v: [np.ndarray, ...], c2v: [np.ndarray, ...], max_iters=10_000, alpha=1) -> [Callable[[float, ...], float], str, int]:
    """
    Calcula a equação perceptron com regra delta sobre duas classes
    :param c1v: Conjunto de vetores de treinamento da classe 1
    :param c2v: Conjunto de vetores de treinamento da classe 2
    :param max_iters: O número máximo de iterações do algoritmo
    :param alpha: O fator de randomização, recomendado usar valores entre 0.1 a 1
    :return: A equação do perceptron para os vetores, representação legível dessa equação e a quantidade de iterações necessárias.
    """
    # inicializa pesos com vetor de zeros, aumentado de zero [0, 0, ..., 0]
    w = np.zeros(len(c1v[0]) + 1)

    # função de obtenção de erro
    # 1/2 * (r - w.T * yk)**2
    get_err = lambda r, x: 1/2 * ((r - w.T.dot(yk))**2)

    # erros acumulados
    errors = []

    # alterna c1, c2, c1, c2
    # aumenta vetores (..., 1)
    train_vectors = [augment(v, 1) for pair in zip(c1v, c2v) for v in pair]

    # condição de parada
    # erro < 0.1, iterações > 3
    stop = False

    while len(errors) < max_iters and not stop:
        for i, xk in enumerate(train_vectors):
            r = 1 if i%2 == 0 else -1
            
            # err = 1/2 * ((r - w.T * xk) **2)
            err = 1/2 * ((r - w.T.dot(xk))**2) # E(w)
            errors.append(err)
            
            # w = w + alpha * (r - (w.T*xk)) * xk
            w = w + alpha * (r - w.T.dot(xk)) * xk

            # extrai média dos 2 últimos erros
            total_err = sum(errors[-2:])/2

            # pára se o erro é menor que o aceitável
            # e se os vetores de treino foram usados pelo menos uma vez
            if total_err <= 0.1 and len(errors) > len(train_vectors):
                stop = True
                break

    eq = ""
    for i, wi in enumerate(w):
        eq += _get_term(i, wi, i==len(w)-1)
    
    func = lambda x: w.dot(augment(x, 1))
    # func = lambda x: w * augment(x, 1)
    return func, eq, len(errors)
