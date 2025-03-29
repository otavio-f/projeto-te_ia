"""
Classificadores
:author: Otávio Ferreira
"""

import math
from dataclasses import dataclass
from dataclasses import field
from vector.vector import Vector
from typing import Callable, Any, Iterable, List


def _get_term(i: int, xk: float, is_indep=False) -> str:
    """
    Retorna uma representação do termo na equação. Ex.: x1, -x2, x3
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
    else:
        mul = f"{abs(xk)}"

    if is_indep:
        term = ""
    else:
        term = f"x{i+1}"
    
    return f"{sign}{mul}{term}"


def euclidean_dist(v: Iterable[float]) -> [Callable[[float, ...], float], str]:
    """
    Calcula a equação de distância euclideana.
    :param v: O vetor característica.
    :returns: A equação de distância euclideana e uma representação legível dessa equação.
    """
    m = Vector.of(*v)
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

    return lambda x: math.sqrt((x - m).T * (x - m)), eq


def max_dist(v: Iterable[float]) -> [Callable[[float, ...], float], str]:
    """
    Calcula a equação de classificador máximo.
    :param v: O vetor característica.
    :returns: A equação de classificador máximo e uma representação legível dessa equação.
    """
    m = Vector.of(*v)
    eq = "".join(_get_term(i, mi) for i,mi in enumerate(m))

    extra = -1/2 * (m.T * m)
    if extra != 0:
        sign = "-" if extra < 0 else '+'
        eq += f'{sign}{abs(extra)}'
    
    return lambda x: (Vector(x).T * m) - (1/2 * (m.T * m)), eq


def dij(vi: Iterable[float], vj: Iterable[float]) -> [Callable[[float, ...], float], str]:
    """
    Calcula a equação de superfície de decisão sobre duas classes
    :param vi: Vetor característica da classe i
    :param vj: Vetor característica da classe j
    :return: A equação de superfície de decisão e uma representação legível dessa equação.
    """
    mi = Vector(vi)
    mj = Vector(vj)

    eq = "".join(_get_term(i, di) for i,di in enumerate(mi-mj))
    
    extra = -1/2 * ((mi - mj).T * (mi + mj))
    if extra != 0:
        sign = '-' if extra < 0 else '+'
        eq += f'{sign}{abs(extra)}'
    
    return lambda x: ((mi - mj).T * x) - 1/2 * ((mi - mj).T * (mi + mj)), eq


def perceptron(c1v: Iterable[float], c2v: Iterable[float], max_iters=1_000_000, c=1) -> [Callable[[float, ...], float], str, int]:
    """
    Calcula a equação sobre duas classes pelo método perceptron
    :param c1v: Conjunto de vetores de treinamento da classe 1
    :param c2v: Conjunto de vetores de treinamento da classe 2
    :param max_iters: O número máximo de iterações do algoritmo
    :param c: O fator de randomização, recomendado usar valores entre 0 a 1
    :return: A equação do perceptron para os vetores, representação legível dessa equação e a quantidade de iterações necessárias.
    """
    w = Vector([0 for _ in c1v[0]]).augment(0) # vetor aumentado [..., 0]
    iters = 0
    vecs = [v for pair in zip(c1v, c2v) for v in pair] # alternado c1, c2, c1, c2
    stop = False
    clean_steps = 0
    while iters < max_iters and not stop:
        for i,v in enumerate(vecs):
            iters += 1
            xk = v.augment(1) # Vetor aumentado [..., 1]
            test = w * xk.T
            test_ok = (test > 0) if i%2==0 else (test < 0) # c1 par >0, c2 impar <0
            if not test_ok:
                clean_steps = 0
                w = w + (c * xk)
            else:
                clean_steps += 1
            if clean_steps > 3:
                stop = True
                break

    eq = ""
    for i, wi in enumerate(w):
        eq += _get_term(i, wi, i==len(w)-1)
    
    return lambda x: w * Vector(x).augment(1), eq, iters


def delta_perceptron(c1v: [float], c2v: [float], max_iters=1_000_000, alpha=1) -> [Callable[[float, ...], float], str, int]:
    """
    Calcula a equação perceptron com regra delta sobre duas classes
    :param c1v: Conjunto de vetores de treinamento da classe 1
    :param c2v: Conjunto de vetores de treinamento da classe 2
    :param max_iters: O número máximo de iterações do algoritmo
    :param alpha: O fator de randomização, recomendado usar valores entre 0.1 a 1
    :return: A equação do perceptron para os vetores, representação legível dessa equação e a quantidade de iterações necessárias.
    """
    w = Vector([0 for _ in c1v[0]]).augment(0) # vetor aumentado [..., 0]
    get_err = lambda r, x: 1/2 * ((r - w.T * yk)**2)
    errors = []
    vecs = [v for pair in zip(c1v, c2v) for v in pair] # alternado c1, c2, c1, c2
    stop = False

    while len(errors) < max_iters and not stop:
        for i, v in enumerate(vecs):
            r = 1 if i%2 == 0 else -1 # c1=1, c2=-1
            xk = v.augment(1)
            err = 1/2 * ((r - w.T * xk) **2)
            errors.append(err)
            w = w + alpha * (r - (w.T*xk)) * xk
            if sum(errors[-2:])/2 <= 0.1:
                stop = True
                break

    eq = ""
    for i, wi in enumerate(w):
        eq += _get_term(i, wi, i==len(w)-1)
    
    return lambda x: w * Vector(x).augment(1), eq, errors
