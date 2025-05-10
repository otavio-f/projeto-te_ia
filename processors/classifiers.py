"""
Classificadores
:author: Otávio Ferreira
"""

import numpy as np
from dataclasses import dataclass
from typing import Collection
import itertools


@dataclass(frozen=True)
class ClassifiedSample:
    "Armazena uma amostra classificada."

    data: np.array
    "Vetor de amostra."

    expected: str
    "Classe esperada"

    predicted: str
    "Classe prevista"

    value: float
    "Valor da classificação"


def convert_to_latex(i: int, xk: float, independent=False, hide_first_plus=True, variable="x") -> str:
    """
    Retorna uma representação do termo na equação, compatível com latex.

    :param i: índice do termo
    :param xk: multiplicador do termo
    :param independente: se é um termo independente da variável
    :param hide_first_plus: se True, esconde o sinal positivo do primeiro termo da equação
    :param variable: nome da variável
    """
    if xk == 0:
        return ""
    
    if xk < 0:
        sign = "-"
    elif i==0 and hide_first_plus:
        sign = ""
    else:
        sign = "+"
    
    if abs(xk) == 1 and not independent:
        mul = ""
    elif abs(xk - int(xk)) == 0: # é inteiro
        mul = f"{abs(int(xk))}"
    else:
        mul = f"{str(abs(xk))}"

    if independent:
        term = ""
    else:
        term = f"{variable}_{{{i+1}}}"
    
    return f"{sign}{mul}{term}"

def augment(v: np.ndarray, aug: int|float) -> np.ndarray:
    """
    Aumenta um vetor com um termo.

    :param v: Vetor que irá ser aumentado
    :param aug: Número que irá aumentar o vetor
    :returns: O vetor com o elemento aumentado ao final.
    """
    result = np.concat((v, [aug,]))
    return result

def to_matrix(v: np.ndarray) -> np.matrix:
    """
    Transforma o vetor em uma matriz coluna

    :param v: Vetor a ser transformado
    :returns: Uma matriz coluna
    """
    return np.asmatrix(v).T


class UntrainedClassifier(Exception):
    """Exceção que indica que um classificador não foi treinado."""
    pass


class EuclideanDistance(object):
    """Classificador de distância euclideana."""
    def __init__(self):
        pass
    
    def gen_latex(self):
        self.latex = ""
        # x1² + x2² + ... xn²
        for i, _ in enumerate(self.m):
            self.latex += convert_to_latex(i, 1, variable="x^{2}")
        
        # - 2*m1x1 - 2*m2x2 - ... - 2*mnxn
        for i, mi in enumerate(self.m):
            self.latex += convert_to_latex(i, -2*mi, hide_first_plus=False)
        
        # m1² + m2² + ... + mn²
        total = sum(mi**2 for mi in self.m)
        self.latex += convert_to_latex(1, total, independent=True, hide_first_plus=False)
        
        self.latex = f"\\sqrt{{{self.latex}}}"

    def fit(self, m: np.ndarray):
        self.m = m
        self.gen_latex()

    def predict(self, x: np.ndarray) -> float:
        if not hasattr(self, 'm'):
            raise UntrainedClassifier("Classificador não foi treinado!")
        
        M = to_matrix(self.m)
        X = to_matrix(x)
        return np.sqrt((X - M).T * (X - M)).item()


class MinimumDistance(object):
    """Classificador de distância mínima."""
    def __init__(self):
        pass
    
    def gen_latex(self):
        self.latex = "".join(convert_to_latex(i, mi) for i,mi in enumerate(self.m))

        # extra = -1/2 * (m.T * m)
        extra = -1/2 * self.m @ self.m
        self.latex += convert_to_latex(1, extra, independent=True, hide_first_plus=False)
        

    def fit(self, m: np.ndarray):
        self.m = m
        self.gen_latex()

    def predict(self, x: np.ndarray) -> float:
        if not hasattr(self, 'm'):
            raise UntrainedClassifier("Classificador não foi treinado!")
        
        M = to_matrix(self.m)
        X = to_matrix(x)
        result = (X.T * M) - (1/2 * (M.T * M))
        return result.item()


class MinimumDistance2(object):
    """Classificador de distância mínima com superfície de decisão."""
    def __init__(self):
        pass
    
    def gen_latex(self):
        mi = self.mi
        mj = self.mj

        self.latex = "".join(convert_to_latex(i, mi) for i,mi in enumerate(mi - mj))

        # extra = -1/2 * (m.T * m)
        extra = -1/2 * (mi - mj).T @ (mi + mj)
        self.latex += convert_to_latex(1, extra, independent=True, hide_first_plus=False)
        

    def fit(self, mi: np.ndarray, mj: np.ndarray):
        self.mi = mi
        self.mj = mj
        self.gen_latex()

    def predict(self, x: np.ndarray) -> float:
        if not hasattr(self, 'mi'):
            raise UntrainedClassifier("Classificador não foi treinado!")
        
        Mi = to_matrix(self.mi)
        Mj = to_matrix(self.mj)
        X = to_matrix(x)
        result = (Mi - Mj).T * X - 1/2 * ((Mi - Mj).T * (Mi + Mj))
        return result.item()


class Perceptron(object):
    """Classificador perceptron."""
    def __init__(self, c: float, max_iters: int):
        """
        :param c: O fator de aprendizado, um número maior que 0 e menor ou igual a 1.
        :param max_iters: O limite de iterações permitido, deve ser maior que zero.
        """
        if c <= 0 or c > 1:
            raise ValueError("Valor de c fora do intervalo permitido!")
        if max_iters <= 0 or type(max_iters) is not int:
            raise ValueError("Valor inválido para o número máximo de iterações!")
        self.c = c
        self.max_iters = max_iters
    
    def gen_latex(self):
        self.latex = ""
        for i, wi in enumerate(self.w):
            self.latex += convert_to_latex(i, wi, i==len(self.w)-1)

    def fit(self, Ci: Collection[np.ndarray], Cj: Collection[np.ndarray]):
        # Inicializa pesos com vetor de zeros
        W = np.zeros(len(Ci[0]) + 1)

        # alterna c1, c2, c1, c2
        # aumenta vetores (..., 1)
        # vetores da classe 1 estão nos índices pares
        # vetores da classe 2 estão nos índices ímpares
        train_vectors = (augment(v, 1) for pair in zip(Ci, Cj) for v in pair)

        self.errors = [] # erro de cada iteração
        # True se houve erro, False se não houve

        for K, X in enumerate(itertools.cycle(train_vectors)):
            # itera enquanto as condições de parada não foram atendidas
            test = W.T @ X
            if K%2 == 0:
                # classe 1
                if not test > 0:
                    W = W + (self.c * X)
                self.errors.append(not test > 0)
            if K%2 == 1:
                # classe 2
                if not test < 0:
                    W = W - (self.c * X)
                self.errors.append(not test < 0)
                
            if K > len(Ci)+len(Cj):
                if not any(self.errors[-5:]):
                    # pare se as últimas cinco iterações não tiveram erro
                    self.w = W
                    self.gen_latex()
                    return
                
            if K >= self.max_iters:
                # pare quando superar o limite de iterações
                # nesse caso não converge
                self.w = W
                self.gen_latex()
                return

    def predict(self, x: np.ndarray) -> float:
        if not hasattr(self, 'w'):
            raise UntrainedClassifier("Classificador não foi treinado!")
        
        W = to_matrix(self.w)
        X = to_matrix(augment(x, 1))
        return (W.T * X).item()
    
    @property
    def iterations(self):
        return len(self.errors)


class PerceptronDelta(object):
    """Classificador perceptron com regra delta."""
    def __init__(self, alpha: float, max_iters: int):
        """
        :param alpha: O fator de aprendizado, um número maior que 0 e menor ou igual a 1.
        :param max_iters: O limite de iterações permitido, deve ser maior que zero.
        """
        if alpha <= 0 or alpha > 1:
            raise ValueError("Valor de c fora do intervalo permitido!")
        if max_iters <= 0 or type(max_iters) is not int:
            raise ValueError("Valor inválido para o número máximo de iterações!")
        self.alpha = alpha
        self.max_iters = max_iters
    
    def gen_latex(self):
        self.latex = ""
        for i, wi in enumerate(self.w):
            self.latex += convert_to_latex(i, wi, i==len(self.w)-1)

    def fit(self, Ci: Collection[np.ndarray], Cj: Collection[np.ndarray]):
        # Inicializa pesos com vetor de zeros
        W = np.zeros(len(Ci[0]) + 1)
        alpha = self.alpha

        # alterna c1, c2, c1, c2
        # aumenta vetores (..., 1)
        # vetores da classe 1 estão nos índices pares
        # vetores da classe 2 estão nos índices ímpares
        train_vectors = (augment(v, 1) for pair in zip(Ci, Cj) for v in pair)

        self.errors = []

        # itera enquanto as condições de parada não foram atendidas
        for K, X in enumerate(itertools.cycle(train_vectors)):
            if K%2 == 0: # índice par, classe 1
                R = 1
            if K%2 == 1: # índice ímpar, classe 2
                R = -1
            err = 1/2 * ((R - W.T @ X) ** 2)
            self.errors.append(err)

            W = W + alpha * (R - W.T @ X) * X

            if K > len(Ci)+len(Cj):
                if np.mean(self.errors[-3:]) < 0.1:
                    # pare se as últimas cinco iterações tiveram erro baixo
                    self.w = W
                    self.gen_latex()
                    return

            if K > self.max_iters:
                # pare quando superar o limite de iterações
                # nesse caso não converge
                self.w = W
                self.gen_latex()
                return


    def predict(self, x: np.ndarray) -> float:
        if not hasattr(self, 'w'):
            raise UntrainedClassifier("Classificador não foi treinado!")
        
        W = to_matrix(self.w)
        X = to_matrix(augment(x, 1))
        return (W.T * X).item()

    @property
    def iterations(self):
        return len(self.errors)

class Bayes(object):
    """Classificador de distância mínima."""
    def __init__(self):
        pass
    
    def gen_latex(self):
        Mi = to_matrix(self.mi)
        Mj = to_matrix(self.mj)
        
        x_terms = (self.cov_i.I * Mi) - (self.cov_j.I * Mj)

        self.latex = "".join(convert_to_latex(i, xi) for i,xi in enumerate(x_terms.A1))
        
        extra = (np.log(self.prob_i) - np.log(self.prob_j))
        extra += (-1/2 * (Mi.T * self.cov_i.I * Mi)) - (-1/2 * (Mj.T * self.cov_j.I * Mj)).item()

        self.latex += convert_to_latex(1, extra, independent=True, hide_first_plus=False)
        
        
    def get_covariance_matrix(self, train_vectors: list[np.ndarray], m: np.ndarray) -> np.matrix:
        # inicialização
        M = to_matrix(m)
        E = np.asmatrix(np.zeros((M * M.T).shape))

        # somatório
        for v in train_vectors:
            X = to_matrix(v)
            E += (X * X.T) - (M * M.T)

        n = len(train_vectors)
        return 1/n * E

    def fit(self, train_Ci: Collection[np.ndarray], train_Cj: Collection[np.ndarray], prob_Ci: float, prob_Cj: float):
        self.prob_i = prob_Ci
        self.prob_j = prob_Cj

        self.mi = np.mean(train_Ci, axis=0)
        self.mj = np.mean(train_Cj, axis=0)

        self.ci = train_Ci
        self.cj = train_Cj

        self.cov_i = self.get_covariance_matrix(train_Ci, self.mi)
        self.cov_j = self.get_covariance_matrix(train_Cj, self.mj)

        self.gen_latex()


    def predict(self, x: np.ndarray) -> float:
        if not hasattr(self, 'mi'):
            raise UntrainedClassifier("Classificador não foi treinado!")
        
        Mi = to_matrix(self.mi)
        Mj = to_matrix(self.mj)
        X = to_matrix(x)

        di = np.log(self.prob_i) + (X.T * self.cov_i.I * Mi) - (1/2 * Mi.T * self.cov_i.I * Mi)
        dj = np.log(self.prob_j) + (X.T * self.cov_j.I * Mj) - (1/2 * Mj.T * self.cov_j.I * Mj)
        return (di - dj).item()
