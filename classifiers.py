"""
Classificadores e avaliadores
:author: Otávio Ferreira
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Callable
from statistics import NormalDist


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


@dataclass(frozen=True)
class TrainedClassifier:
    "Armazena o resultado do treinamento de um algoritmo de classificação."
    
    eq: str
    "Representação matemática da equação de classificação."
    
    func: Callable[[np.ndarray], float]
    "Função de classificação que recebe um vetor e retorna um número"

    iters: int=-1
    """Número de iterações necessárias para treino.
    Valor negativo indica que o algoritmo não é iterativo."""


class Classifiers(object):
    "Classe que contém métodos para treino de classificadores."

    @staticmethod
    def euclidean_dist(m: np.ndarray) -> TrainedClassifier:
        """
        Calcula a equação de distância euclideana.

        :param m: O vetor característica.
        :returns: Um classificador treinado.
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
        return TrainedClassifier(eq, func)

    @staticmethod
    def max_dist(m: np.ndarray) -> TrainedClassifier:
        """
        Calcula a equação de classificador máximo.

        :param m: O vetor característica.
        :returns: Um classificador treinado.
        """
        eq = "".join(_get_term(i, mi) for i,mi in enumerate(m))

        # extra = -1/2 * (m.T * m)
        extra = -1/2 * m.T.dot(m)
        if extra != 0:
            sign = "-" if extra < 0 else '+'
            eq += f'{sign}{abs(extra)}'
        
        # func = lambda x: (x.T * m) - (1/2 * (m.T * m))
        func = lambda x: m.T.dot(x) - (1/2 * (m.dot(m)))
        return TrainedClassifier(eq, func)

    @staticmethod
    def dij(mi: np.ndarray, mj: np.ndarray) -> TrainedClassifier:
        """
        Calcula a equação de superfície de decisão sobre duas classes
        :param mi: Vetor característica da classe i
        :param mj: Vetor característica da classe j
        :returns: Um classificador treinado.
        """
        eq = "".join(_get_term(i, di) for i,di in enumerate(mi-mj))
        
        # extra = -1/2 * ((mi - mj).T * (mi + mj))
        extra = -1/2 * (mi - mj).T.dot(mi + mj)
        if extra != 0:
            sign = '-' if extra < 0 else '+'
            eq += f'{sign}{abs(extra)}'
        
        # func = lambda x: ((mi - mj).T * x) - 1/2 * ((mi - mj).T * (mi + mj))
        func = lambda x: (mi - mj).T.dot(x) - 1/2 * (mi - mj).T.dot(mi + mj)
        return TrainedClassifier(eq, func)

    @staticmethod
    def perceptron(c1v: list[np.ndarray], c2v: list[np.ndarray], max_iters=10_000, c=1) -> TrainedClassifier:
        """
        Calcula a equação sobre duas classes pelo método perceptron.

        :param c1v: Conjunto de vetores de treinamento da classe 1
        :param c2v: Conjunto de vetores de treinamento da classe 2
        :param max_iters: O número máximo de iterações do algoritmo
        :param c: O fator de randomização, recomendado usar valores entre 0 a 1
        :returns: Um classificador treinado.
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

        return TrainedClassifier(eq, func, iters)


    @staticmethod
    def delta_perceptron(c1v: list[np.ndarray], c2v: list[np.ndarray], max_iters=10_000, alpha=1) -> TrainedClassifier:
        """
        Calcula a equação perceptron com regra delta sobre duas classes.

        :param c1v: Conjunto de vetores de treinamento da classe 1
        :param c2v: Conjunto de vetores de treinamento da classe 2
        :param max_iters: O número máximo de iterações do algoritmo
        :param alpha: O fator de randomização, recomendado usar valores entre 0.1 a 1
        :returns: Um classificador treinado.
        """
        # inicializa pesos com vetor de zeros, aumentado de zero [0, 0, ..., 0]
        w = np.zeros(len(c1v[0]) + 1)

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
        
        return TrainedClassifier(eq, func, len(errors))


class ConfusionMatrix(object):
    """
    Matriz de confusão.
    """
    def __init__(self, matrix: list[list[int]]):
        """
        Construtor

        :param matrix: Matriz de confusão.
        """
        self._matrix = matrix
        self._c = len(self._matrix)
        for line in self._matrix:
            if len(line) != self.c:
                raise ValueError("Incorrect Array!")
        self._m = sum(x for line in self._matrix for x in line)
        self._Ag = sum(self._matrix[i][i] for i in range(self.c))/self.m
        self._Aa = sum(ai_*a_i for ai_,a_i in zip(self["i+"], self["+i"])) / (self.m**2)
        self._kappa = (self.Ag - self.Aa) / (1 - self.Aa)
        self._tau = (self.Ag - 1/self.c) / (1 - 1/self.c)
        self._kappa_var = self.__kappa_var()
        self._tau_var = self.__tau_var()
    
    def __kappa_var(self) -> float:
        """
        Calcula a variância do coeficiente kappa.

        :returns: A variancia de K 
        """
        v1 = self.Ag
        v2 = self.Aa

        v3 = sum(self[i][i]*(self["i+"][i] + self["+i"][i]) for i in range(self.c))
        v3 /= self.m ** 2
        
        v4 = sum(self[i][j] * (self["i+"][j] + self["+i"][i]) 
            for i in range(self.c)
            for j in range(self.c))
        v4 /= self.m ** 3

        part1 = v1 * (1 - v1)
        part1 /= (1 - v1) **2

        part2 = 2*(1 - v1) * (2 * v1 * v2 - v3)
        part2 /= (1 - v2) **3

        part3 = ((1 - v1) **2) * (v4 - 4 * v2)**2
        part3 /= (1 - v2)**4

        var = (1 / self.m) * (part1 + part2 + part3)
        return var

    def __tau_var(self) -> float:
        """
        Calcula a variância do coeficiente tau.

        :returns: A variância de tau.
        """
        part1 = (1 / self.m)

        part2 = (self.Ag * (1 - self.Ag))
        part2 /= (1 - 1/self.c)**2

        return part1 * part2


    def __getitem__(self, i: int|str) -> list[int]|int:
        """
        Obtém uma linha dessa matriz.

        :param i: Índice da linha ou coluna.
        Valores em string retornam colunas.
        Valores terminados em "+" retornam o acumulado parcial da linha, por exemplo "5+"
        Valores iniciados em "+" retornam o acumulado parcial da coluna, por exemplo "+1"
        "i+" retorna todos os acumulados parciais da linha
        "+i" retorna todos os acumulados parciais da coluna
        """
        if type(i) is str:
            if i == "+i":
                # linha de somas parciais
                columns = tuple(zip(*self._matrix))
                return [sum(columns[i]) for i in range(self.c)]
            if i == "i+":
                # coluna de somas parciais
                return [sum(self._matrix[i]) for i in range(self.c)]
            if i.startswith("+"):
                # soma parcial da coluna i
                index = int(i[1:])
                columns = tuple(zip(*self._matrix))
                return sum(columns[index])
            if i.endswith("+"):
                # soma parcial da linha i
                index = int(i[:-1])
                return sum(self._matrix[index])
            else:
                # coluna i
                try:
                    index = int(i)
                    columns = tuple(zip(*self._matrix))
                    return columns(index)
                except:
                    pass
                raise ValueError("Índice inválido!")
        return self._matrix[i]

    @property
    def columns(self) -> list[list[int]]:
        """
        Troca linhas por colunas da matriz de confusão.
        :returns: A matriz de confusão transposta.
        """
        return list(zip(*self._matrix))
    
    @property
    def c(self):
        """A quantidade de classes."""
        return self._c

    @property
    def m(self):
        """O total de amostras."""
        return self._m
    
    @property
    def Ag(self):
        """O acerto geral."""
        return self._Ag
        
    @property
    def Aa(self):
        """O acerto casual/aleatório."""
        return self._Aa

    @property
    def kappa(self):
        """O coeficiente kappa."""
        return self._kappa

    @property
    def tau(self):
        """O coeficiente tau."""
        return self._tau

    @property
    def kappa_var(self):
        """A variância do coeficiente kappa."""
        return self._kappa_var

    @property
    def tau_var(self):
        """A variância do coeficiente tau."""
        return self._tau_var    

    def to_binary_matrix(self, ci: int) -> 'BinaryMatrix':
        """
        Transforma essa matriz em uma matriz binária selecionando uma classe como referência.

        A matriz resultante indica como os elementos foram classificados em relação a classe.
        :param ci: Indice começando em 0 da classe a ser selecionada.
        :returns: Uma matriz binária da classificação de uma classe em relação às outras.
        """
        # Verdadeiro Positivo: diagonal principal (A11, A22, A33, ...)
        vp = sum(self[i][i] for i in range(self.c))

        # Verdadeiro Negativo: itens da linha que não são da diagonal principal
        # (A12, A22, A32, ...) na classe 1
        vn = sum(self[i][ci] for i in range(self.c) if i != ci)

        # Falso Positivo: itens da coluna que não são da diagonal principal
        # (A12, A13, A14, ...) na classe 1
        fp = sum(self[ci][i] for i in range(self.c) if i != ci)

        # Falso Negativo: itens que não são da mesma linha, coluna ou diagonal principal
        # (A23, A24, A32, ...) na classe 1
        fn = sum(self[i][j]
            for i in range(self.c)
            for j in range(self.c)
            if i != j and i != ci and j != ci)

        return BinaryMatrix(vp, fp, fn, vn)


class BinaryMatrix(object):
    """
    Matriz binária.
    """
    def __init__(self, vp: int, fp: int, fn: int, vn: int):
        self._vp = vp
        self._fp = fp
        self._fn = fn
        self._vn = vn

        m = vp+fp+fn+vn
        self._Pr = vp / (vp + fp) # Precisão
        self._Re = vp / (vp + fn) # Sensibilidade
        self._Es = vn / (fp + vn) # Especificidade
        self._Ac = (vp + vn) / m # Acurácia

    @property
    def vp(self) -> int:
        """Verdadeiro positivo."""
        return self._vp

    @property
    def fp(self) -> int:
        """Falso positivo."""
        return self._fp

    @property
    def fn(self) -> int:
        """Falso negativo."""
        return self._fn

    @property
    def vn(self) -> int:
        """Verdadeiro negativo."""
        return self._vn

    @property
    def Pr(self) -> float:
        """Precisão."""
        return self._Pr
    
    @property
    def Se(self) -> float:
        """Sensibilidade."""
        return self._Re
    
    @property
    def Re(self) -> float:
        """Sensibilidade."""
        return self._Re
    
    @property
    def Es(self) -> float:
        """Especificidade."""
        return self._Es
    
    @property
    def Ac(self) -> float:
        """Acurácia."""
        return self._Ac

    def F(self, b=1) -> float:
        """
        F-score

        :params b: O parâmetro de escolha. Pode ser 1, 2, ou 1/2.
        :returns: O f-score.
        :raises ValueError: se o parâmetro b for inválido
        """
        if b not in (1, 2, 0.5):
            raise ValueError("Valor de b inválido. b só pode ser 1, 2, ou 1/2.")

        f = self.Pr * self.Re
        f /= ((b**2) * self.Pr) + self.Re

        return (1+b) * f
    
    def matthews(self) -> float:
        """
        Coeficiente de Matthews.
        :returns: O coeficiente.
        """
        vp, vn, fp, fn = self.vp, self.vn, self.fp, self.fn
        mcc = vp * vn - fp * fn
        mcc /= math.sqrt((vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))

        return mcc
    

def is_kappa_significant(m1: ConfusionMatrix, m2: ConfusionMatrix, alpha: float) -> bool:
    """
    Verifica se há diferença entre os coeficientes kappa com um nível de significância.
    :param m1: A matriz de confusão
    :param m2: A matriz de confusão
    :param alpha: Nível de significância
    :returns: True se há diferença entre os coeficientes kappa, senão False
    """
    r = 1 - alpha/2
    zr = NormalDist().inv_cdf(r)
    zk = (m1.kappa - m2.kappa) / math.sqrt(m1.kappa_var + m2.kappa_var)
    
    return abs(zk) > zr

def is_tau_significant(m1: ConfusionMatrix, m2: ConfusionMatrix, alpha: float) -> bool:
    """
    Verifica se há diferença entre os coeficientes tau com um nível de significância.
    :param m1: A matriz de confusão
    :param m2: A matriz de confusão
    :param alpha: Nível de significância
    :returns: True se há diferença entre os coeficientes tau, senão False
    """
    r = 1 - alpha/2
    zr = NormalDist().inv_cdf(r)
    zk = (m1.tau - m2.tau) / math.sqrt(m1.tau_var + m2.tau_var)
    
    return abs(zk) > zr


if __name__ == "__main__":
    matrix = ConfusionMatrix([
        [140, 20, 0, 0],
        [10, 130, 0, 0],
        [5, 0, 150, 10],
        [15, 10, 0, 120]
    ])

    print("c: ", matrix.c)
    print("A+i: ", matrix["+i"])
    print("Ai+: ", matrix["i+"])
    print("m: ", matrix.m)
    print("Ag: ", matrix.Ag)
    print("Aa: ", matrix.Aa)
    print("kappa: ", matrix.kappa, matrix.kappa_var)
    print("tau: ", matrix.tau, matrix.tau_var)

    matrix2 = ConfusionMatrix([
        [140, 30, 2, 0],
        [10, 110, 5, 0],
        [0, 10, 140, 0],
        [20, 10, 3, 140]
    ])

    print("Is kappa significant on 5%?:", is_kappa_significant(matrix, matrix2, 0.05))
    print("Is tau significant on 5%?:", is_tau_significant(matrix, matrix2, 0.05))

    m1bin = matrix2.to_binary_matrix(0)
    print("\n\nMatriz binária sobre w0")
    print("verdadeiros positivos: ", m1bin.vp)
    print("verdadeiros negativos: ", m1bin.vn)
    print("falsos positivos: ", m1bin.fp)
    print("falsos negativos: ", m1bin.fn)

    print("\nPrecisão: ", m1bin.Pr)
    print("Sensibilidade: ", m1bin.Se)
    print("Especificidade: ", m1bin.Es)
    print("Acurácia: ", m1bin.Ac)

    print("\nF1: ", m1bin.F(1))
    print("F2: ", m1bin.F(2))
    print("F1/2: ", m1bin.F(1/2))

    print("Mattheus: ", m1bin.matthews())