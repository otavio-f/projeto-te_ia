"""
Avaliadores de classificadores.
:author: Otávio Ferreira.
"""
import numpy as np
import math
from statistics import NormalDist

from dataclasses import dataclass

from processors.classifiers import ClassifiedSample


@dataclass(frozen=True, init=False)
class ConfusionMatrix:
    """
    Matriz de confusão.
    """

    _M: np.matrix
    "Matriz de confusão"
    
    classes: list[str]
    "Classes da matriz de confusão."

    def __init__(self, m: list[list[str]], classes: list[str]):
        """
        Construtor.

        :param m: Matriz de confusão.
        Deve ser uma lista de duas dimensões, cada uma com N elementos (NxN)
        :param classes: Classes da matriz de confusão.
        Deve ter a mesma quantidade de elementos que cada linha/coluna da matriz.
        :raises ValueError: Se a matriz não for válida.
        """
        # a classe é pseudo-imutável
        # o único modo de setar os atributos é usando object.__setattr__
        object.__setattr__(self, '_M', np.asmatrix(m, dtype=np.uint16))
        object.__setattr__(self, 'classes', classes)
        
        # checa se a matriz é válida
        # deve ser uma matriz quadrada NxN, sendo N o número de classes
        if self._M.shape != (self.c, self.c):
            raise ValueError("Number of classes and matrix dimensions mismatch.")

    def M(self, i: int|str, j: int|None = None) -> int|np.ndarray:
        """
        Obtém um item dessa matriz.
        Se for especificado um valor inteiro, j também deve ser informado.
        Se for especificado um valor em string, j será ignorado.
        Valores terminados em \"+\" retornam o acumulado parcial da linha, por exemplo \"5+\"
        Valores iniciados em \"+\" retornam o acumulado parcial da coluna, por exemplo \"+1\"
        \"i+\" retorna todos os acumulados parciais da linha
        \"+i\" retorna todos os acumulados parciais da coluna

        :param i: Índice da linha ou parcial.
        :param j: Índice da coluna.

        :returns: Item ou linha ou coluna se o primeiro parâmetro indica índice parcial.
        """
        # item i,j da matriz
        if type(i) is int:
            result = self._M[i, j]
            return self._M[i, j]
        
        # linha de somas parciais
        elif i == "+i":
            result = self._M.sum(axis=0).A1
        
        # coluna de somas parciais
        elif i == "i+":
            result = self._M.sum(axis=1).A1
        
        # soma parcial da coluna i, ex.: "+5"
        elif i.startswith("+"):
            index = int(i[1:])
            result = self._M.T[index].sum()
        
        # soma parcial da linha i, ex.: "5+"
        elif i.endswith("+"):
            index = int(i[:-1])
            result = self._M[index].sum()

        else:
            raise NotImplementedError()

        return result

    @property
    def c(self):
        """A quantidade de classes."""
        return len(self.classes)

    @property
    def m(self):
        """O total de amostras."""
        return self._M.sum()
    
    @property
    def Ag(self):
        """O acerto geral."""
        M = self._M
        return M.diagonal().sum() / M.sum()
        
        
    @property
    def Aa(self):
        """Acerto casual/aleatório."""
        # multiplica termo a termo
        # (a1+ * a+1, a2+ * a+2, ..., an+ * a+n)
        partial_prod = np.multiply(self.M("i+"), self.M("+i"))

        # somatório dos produtos termo a termo
        partial_sums = partial_prod.sum()

        return partial_sums / (self.m**2)
        
    @property
    def Ap(self):
        """Acurácia do produtor.""" 
        return self._M.diagonal() / self.M("+i")
    
    @property
    def Au(self):
        """Acurácia do usuário."""
        return self._M.diagonal() / self.M("i+")

    @property
    def kappa(self):
        """O coeficiente kappa."""
        return (self.Ag - self.Aa) / (1 - self.Aa)

    @property
    def kappa_var(self) -> float:
        """
        Calcula a variância do coeficiente kappa.

        :returns: A variancia de K 
        """
        v1 = self.Ag
        v2 = self.Aa

        v3 = self._M.diagonal().T * (self.M("i+") + self.M("+i"))
        v3 = v3.sum() / (self.m ** 2)
        # v3 = sum(self.M(i, i) * (self.M("i+")[i] + self.M("+i")[i]) for i in range(self.c))
        # v3 /= (self.m ** 2)
        
        v4 = sum(self.M(i, j) * (self.M("i+")[j] + self.M("+i")[i]) 
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

    @property
    def tau(self):
        """O coeficiente tau."""
        return (self.Ag - 1/self.c) / (1 - 1/self.c)

    @property
    def tau_var(self) -> float:
        """
        Calcula a variância do coeficiente tau.

        :returns: A variância de tau.
        """
        part1 = (1 / self.m)

        part2 = (self.Ag * (1 - self.Ag))
        part2 /= (1 - 1/self.c)**2

        return part1 * part2


@dataclass(frozen=True)
class BinaryMatrix:
    """
    Matriz binária.
    """

    vp: int
    "Verdadeiro positivo"

    fp: int
    "Falso positivo"

    fn: int
    "Falso negativo"

    vn: int
    "Verdadeiro negativo"

    @property
    def Pr(self) -> float:
        """Precisão."""
        return self.vp / (self.vp + self.fp)
    
    @property
    def Se(self) -> float:
        """Sensibilidade."""
        return self.Re
    
    @property
    def Re(self) -> float:
        """Sensibilidade."""
        return self.vp / (self.vp + self.fn)
    
    @property
    def Es(self) -> float:
        """Especificidade."""
        self._Es = self.vn / (self.fp + self.vn)
    
    @property
    def Ac(self) -> float:
        """Acurácia."""
        m = self.vp + self.fp + self.fn + self.vn
        return (self.vp + self.vn) / m

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
    
    @property
    def matthews(self) -> float:
        """
        Coeficiente de Matthews.
        :returns: O coeficiente.
        """
        vp, vn, fp, fn = self.vp, self.vn, self.fp, self.fn
        mcc = vp * vn - fp * fn
        mcc /= math.sqrt((vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))

        return mcc
    
    @staticmethod
    def from_confusion_matrix(cm: ConfusionMatrix, cls: str) -> 'BinaryMatrix':
        """
        Cria uma matriz binária a partir de uma classe de uma matriz de confusão.
        
        :param cm: A matriz de confusão.
        :param cls: O nome da classe.
        :returns: Uma instância de matriz binária que classifica a classe em relação às outras.
        """
        # obtém o índice da classe
        ci = cm.classes.index(cls)

        # Verdadeiro Positivo: diagonal principal (A11, A22, A33, ...)
        vp = sum(cm.M(i, i) for i in range(cm.c))

        # Verdadeiro Negativo: itens da linha que não são da diagonal principal
        # (A12, A22, A32, ...) na classe 1
        vn = sum(cm.M(i, ci) for i in range(cm.c) if i != ci)

        # Falso Positivo: itens da coluna que não são da diagonal principal
        # (A12, A13, A14, ...) na classe 1
        fp = sum(cm.M(ci, i) for i in range(cm.c) if i != ci)

        # Falso Negativo: itens que não são da mesma linha, coluna ou diagonal principal
        # (A23, A24, A32, ...) na classe 1
        fn = sum(cm.M(i, j)
            for i in range(cm.c)
            for j in range(cm.c)
            if i != j and i != ci and j != ci)

        return BinaryMatrix(vp, fp, fn, vn)


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
    zk = (m1.kappa - m2.kappa) / np.sqrt(m1.kappa_var + m2.kappa_var)
    
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
    matrix = ConfusionMatrix(
        ("w1", "w2", "w3", "w4"),
        [
        [140, 20, 0, 0],
        [10, 130, 0, 0],
        [5, 0, 150, 10],
        [15, 10, 0, 120]
    ])

    print("c: ", matrix.c)
    print("A+i: ", matrix.M("+i"))
    print("Ai+: ", matrix.M("i+"))
    print("m: ", matrix.m)
    print("Ag: ", matrix.Ag)
    print("Aa: ", matrix.Aa)
    print("kappa: ", matrix.kappa, matrix.kappa_var)
    print("tau: ", matrix.tau, matrix.tau_var)

    matrix2 = ConfusionMatrix(
        ("w1", "w2", "w3", "w4"),
        [
        [140, 30, 2, 0],
        [10, 110, 5, 0],
        [0, 10, 140, 0],
        [20, 10, 3, 140]
    ])

    print("Is kappa significant on 5%?:", is_kappa_significant(matrix, matrix2, 0.05))
    print("Is tau significant on 5%?:", is_tau_significant(matrix, matrix2, 0.05))

    m2bin = BinaryMatrix.from_confusion_matrix(matrix2, "w1")
    print("\n\nMatriz binária sobre w0")
    print("verdadeiros positivos: ", m2bin.vp)
    print("verdadeiros negativos: ", m2bin.vn)
    print("falsos positivos: ", m2bin.fp)
    print("falsos negativos: ", m2bin.fn)

    print("\nPrecisão: ", m2bin.Pr)
    print("Sensibilidade: ", m2bin.Se)
    print("Especificidade: ", m2bin.Es)
    print("Acurácia: ", m2bin.Ac)

    print("\nF1: ", m2bin.F(1))
    print("F2: ", m2bin.F(2))
    print("F1/2: ", m2bin.F(1/2))

    print("Mattheus: ", m2bin.matthews())