from dataclasses import dataclass
from dataclasses import field
from statistics import NormalDist
import math


class ConfusionMatrix(object):
    """
    Matriz de confusão.
    """
    def __init__(self, matrix: list[list[int]]):
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
        Valores terminados em "+" retornam o acumulado parcial da linha, ex.: "5+"
        Valores iniciados em "+" retornam o acumulado parcial da coluna, ex.: "+1"
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