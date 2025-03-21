"""
Módulo vector
:author: Otávio Ferreira
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Vector(object):
    """
    Define um vetor de números inteiros ou de ponto flutuante
    """
    __data: [float, ...]
    __title: str = ""

    @property
    def title(self) -> 'String':
        """
        O título do vetor
        """
        return self.__title

    @property
    def T(self) -> 'Vector':
        """
        Vetor transposto.
        :returns: A transposição desse vetor.
        """
        return Vector(np.transpose(self.__data).tolist())

    @property
    def m(self) -> float:
        """
        Calcula a média.
        :returns: A média dos valores desse conjunto.
        """
        return float(np.mean(self.__data))

    def augment(self, end: int) -> 'Vector':
        """
        Retorna a versão aumentada desse vetor
        :param end: Valor a ser adicionado ao final
        :returns: Esse mesmo vetor com um zero ao final
        """
        return Vector(list(self.__data)+[end])

    def __len__(self) -> int:
        """
        Calcula o tamanho do vetor.
        :returns: A quantidade de itens do vetor.
        """
        return len(self.__data)

    def __repr__(self) -> str:
        """
        Converte os dados dessa instância pra uma versão legível
        :returns: Uma lista com os elementos desse vetor.
        """
        return self.__str__()
    
    def __str__(self) -> str:
        """
        Converte os dados dessa instância pra uma versão legível
        :returns: Uma lista com os elementos desse vetor.
        """
        if len(self) < 5:
            vect = ", ".join(str(x) for x in self.__data)
        else:
            vect = ", ".join(str(x) for x in list(self.__data)[:5]) + ", ..."
        return f"<{vect}>"
        # return f"<Vector \"{self.title}\" with {len(self)} items: {vect}>"

    def __iter__(self):
        """
        Itera sobre os elementos.
        :returns: Um iterador.
        """
        for x in self.__data:
            yield x
    
    def __eq__(self, another: 'Vector | float') -> bool:
        """
        Compara essa instância a outra instância de lista ou vetor.
        :returns: True se os elementos forem os mesmos
        """
        try:
            return all(x==y for x,y in zip(self.__data, another))
        except:
            return False

    def __add__(self, another: 'Vector | float') -> 'Vector':
        """
        Faz a adição desse vetor a um número ou outro vetor.
        Caso o outro parâmetro for um vetor, faz a adição termo a termo, retornando outro vetor.
        Caso o parâmetro for um escalar, adiciona o escalar a todos os termos.
        :param another: Um vetor ou escalar a ser adicionado
        :returns: Um novo vetor com o resultado da soma
        """
        try:
            return Vector.of(x + y for x, y in zip(self.__data, another))
        except TypeError:
            return Vector.of(x + another for x in self.__data)

    def __radd__(self, another: 'Vector | float') -> 'Vector':
        """
        Faz a adição desse vetor a um número ou outro vetor.
        Caso o outro parâmetro for um vetor, faz a adição termo a termo, retornando outro vetor.
        Caso o parâmetro for um escalar, adiciona o escalar a todos os termos.
        :param another: Um vetor ou escalar a ser adicionado
        :returns: Um novo vetor com o resultado da soma
        """
        return self + another
    
    def __sub__(self, another: 'Vector | float') -> 'Vector':
        """
        Subtrai um número ou outro vetor desse vetor.
        Caso o outro parâmetro for um vetor, faz a subtração termo a termo, retornando outro vetor.
        Caso o parâmetro for um escalar, adiciona o escalar a todos os termos.
        :param another: Um vetor ou escalar a ser subtraído
        :returns: Um novo vetor com o resultado da subtração
        """
        try:
            return Vector.of(x - y for x, y in zip(self.__data, another))
        except TypeError:
            return Vector.of(x - another for x in self.__data)

    def __rsub__(self, another: 'Vector | float') -> 'Vector':
        """
        Subtrai esse vetor de um número ou outro vetor.
        Caso o outro parâmetro for um vetor, faz a subtração termo a termo, retornando outro vetor.
        Caso o parâmetro for um escalar, adiciona o escalar a todos os termos.
        :param another: Um vetor ou escalar a ser subtraído
        :returns: Um novo vetor com o resultado da subtração
        """
        return -self + another

    def __mul__(self, another: 'Vector | float') -> 'float | Vector':
        """
        Multiplica esse vetor por um número ou outro vetor.
        Caso o outro parâmetro for um vetor, faz o produto vetorial, retornando um escalar.
        Caso o parâmetro for um escalar, multiplica o escalar a todos os termos.
        :param another: Um vetor ou escalar a ser multiplicado
        :returns: Um novo vetor ou um escalar
        """
        try:
            result = np.dot(self.__data, list(another))
            return float(result)
        except TypeError:
            result = np.dot(self.__data, another)
            return Vector.of(result.tolist())
    
    def __rmul__(self, another: 'Vector | float') -> 'float | Vector':
        """
        Multiplica esse vetor por um número ou outro vetor.
        Caso o outro parâmetro for um vetor, faz o produto vetorial, retornando um escalar.
        Caso o parâmetro for um escalar, multiplica o escalar a todos os termos.
        :param another: Um vetor ou escalar a ser multiplicado
        :returns: Um novo vetor
        """
        return self * another

    def __neg__(self) -> 'Vector':
        """
        Inverte o sinal de todos os elementos desse vetor.
        :returns: Um novo vetor com o sinal invertido em todos os elementos
        """
        return Vector.of(-x for x in self.__data)

    def of(*items: 'Generator|list|tuple') -> 'Vector':
        """
        Gera um vetor a partir de um conjunto de elementos.
        :param *items: Uma expressão geradora, uma lista ou tupla de elementos.
        """
        if len(items) == 1:
            itemlist = tuple(items[0])
        else:
            itemlist = tuple(items)
        return Vector(itemlist)


class Equation(object):
    def __init__(self, *terms):
        self.__terms = terms[:-1]
        self.__indep = terms[-1]

    def __get_term(self, index: int) -> str:
        val = self.__terms[index]
        sign = "-" if val < 0 else "+"
        term = "" if abs(val) == 1 else f"{abs(val)}"

        if index == 0:
            return f"{sign}{term}x{index+1}"
        return f"{sign} {term}x{index+1}"
        
    def __call__(self, *terms) -> float:
        if len(terms) != len(self.__terms):
            raise ValueError("Equation terms mismatch!")
        return sum(k*x for k,x in zip(self.__terms, terms)) + self.__indep
    
    def __str__(self) -> str:
        if len(self.__terms) == 0:
            return str(self.__indep)
        result = " ".join(self.__get_term(i) for i in range(len(self.__terms)))
        if self.__indep != 0:
            result += " - " if self.__indep < 0 else " + "
            result += str(self.__indep)
        return result
    
    def __repr__(self) -> str:
        return self.__str__()
