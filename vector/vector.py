"""
Módulo vector
:author: Otávio Ferreira
"""

import numpy as np
from dataclasses import dataclass
from dataclasses import field


class Vector(object):
    """
    Define um vetor de números inteiros ou de ponto flutuante
    """

    def __init__(self, arr: list, title: str=""):
        """
        Construtor
        :param arr: O array de dados
        :param title: Um nome que descreve os dados
        """
        self.__data__ = np.array(arr)
        self.__title = title
        
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
        return Vector(self.__data__.T)

    @property
    def m(self) -> float:
        """
        Calcula a média.
        :returns: A média dos valores desse conjunto.
        """
        return np.mean(self.__data__)

    def augment(self, end: int) -> 'Vector':
        """
        Retorna a versão aumentada desse vetor
        :param end: Valor a ser adicionado ao final
        :returns: Esse mesmo vetor com um zero ao final
        """
        return Vector(list(self)+[end])

    def __len__(self) -> int:
        """
        Calcula o tamanho do vetor.
        :returns: A quantidade de itens do vetor.
        """
        return len(self.__data__)

    def __str__(self):
        """
        Converte os dados dessa instância pra uma versão legível
        :returns: Uma lista com os elementos desse vetor.
        """
        return str(self.__data__)
    
    def __repr__(self) -> str:
        if len(self) < 5:
            vect = ", ".join(float(x) for x in self.__data__)
        else:
            vect = ", ".join(float(x) for x in list(self.__data__)[:5]) + ", ..."
        return f"<{vect}>"
        # return f"<Vector \"{self.title}\" with {len(self)} items: {vect}>"

    def __iter__(self):
        """
        Itera sobre os elementos.
        :returns: Um iterador.
        """
        for i in self.__data__:
            yield float(i)
        # return self.__data__.__iter__()
    
    def __eq__(self, another: 'Vector | float') -> bool:
        """
        Compara essa instância a outra instância de lista ou vetor.
        :returns: True se os elementos forem os mesmos
        """
        try:
            return all(self.__data__ == tuple(another))
        except:
            return False

    def __add__(self, another: 'Vector | float') -> 'Vector':
        """
        Adiciona esse vetor a um número ou outro vetor
        :param another: Um vetor ou escalar a ser adicionado
        :returns: Um novo vetor com o resultado da soma
        """
        try:
            return Vector(self.__data__ + tuple(another))
        except TypeError:
            return Vector(self.__data__ + another)

    def __radd__(self, another: 'Vector | float') -> 'Vector':
        """
        Adiciona um número ou outro vetor a esse vetor
        :param another: Um vetor ou escalar a ser adicionado
        :returns: Um novo vetor com o resultado da soma
        """
        return self + another
    
    def __sub__(self, another: 'Vector | float') -> 'Vector':
        """
        Subtrai um número ou outro vetor desse vetor
        :param another: Um vetor ou escalar a ser subtraído
        :returns: Um novo vetor com o resultado da subtração
        """
        try:
            return Vector(self.__data__ - tuple(another))
        except TypeError:
            return Vector(self.__data__ - another)

    def __rsub__(self, another: 'Vector | float') -> 'Vector':
        """
        Subtrai esse vetor de um número ou outro vetor
        :param another: Um vetor ou escalar a ser subtraído
        :returns: Um novo vetor com o resultado da subtração
        """
        return -self + another

    def __mul__(self, another: 'Vector | float') -> 'float | Vector':
        """
        Multiplica esse vetor por um número ou outro vetor
        :param another: Um vetor ou escalar a ser multiplicado
        :returns: Um novo vetor ou um escalar
        """
        try:
            return self.__data__.dot(tuple(another))
        except TypeError: # can't be converted to list
            result = self.__data__.dot(another)
            return Vector(result)
    
    def __rmul__(self, another: 'Vector | float') -> 'float | Vector':
        """
        Multiplica esse vetor por um número ou outro vetor
        :param another: Um vetor ou escalar a ser multiplicado
        :returns: Um novo vetor
        """
        return self * another

    def __neg__(self) -> 'Vector':
        """
        Inverte o sinal de todos os elementos desse vetor
        :returns: Um novo vetor com o sinal invertido em todos os elementos
        """
        return Vector(self.__data__ * -1)
