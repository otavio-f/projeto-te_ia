"""
Especifica as exceções adicionais que podem ser lançadas pelo servidor.
"""

class InvalidClass(Exception):
    """Lançada quando uma classe inválida for detectada."""
    pass


class InvalidDataset(Exception):
    """Lançada quando um conjunto de dados inválido for detectada."""
    pass


class InvalidAlgorithm(Exception):
    """Lançada quando um algoritmo não é válido."""
    pass
