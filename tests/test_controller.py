import unittest
from server.controller import Controller


class ControllerTestCase(unittest.TestCase):
    """
    Classe de teste de unidade do controlador de operações
    """

    def testGenerateDataset(self):
        """
        Testa geração de datasets.
        """
        control = Controller()
        ds_id: str = control.gen_dataset(
            train_percent=.8,
            columns=["Sepal length", "Sepal width"])

        self.assertIsInstance(ds_id, str)
        self.assertEquals(8, len(ds_id))


    def testGetAllDatasets(self):
        """
        Testa listagem de datasets.
        """
        control = Controller()
        
        self.assertEquals(0, len(control.get_all_datasets()))

        control.gen_dataset(
            train_percent=.8,
            columns=["Sepal length", "Sepal width"])
        
        control.gen_dataset(
            train_percent=.8,
            columns=["Petal length", "Petal width"])

        self.assertEquals(2, len(control.get_all_datasets()))


    def testGetDataset(self):
        """
        Testa obtenção de datasets.
        """
        control = Controller()
        ds_id: str = control.gen_dataset(
            train_percent=.8,
            columns=["Sepal length", "Sepal width"])
        
        result = control.get_dataset(ds_id)

        self.assertIsNotNone(result)
        self.assertListEqual(["Sepal length", "Sepal width"], result.columns)
        self.assertListEqual(["setosa", "versicolor", "virginica"], list(result.classes))

        self.assertEquals(40, result.get_trainset("setosa").length)
        self.assertEquals(40, result.get_trainset("versicolor").length)
        self.assertEquals(40, result.get_trainset("virginica").length)

        self.assertEquals(30, len(result.test_set))

