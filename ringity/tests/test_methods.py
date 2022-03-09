import os
import ringity
import unittest

RINGITY_PATH = os.path.dirname(ringity.__file__)

class TestReadAndWrite(unittest.TestCase):
    def test_save_and_load(self):
        dgm1 = ringity.random_pdgm(100)
        ringity.write_pdiagram(dgm1, "tmp/random_dgm.txt")
        dgm2 = ringity.read_pdiagram("tmp/random_dgm.txt")
        self.assertEqual(dgm1,dgm2)
        os.remove("tmp/random_dgm.txt")

if __name__ == '__main__':
    unittest.main()
