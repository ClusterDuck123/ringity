import os
import ringity
import unittest

RINGITY_PATH = os.path.dirname(ringity.__file__)

class TestReadAndWrite(unittest.TestCase):
    def test_save_and_load(self):
        pdgm1 = ringity.random_pdgm(100)
        ringity.write_pdiagram(pdgm1, "tmp/random_dgm.txt")

        pdgm2 = ringity.read_pdiagram("tmp/random_dgm.txt")
        os.remove("tmp/random_dgm.txt")
        
        self.assertEqual(pdgm1, pdgm2)

if __name__ == '__main__':
    unittest.main()
