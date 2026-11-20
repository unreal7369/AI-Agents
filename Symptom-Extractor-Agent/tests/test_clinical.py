import unittest
from clinical import extract_from_patient, aggregate

class TestExtraction(unittest.TestCase):
    def test_duration(self):
        out = extract_from_patient('I have fever for 3 days.')
        self.assertEqual(out[0]['duration'], '3 days')
    def test_negation(self):
        out = extract_from_patient('No cough.')
        self.assertEqual(out[0]['status'], 'absent')
    def test_multiple(self):
        self.assertEqual(len(aggregate(['I have fever and headache.', 'I have body pain.'])), 3)

if __name__ == '__main__':
    unittest.main()
