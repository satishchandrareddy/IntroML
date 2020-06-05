# unittest_example.py
#
import unittest

class Test(unittest.TestCase):
	def test1(self):
		x = 7
		y = 8
		z1 = (x+y)*(x+y)
		z2 = x*x + 2*x*y + y*y
		error = abs(z1-z2)
		self.assertLessEqual(error,1e-7)
  
if __name__ == "__main__":
    unittest.main()