import unittest

from utils import prefix_to_infix


class TestUtils(unittest.TestCase):
    """
    Test utility functions in utils.py
    """
    def test_infix_to_prefix(self):
        in_out_pairs = [
            ("a", "a"),
            ("! a", "!(a)"),
            ("& a b", "(a&b)"),
            ("U ! a b", "(!(a)Ub)"),
            ("-> a b", "(a->b)"),
            ("<-> a b", "(a<->b)"),
            ("G F X ! a", "G(F(X(!(a))))"),
            ("& F & a F b G ! c", "(F((a&F(b)))&G(!(c)))"),
            ("& F & a F b -> a U ! a b", "(F((a&F(b)))&(a->(!(a)Ub)))")
        ]
        for prefix_formula, infix_formula in in_out_pairs:
            output_formula = prefix_to_infix(prefix_formula)
            self.assertEqual(output_formula, infix_formula, "incorrect output infix formula")


if __name__ == '__main__':
    unittest.main()
