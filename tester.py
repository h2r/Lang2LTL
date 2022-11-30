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
            ("& a b", "(a & b)"),
            ("U ! a b", "(!(a) U b)"),
            ("-> a b", "(a -> b)"),
            ("<-> a b", "(a <-> b)"),
            ("G F X ! a", "G(F(X(!(a))))"),
            ("& F & a F b G ! c", "(F((a & F(b))) & G(!(c)))"),
            ("& F & a F b -> a U ! a b", "(F((a & F(b))) & (a -> (!(a) U b)))"),
            ("G -> b X U b & ! b U ! b a", "G((b -> X((b U (!(b) & (!(b) U a))))))"),
            ("& G -> b X U b & ! b U ! b a G -> c X U c & ! c U ! c b",
             "(G((b -> X((b U (!(b) & (!(b) U a)))))) & G((c -> X((c U (!(c) & (!(c) U b)))))))"),
        ]
        for prefix_formula, infix_formula in in_out_pairs:
            output_formula = prefix_to_infix(prefix_formula)
            self.assertEqual(output_formula, infix_formula, "incorrect output infix formula")


if __name__ == '__main__':
    unittest.main()
