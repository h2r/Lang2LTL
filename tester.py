import unittest

from utils import substitute_single_letter, substitute_single_word, prefix_to_infix


class TestUtils(unittest.TestCase):
    """
    Test utility functions in utils.py
    """
    def test_substitute_single_letter(self):
        in_out_pairs = [
            ("F a", {"a": "b"}, "F b"),
            ("& F a & F b F c", {"a": "b", "b": "c", "c": "a"}, "& F b & F c F a"),  # multiple subs
            ("go to a and b", {"a": "b", "b": "a"}, "go to b and a"),  # not sub letters in words
            ("you must visit a first before going to b", {"a": "b", "b": "a"}, "you must visit b first before going to a"),  # not sub letters in words
            ("F a", {"b": "a"}, "F a"),  # no effect
            ("F a", {"a": "a"}, "F a"),  # sub same letter
            ("aabc", {"a": "b"}, "aabc"),  # no space separate
            # below only work with substitute_single_word
            ("finally go to green one, then green", {"green": "green room", "green one": "green room"}, "finally go to green room one, then green room")
        ]
        for in_str, sub_map, true_out_str in in_out_pairs:
            out_str = substitute_single_letter(in_str, sub_map)
            self.assertEqual(out_str, true_out_str, "incorrect output string")

    def test_substitute_single_word(self):
        in_out_pairs = [
            ("finally go to green one, then green", {"green": "green room", "green one": "green room"}, "finally go to green room, then green room"),
            # below only work with substitute_single_letter
            ("aabc", {"a": "b"}, "bbbc"),
            ("go to a and b", {"a": "b", "b": "a"}, "go to b bnd a"),
        ]
        for in_str, sub_map, true_out_str in in_out_pairs:
            out_str = substitute_single_word(in_str, sub_map)
            self.assertEqual(out_str, true_out_str, "incorrect output string")

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
