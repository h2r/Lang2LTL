import unittest
import string

from utils import substitute_single_letter, substitute_single_word, shorten_prop, prefix_to_infix


class TestUtils(unittest.TestCase):
    """
    Test utility functions in utils.py
    """
    def test_substitute_single_letter(self):
        in_out_pairs = [
            (("go to a.", "F a"), {"a": "b"}, ("go to b", "F b")),  # single sub
            (("visit a, b and c.", "& & F a F b F c"), {"a": "b", "b": "c", "c": "a"}, ("visit b c and a", "& & F b F c F a")),  # multiple subs
            (("go to a and b", "& F a F b"), {"a": "b", "b": "a"}, ("go to b and a", "& F b F a")),  # not sub letters in words
            (("visit a then b then c, but do not visit b until a, do not visit c until b.", "& F & a F & b c & U ! b a U ! c b"),
             {"a": "b", "b": "c", "c": "a"},
             ("visit b then c then a but do not visit c until b do not visit a until c", "& F & b F & c a & U ! c b U ! a c")),  # multiple subs of same letter
            (("go to a.", "F a"), {"b": "a"}, ("go to a", "F a")),  # no effect: sub letter not in string
            (("go to a.", "F a"), {"a": "a"}, ("go to a", "F a")),  # no effect: sub same letter
            (("visit a,b, and c.", "& & F a F b F c"), {"a": "b", "b": "c", "c": "a"}, ("visit ab and a", "& & F b F c F a")),  # no space separate
            (("aabc", " "), {"a": "b"}, ("aabc", "")),  # no space separate
            # below only work with substitute_single_word
            (("finally go to green one, then green.", "F & green one green"), {"green": "green room", "green one": "green room"},
             ("finally go to green room one then green room", "F & green room one green room")),
        ]
        for idx, ((utt, ltl), sub_map, (true_utt_perm, true_ltl_perm)) in enumerate(in_out_pairs):
            utt = utt.translate(str.maketrans('', '', string.punctuation))  # remove punctuations for substitution
            utt_perm = substitute_single_letter(utt, sub_map)
            ltl_perm = substitute_single_letter(ltl, sub_map)
            self.assertEqual(utt_perm, true_utt_perm, f"incorrect output string: {idx}")
            self.assertEqual(ltl_perm, true_ltl_perm, f"incorrect output string: {idx}")

    def test_substitute_single_word(self):
        in_out_pairs = [
            ("go to green one, then green", {"green": "green room", "green one": "green room"}, "go to green room, then green room"),
            # ("go to 60 park, then stuart and george street.", {"60 park": "cafe", "stuart and george street": "main street", "0": "2"},
            #  "go to cafe, then main street."),  # number as key of sub_map
            ("go to Emerson College - Union Bank Building, then Emerson College", {"Emerson College - Union Bank Building": "building", "Emerson College": "college"},
             "go to building, then college"),  # RE is substring of another RE
            ("& U ! b a F b", {"a": "b", "b": "a"}, "& U ! a b F a"),  # multiple subs of same letter
            # below only work with substitute_single_letter
            ("aabc", {"a": "b"}, "bbbc"),
            ("go to a and b", {"a": "b", "b": "a"}, "go to b bnd a"),
        ]
        for idx, (in_str, sub_map, true_out_str) in enumerate(in_out_pairs):
            out_str, subs_done = substitute_single_word(in_str, sub_map)
            self.assertEqual(out_str, true_out_str, f"incorrect output string: {idx}\n{true_out_str}\n{out_str}")

    def test_shorten_prop(self):
        in_out_pairs = [
            ("massachusetts_state_transportation_building", "ma_st_tr_bu"),
            ("washington_st_@_kneeland_st", "wa_st_@_kn_st"),
            ("montien", "mo"),
            ("62_on_the_park", "62_on_th_pa"),
        ]
        for prop_long, true_prop_short in in_out_pairs:
            out_prop_short = shorten_prop(prop_long)
            self.assertEqual(true_prop_short, out_prop_short, "incorrect shorten prop")

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
