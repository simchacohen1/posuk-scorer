"""Offline tests for scoring.py - no Flask, no OpenAI, no API cost.

Run:  python -m unittest test_scoring.py
"""
import unittest

from scoring import (normalize_word, score_single_word, score_text,
                     strip_section_markers, words_match)


def misses(expected, heard):
    _, words = score_text(expected, heard)
    return [w for w, ok in words if not ok]


class CorrectReadingsStillPass(unittest.TestCase):
    """Whisper's spelling differs from the Chumash, but the reading is right."""

    def test_whisper_full_spelling(self):
        pairs = [
            ("אֹהֶל", "אוהל"), ("כְּחֹם", "כחום"), ("סֹלֶת", "סולת"),
            ("עֻגֹת", "עוגות"), ("אַבְרָהָם", "אברם"), ("אַבְרָהָם", "אברוהם"),
            ("וְהָאֱלֹהִים", "והאלוהים"), ("אֲדֹנָי", "אדוני"),
            ("יוֹשֵׁב", "ישב"), ("מִשְׁפָּט", "משפט"),
        ]
        for exp, heard in pairs:
            with self.subTest(exp=exp, heard=heard):
                self.assertTrue(words_match(exp, heard))

    def test_sound_alike_letters(self):
        # Ashkenazi sav, and Whisper writing ק/ט for כ/ת sounds
        for exp, heard in [("כִּי", "קי"), ("בַּיִת", "בייס"), ("תַּחַת", "טחת"),
                           ("נָא", "נה"), ("אֶל", "על"), ("הִנֵּה", "הני")]:
            with self.subTest(exp=exp, heard=heard):
                self.assertTrue(words_match(exp, heard))

    def test_the_name_in_any_form(self):
        for heard in ["השם", "אדני", "אדוני", "אדושם", "הויה"]:
            with self.subTest(heard=heard):
                self.assertTrue(words_match("ה׳", heard))
        self.assertTrue(words_match("וַה׳", "ואדושם"))
        self.assertTrue(words_match("מֵה׳", "מהשם"))

    def test_full_posuk_with_nikud_against_plain_transcript(self):
        exp = "וַיֵּרָא אֵלָיו ה׳ בְּאֵלֹנֵי מַמְרֵא וְהוּא יֹשֵׁב פֶּתַח־הָאֹהֶל כְּחֹם הַיּוֹם"
        heard = "וירא אליו השם באלוני ממרא והוא יושב פתח האוהל כחום היום"
        self.assertEqual(score_text(exp, heard)[0], 100)

    def test_two_words_written_as_one(self):
        self.assertEqual(misses("יֻקַּח נָא מְעַט מַיִם", "יוקחנא מעט מים"), [])
        self.assertEqual(misses("פֶּתַח הָאֹהֶל", "פסחהאל"), [])

    def test_hashem_misheard(self):
        # spellings Whisper produced for "Hashem" in saved attempts
        for heard in ["חשם", "השן"]:
            with self.subTest(heard=heard):
                self.assertTrue(words_match("ה׳", heard))

    def test_two_words_found_a_little_out_of_place(self):
        self.assertEqual(misses("וְנִבְרְכוּ בוֹ כֹּל גּוֹיֵי הָאָרֶץ",
                                "ונברכובו כל גויי הארץ"), [])

    def test_es_is_free(self):
        self.assertEqual(misses("אֶת בָּנָיו", "בניו"), [])

    def test_section_markers_are_not_words(self):
        self.assertEqual(strip_section_markers("וַיֹּאמֶר {ס}"), "וַיֹּאמֶר")
        self.assertEqual(score_text("הוּא אֲבִי בְנֵי עַמּוֹן עַד הַיּוֹם {ס}",
                                    "הוא אבי בני עמון עד היום")[0], 100)

    def test_extra_words_do_not_count_against(self):
        self.assertEqual(misses("וַיֹּאמֶר אֵלָיו", "אה ויאמר אליו"), [])


class WrongReadingsAreCaught(unittest.TestCase):
    """Each of these used to score 100%."""

    def test_wrong_words(self):
        pairs = [
            ("אֵלָיו", "אלי"),        # to him -> to me
            ("וַיֹּאמֶר", "ויאמרו"),   # he said -> they said
            ("וַיֵּרֶד", "וירא"),      # went down -> appeared
            ("שְׁלֹשָׁה", "שלש"),
            ("וַיֵּרָא", "ירא"),       # dropped the vav
            ("אֵלָיו", "אלה"),
            ("וַיֵּשֶׁב", "וישר"),
            ("אָשׁוּב", "שוב"),        # dropped "I will"
            ("עֲנָוִים", "עני"),
        ]
        for exp, heard in pairs:
            with self.subTest(exp=exp, heard=heard):
                self.assertFalse(words_match(exp, heard))

    def test_skipped_word(self):
        self.assertEqual(misses("צַדִּיק עִם רָשָׁע", "צדיק רשע"), ["עִם"])

    def test_short_word_not_found_inside_a_longer_one(self):
        # old fusion rule credited אליו because "אלי" sat inside a token
        self.assertEqual(misses("וַיֵּרָא אֵלָיו ה׳", "וירא אלי השם"), ["אֵלָיו"])

    def test_two_wrong_words_in_a_posuk(self):
        exp = "וַיֹּאמְרוּ אֵלָיו אַיֵּה שָׂרָה אִשְׁתֶּךָ"
        self.assertEqual(misses(exp, "ויאמר אליו איה רבקה אשתך"),
                         ["וַיֹּאמְרוּ", "שָׂרָה"])


class PronunciationAndSpellingStillPass(unittest.TestCase):
    """Correct readings that the old rules failed or passed only by luck."""

    def test_sin_heard_as_samech(self):
        for exp, heard in [("שָׂרָה", "סרה"), ("יִשְׂרָאֵל", "יסראל"),
                           ("עֵשָׂו", "עסו"), ("וַיִּשָּׂא", "ויסא"),
                           ("שָׂרָה", "שרה")]:
            with self.subTest(exp=exp, heard=heard):
                self.assertTrue(words_match(exp, heard))

    def test_elokim(self):
        for exp, heard in [("אֱלֹהִים", "אלוקים"), ("וֵאלֹהִים", "ואלוקים"),
                           ("הָאֱלֹהִים", "האלוקים"), ("אֱלֹהֶיךָ", "אלוקיך"),
                           ("אֱלֹהֵינוּ", "אלוקינו")]:
            with self.subTest(exp=exp, heard=heard):
                self.assertTrue(words_match(exp, heard))

    def test_full_or_short_spelling(self):
        for exp, heard in [("כָּל", "כול"), ("אֹתוֹ", "אותו"), ("מְאֹד", "מאוד"),
                           ("חַוָּה", "חווה"), ("רְאוּבֵן", "רובן"),
                           ("צָהֳרַיִם", "צהריים"), ("לִפְנֵי", "לפניי"),
                           ("אָבִיו", "אביב")]:
            with self.subTest(exp=exp, heard=heard):
                self.assertTrue(words_match(exp, heard))

    def test_whole_posuk_ashkenazi_reading(self):
        exp = "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ"
        self.assertEqual(score_text(exp, "בראשיס ברא אלוקים עס השמיים ואס הארץ")[0], 100)


class NearMissesAreCaught(unittest.TestCase):
    """A one-letter change in a short word is usually a different word.
    The old one-letter allowance passed all of these."""

    def test_different_short_words(self):
        pairs = [
            ("וַיִּשְׁלַח", "וישכח"),   # sent -> forgot
            ("יָדַע", "ירא"),           # knew -> feared
            ("וַיֵּלֶךְ", "וימלך"),     # went -> reigned
            ("וַיִּשְׁכַּב", "וישב"),   # lay down -> sat
            ("וַיִּזְכֹּר", "וישכר"),
            ("וַיִּשְׁלַח", "ויסלח"),   # shin is not a sin
            ("מִשְׁפָּחָה", "מספחה"),
        ]
        for exp, heard in pairs:
            with self.subTest(exp=exp, heard=heard):
                self.assertFalse(words_match(exp, heard))

    def test_changed_suffix_on_a_long_word(self):
        self.assertFalse(words_match("אֲבוֹתֵיכֶם", "אבותיהם"))   # your -> their
        self.assertFalse(words_match("אֲבוֹתֵיהֶם", "אבותיכם"))


class SilenceAndUnrelatedSpeech(unittest.TestCase):
    P3 = "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ"

    def test_silence_scores_zero(self):
        # את used to be free even when nothing was read (14%)
        self.assertEqual(score_text(self.P3, "")[0], 0)

    def test_unrelated_speech(self):
        # "umm" used to match אלהים; this scored 29%
        self.assertEqual(score_text(self.P3, "אממ אני לא יודע")[0], 0)
        self.assertEqual(score_text(self.P3, "ויאמר משה אל העם לא תיראו")[0], 0)

    def test_half_the_posuk(self):
        self.assertEqual(score_text(self.P3, "בראשית ברא אלוקים")[0], 43)  # 3 of 7

    def test_es_still_free_inside_a_reading(self):
        self.assertEqual(misses("וַיִּקַּח אַבְרָהָם אֶת הָעֵצִים",
                                "ויקח אברהם העצים"), [])


class RedoOneWord(unittest.TestCase):
    def test_es_needs_to_be_heard(self):
        self.assertFalse(score_single_word("אֶת", ""))
        self.assertFalse(score_single_word("אֶת", "שלום"))
        self.assertTrue(score_single_word("אֶת", "את"))

    def test_sin_and_elokim(self):
        self.assertTrue(score_single_word("שָׂרָה", "סרה"))
        self.assertTrue(score_single_word("אֱלֹהִים", "אלוקים"))

    def test_target_heard(self):
        self.assertTrue(score_single_word("נָא", "נה"))

    def test_neighbour_only_does_not_pass(self):
        # saying a neighbouring word from the context must not pass
        self.assertFalse(score_single_word("נָא", "יקח", context="יקח נא מעט"))

    def test_fused_with_neighbour(self):
        self.assertTrue(score_single_word("נָא", "יקחנא", context="יקח נא מעט"))

    def test_wrong_word(self):
        self.assertFalse(score_single_word("אֵלָיו", "אלי"))


class Normalization(unittest.TestCase):
    def test_nikud_and_trop_ignored(self):
        self.assertEqual(normalize_word("וַיֵּ֣רָא"), normalize_word("וירא"))

    def test_punctuation_ignored(self):
        self.assertEqual(normalize_word("הַיּֽוֹם׃"), normalize_word("היום"))


if __name__ == "__main__":
    unittest.main()
