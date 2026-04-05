# reference: https://github.com/ORI-Muchim/MB-iSTFT-VITS-Korean/blob/main/text/korean.py

import re
from jamo import h2j, j2hcj
import ko_pron
from g2pk2 import G2p

import importlib
import os

# ?≤Ê?win‰∏ãÊó†Ê≥ïË??ñÊ®°??
if os.name == "nt":

    class win_G2p(G2p):
        def check_mecab(self):
            super().check_mecab()
            spam_spec = importlib.util.find_spec("eunjeon")
            non_found = spam_spec is None
            if non_found:
                print("you have to install eunjeon. install it...")
            else:
                installpath = spam_spec.submodule_search_locations[0]
                if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", installpath)):
                    import sys
                    from eunjeon import Mecab as _Mecab

                    class Mecab(_Mecab):
                        def get_dicpath(installpath):
                            if not (re.match(r"^[A-Za-z0-9_/\\:.\-]*$", installpath)):
                                import shutil

                                python_dir = os.getcwd()
                                if installpath[: len(python_dir)].upper() == python_dir.upper():
                                    dicpath = os.path.join(os.path.relpath(installpath, python_dir), "data", "mecabrc")
                                else:
                                    if not os.path.exists("TEMP"):
                                        os.mkdir("TEMP")
                                    if not os.path.exists(os.path.join("TEMP", "ko")):
                                        os.mkdir(os.path.join("TEMP", "ko"))
                                    if os.path.exists(os.path.join("TEMP", "ko", "ko_dict")):
                                        shutil.rmtree(os.path.join("TEMP", "ko", "ko_dict"))

                                    shutil.copytree(
                                        os.path.join(installpath, "data"), os.path.join("TEMP", "ko", "ko_dict")
                                    )
                                    dicpath = os.path.join("TEMP", "ko", "ko_dict", "mecabrc")
                            else:
                                dicpath = os.path.abspath(os.path.join(installpath, "data/mecabrc"))
                            return dicpath

                        def __init__(self, dicpath=get_dicpath(installpath)):
                            super().__init__(dicpath=dicpath)

                    sys.modules["eunjeon"].Mecab = Mecab

    G2p = win_G2p


from text.symbols2 import symbols

# This is a list of Korean classifiers preceded by pure Korean numerals.
_korean_classifiers = (
    "Íµ∞Îç∞ Í∂?Í∞?Í∑∏Î£® ???Ä ??ÎßàÎ¶¨ Î™?Î™®Í∏à Î≠?Î∞?Î∞úÏßù Î∞?Î≤?Î≤?Î≥¥Î£® ???????????Ä????Ïß?Ï±?Ï≤?Ï≤?Ï∂?Ïº§Î†à ????
)

# List of (hangul, hangul divided) pairs:
_hangul_divided = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        # ('??, '?±„ÖÖ'),   # g2pk2, A Syllable-ending Rule
        # ('??, '?¥„Öà'),
        # ('??, '?¥„Öé'),
        # ('??, '?π„Ñ±'),
        # ('??, '?π„ÖÅ'),
        # ('??, '?π„ÖÇ'),
        # ('??, '?π„ÖÖ'),
        # ('??, '?π„Öå'),
        # ('??, '?π„Öç'),
        # ('?Ä', '?π„Öé'),
        # ('??, '?Ç„ÖÖ'),
        ("??, "?ó„Öè"),
        ("??, "?ó„Öê"),
        ("??, "?ó„Ö£"),
        ("??, "?ú„Öì"),
        ("??, "?ú„Öî"),
        ("??, "?ú„Ö£"),
        ("??, "?°„Ö£"),
        ("??, "?£„Öè"),
        ("??, "?£„Öê"),
        ("??, "?£„Öì"),
        ("??, "?£„Öî"),
        ("??, "?£„Öó"),
        ("??, "?£„Öú"),
    ]
]

# List of (Latin alphabet, hangul) pairs:
_latin_to_hangul = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("a", "?êÏù¥"),
        ("b", "Îπ?),
        ("c", "??),
        ("d", "??),
        ("e", "??),
        ("f", "?êÌîÑ"),
        ("g", "ÏßÄ"),
        ("h", "?êÏù¥Ïπ?),
        ("i", "?ÑÏù¥"),
        ("j", "?úÏù¥"),
        ("k", "ÏºÄ??),
        ("l", "??),
        ("m", "??),
        ("n", "??),
        ("o", "??),
        ("p", "??),
        ("q", "??),
        ("r", "?ÑÎ•¥"),
        ("s", "?êÏä§"),
        ("t", "??),
        ("u", "??),
        ("v", "Î∏åÏù¥"),
        ("w", "?îÎ∏î??),
        ("x", "?ëÏä§"),
        ("y", "?Ä??),
        ("z", "?úÌä∏"),
    ]
]

# List of (ipa, lazy ipa) pairs:
_ipa_to_lazy_ipa = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("tÕ°?", " ß"),
        ("dÕ°?", " •"),
        ("…≤", "n^"),
        ("?", "?"),
        (" ∑", "w"),
        ("…≠", "l`"),
        ("?", "…æ"),
        ("…£", "≈ã"),
        ("…∞", "…Ø"),
        ("?", "j"),
        ("?", "?"),
        ("…°", "g"),
        ("\u031a", "#"),
        ("\u0348", "="),
        ("\u031e", ""),
        ("\u0320", ""),
        ("\u0339", ""),
    ]
]


def fix_g2pk2_error(text):
    new_text = ""
    i = 0
    while i < len(text) - 4:
        if (text[i : i + 3] == "?á„Ö°?? or text[i : i + 3] == "?π„Ö°??) and text[i + 3] == " " and text[i + 4] == "??:
            new_text += text[i : i + 3] + " " + "??
            i += 5
        else:
            new_text += text[i]
            i += 1

    new_text += text[i:]
    return new_text


def latin_to_hangul(text):
    for regex, replacement in _latin_to_hangul:
        text = re.sub(regex, replacement, text)
    return text


def divide_hangul(text):
    text = j2hcj(h2j(text))
    for regex, replacement in _hangul_divided:
        text = re.sub(regex, replacement, text)
    return text


def hangul_number(num, sino=True):
    """Reference https://github.com/Kyubyong/g2pK"""
    num = re.sub(",", "", num)

    if num == "0":
        return "??
    if not sino and num == "20":
        return "?§Î¨¥"

    digits = "123456789"
    names = "?ºÏù¥?ºÏÇ¨?§Ïú°Ïπ†ÌåîÍµ?
    digit2name = {d: n for d, n in zip(digits, names)}

    modifiers = "?????????§ÏÑØ ?¨ÏÑØ ?ºÍ≥± ?¨Îçü ?ÑÌôâ"
    decimals = "???§Î¨º ?úÎ•∏ ÎßàÌùî ???àÏàú ?ºÌùî ?¨Îì† ?ÑÌùî"
    digit2mod = {d: mod for d, mod in zip(digits, modifiers.split())}
    digit2dec = {d: dec for d, dec in zip(digits, decimals.split())}

    spelledout = []
    for i, digit in enumerate(num):
        i = len(num) - i - 1
        if sino:
            if i == 0:
                name = digit2name.get(digit, "")
            elif i == 1:
                name = digit2name.get(digit, "") + "??
                name = name.replace("?ºÏã≠", "??)
        else:
            if i == 0:
                name = digit2mod.get(digit, "")
            elif i == 1:
                name = digit2dec.get(digit, "")
        if digit == "0":
            if i % 4 == 0:
                last_three = spelledout[-min(3, len(spelledout)) :]
                if "".join(last_three) == "":
                    spelledout.append("")
                    continue
            else:
                spelledout.append("")
                continue
        if i == 2:
            name = digit2name.get(digit, "") + "Î∞?
            name = name.replace("?ºÎ∞±", "Î∞?)
        elif i == 3:
            name = digit2name.get(digit, "") + "Ï≤?
            name = name.replace("?ºÏ≤ú", "Ï≤?)
        elif i == 4:
            name = digit2name.get(digit, "") + "Îß?
            name = name.replace("?ºÎßå", "Îß?)
        elif i == 5:
            name = digit2name.get(digit, "") + "??
            name = name.replace("?ºÏã≠", "??)
        elif i == 6:
            name = digit2name.get(digit, "") + "Î∞?
            name = name.replace("?ºÎ∞±", "Î∞?)
        elif i == 7:
            name = digit2name.get(digit, "") + "Ï≤?
            name = name.replace("?ºÏ≤ú", "Ï≤?)
        elif i == 8:
            name = digit2name.get(digit, "") + "??
        elif i == 9:
            name = digit2name.get(digit, "") + "??
        elif i == 10:
            name = digit2name.get(digit, "") + "Î∞?
        elif i == 11:
            name = digit2name.get(digit, "") + "Ï≤?
        elif i == 12:
            name = digit2name.get(digit, "") + "Ï°?
        elif i == 13:
            name = digit2name.get(digit, "") + "??
        elif i == 14:
            name = digit2name.get(digit, "") + "Î∞?
        elif i == 15:
            name = digit2name.get(digit, "") + "Ï≤?
        spelledout.append(name)
    return "".join(elem for elem in spelledout)


def number_to_hangul(text):
    """Reference https://github.com/Kyubyong/g2pK"""
    tokens = set(re.findall(r"(\d[\d,]*)([\uac00-\ud71f]+)", text))
    for token in tokens:
        num, classifier = token
        if classifier[:2] in _korean_classifiers or classifier[0] in _korean_classifiers:
            spelledout = hangul_number(num, sino=False)
        else:
            spelledout = hangul_number(num, sino=True)
        text = text.replace(f"{num}{classifier}", f"{spelledout}{classifier}")
    # digit by digit for remaining digits
    digits = "0123456789"
    names = "?ÅÏùº?¥ÏÇº?¨Ïò§?°Ïπ†?îÍµ¨"
    for d, n in zip(digits, names):
        text = text.replace(d, n)
    return text


def korean_to_lazy_ipa(text):
    text = latin_to_hangul(text)
    text = number_to_hangul(text)
    text = re.sub("[\uac00-\ud7af]+", lambda x: ko_pron.romanise(x.group(0), "ipa").split("] ~ [")[0], text)
    for regex, replacement in _ipa_to_lazy_ipa:
        text = re.sub(regex, replacement, text)
    return text


_g2p = G2p()


def korean_to_ipa(text):
    text = latin_to_hangul(text)
    text = number_to_hangul(text)
    text = _g2p(text)
    text = fix_g2pk2_error(text)
    text = korean_to_lazy_ipa(text)
    return text.replace(" ß", "t?").replace(" •", "d?")


def post_replace_ph(ph):
    rep_map = {
        "Ôº?: ",",
        "Ôº?: ",",
        "Ôº?: ",",
        "??: ".",
        "Ôº?: "!",
        "Ôº?: "?",
        "\n": ".",
        "¬∑": ",",
        "??: ",",
        "...": "??,
        " ": "Á©?,
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "??
    return ph


def g2p(text):
    text = latin_to_hangul(text)
    text = _g2p(text)
    text = divide_hangul(text)
    text = fix_g2pk2_error(text)
    text = re.sub(r"([\u3131-\u3163])$", r"\1.", text)
    # text = "".join([post_replace_ph(i) for i in text])
    text = [post_replace_ph(i) for i in text]
    return text


if __name__ == "__main__":
    text = "?àÎÖï?òÏÑ∏??
    print(g2p(text))
