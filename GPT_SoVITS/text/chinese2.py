import os
import re

import cn2an
from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

from text.symbols import punctuation
from text.tone_sandhi import ToneSandhi
from text.zh_normalization.text_normlization import TextNormalizer

normalizer = lambda x: cn2an.transform(x, "an2cn")

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba
import logging

jieba.setLogLevel(logging.CRITICAL)
import jieba.posseg as psg

# is_g2pw_str = os.environ.get("is_g2pw", "True")##默�?开??
# is_g2pw = False#True if is_g2pw_str.lower() == 'true' else False
is_g2pw = True  # True if is_g2pw_str.lower() == 'true' else False
if is_g2pw:
    # print("当前使用g2pw进行?�音?�理")
    from text.g2pw import G2PWPinyin, correct_pronunciation

    parent_directory = os.path.dirname(current_file_path)
    g2pw = G2PWPinyin(
        model_dir="GPT_SoVITS/text/G2PWModel",
        model_source=os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
        v_to_u=False,
        neutral_tone_with_five=True,
    )

rep_map = {
    "�?: ",",
    "�?: ",",
    "�?: ",",
    "??: ".",
    "�?: "!",
    "�?: "?",
    "\n": ".",
    "·": ",",
    "??: ",",
    "...": "??,
    "$": ".",
    "/": ",",
    "??: "-",
    "~": "??,
    "�?: "??,
}

tone_modifier = ToneSandhi()


def replace_punctuation(text):
    text = text.replace("??, "??).replace("??, "�?)
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text)

    return replaced_text


def g2p(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, word2ph = _g2p(sentences)
    return phones, word2ph


def _get_initials_finals(word):
    initials = []
    finals = []

    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)

    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


must_erhua = {"小院??, "?�同??, "?�儿", "?�汉??, "?�欢??, "寻老礼??, "妥妥??, "媳妇??}
not_erhua = {
    "?�儿",
    "为儿",
    "?�儿",
    "?�儿",
    "?�儿",
    "?�儿",
    "?�儿",
    "一??,
    "?�儿",
    "俺儿",
    "妻儿",
    "?�儿",
    "?�儿",
    "乞儿",
    "?�儿",
    "幼儿",
    "孤儿",
    "婴儿",
    "婴幼??,
    "连体??,
    "?�瘫??,
    "流浪??,
    "体弱??,
    "混�???,
    "?�雪??,
    "?�儿",
    "祖儿",
    "美儿",
    "应采??,
    "??��",
    "侄儿",
    "孙儿",
    "侄孙??,
    "女儿",
    "?�儿",
    "红�???,
    "?�儿",
    "?�儿",
    "马儿",
    "鸟儿",
    "?�儿",
    "?�儿",
    "?�儿",
    "少儿",
}


def _merge_erhua(initials: list[str], finals: list[str], word: str, pos: str) -> list[list[str]]:
    """
    Do erhub.
    """
    # fix er1
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "?? and phn == "er1":
            finals[i] = "er2"

    # ?�音
    if word not in must_erhua and (word in not_erhua or pos in {"a", "j", "nr"}):
        return initials, finals

    # "?��? 等情?�直?�返??
    if len(finals) != len(word):
        return initials, finals

    assert len(finals) == len(word)

    # 与前一个字?�同??
    new_initials = []
    new_finals = []
    for i, phn in enumerate(finals):
        if (
            i == len(finals) - 1
            and word[i] == "??
            and phn in {"er2", "er5"}
            and word[-2:] not in not_erhua
            and new_finals
        ):
            phn = "er" + new_finals[-1][-1]

        new_initials.append(initials[i])
        new_finals.append(phn)

    return new_initials, new_finals


def _g2p(segments):
    phones_list = []
    word2ph = []
    for seg in segments:
        pinyins = []
        # Replace all English words in the sentence
        seg = re.sub("[a-zA-Z]+", "", seg)
        seg_cut = psg.lcut(seg)
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        initials = []
        finals = []

        if not is_g2pw:
            for word, pos in seg_cut:
                if pos == "eng":
                    continue
                sub_initials, sub_finals = _get_initials_finals(word)
                sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
                # ?�化
                sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)
                initials.append(sub_initials)
                finals.append(sub_finals)
                # assert len(sub_initials) == len(sub_finals) == len(word)
            initials = sum(initials, [])
            finals = sum(finals, [])
            print("pypinyin结果", initials, finals)
        else:
            # g2pw?�用?�句?�理
            pinyins = g2pw.lazy_pinyin(seg, neutral_tone_with_five=True, style=Style.TONE3)

            pre_word_length = 0
            for word, pos in seg_cut:
                sub_initials = []
                sub_finals = []
                now_word_length = pre_word_length + len(word)

                if pos == "eng":
                    pre_word_length = now_word_length
                    continue

                word_pinyins = pinyins[pre_word_length:now_word_length]

                # 多音字消�?
                word_pinyins = correct_pronunciation(word, word_pinyins)

                for pinyin in word_pinyins:
                    if pinyin[0].isalpha():
                        sub_initials.append(to_initials(pinyin))
                        sub_finals.append(to_finals_tone3(pinyin, neutral_tone_with_five=True))
                    else:
                        sub_initials.append(pinyin)
                        sub_finals.append(pinyin)

                pre_word_length = now_word_length
                sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
                # ?�化
                sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)
                initials.append(sub_initials)
                finals.append(sub_finals)

            initials = sum(initials, [])
            finals = sum(finals, [])
            # print("g2pw结果",initials,finals)

        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c == v:
                assert c in punctuation
                phone = [c]
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    # 多音??
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # ?�音??
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                new_c, new_v = pinyin_to_symbol_map[pinyin].split(" ")
                new_v = new_v + tone
                phone = [new_c, new_v]
                word2ph.append(len(phone))

            phones_list += phone
    return phones_list, word2ph


def replace_punctuation_with_en(text):
    text = text.replace("??, "??).replace("??, "�?)
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(r"[^\u4e00-\u9fa5A-Za-z" + "".join(punctuation) + r"]+", "", replaced_text)

    return replaced_text


def replace_consecutive_punctuation(text):
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def text_normalize(text):
    # https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation(sentence)

    # ?�免?�复?�点引起?�参?�泄??
    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text


if __name__ == "__main__":
    text = "?�——但??��原神》是??米哈\游自主，?�发?��?款全.?��??�世???�险游戏"
    text = "?�呣?�～就是??��人的鼹鼠?�吧�?
    text = "你�?"
    text = text_normalize(text)
    print(g2p(text))


# # 示例?�法
# text = "这是一个示例文?�：,你�?！这???个测�?.."
# print(g2p_paddle(text))  # 输出: 这是一个示例文?�你好这???个测�?
