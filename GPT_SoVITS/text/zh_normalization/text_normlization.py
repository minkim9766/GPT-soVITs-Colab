# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import List

from .char_convert import tranditional_to_simplified
from .chronology import RE_DATE
from .chronology import RE_DATE2
from .chronology import RE_TIME
from .chronology import RE_TIME_RANGE
from .chronology import replace_date
from .chronology import replace_date2
from .chronology import replace_time
from .constants import F2H_ASCII_LETTERS
from .constants import F2H_DIGITS
from .constants import F2H_SPACE
from .num import RE_VERSION_NUM
from .num import RE_DECIMAL_NUM
from .num import RE_DEFAULT_NUM
from .num import RE_FRAC
from .num import RE_INTEGER
from .num import RE_NUMBER
from .num import RE_PERCENTAGE
from .num import RE_POSITIVE_QUANTIFIERS
from .num import RE_RANGE
from .num import RE_TO_RANGE
from .num import RE_ASMD
from .num import RE_POWER
from .num import replace_vrsion_num
from .num import replace_default_num
from .num import replace_frac
from .num import replace_negative_num
from .num import replace_number
from .num import replace_percentage
from .num import replace_positive_quantifier
from .num import replace_range
from .num import replace_to_range
from .num import replace_asmd
from .num import replace_power
from .phonecode import RE_MOBILE_PHONE
from .phonecode import RE_NATIONAL_UNIFORM_NUMBER
from .phonecode import RE_TELEPHONE
from .phonecode import replace_mobile
from .phonecode import replace_phone
from .quantifier import RE_TEMPERATURE
from .quantifier import replace_measure
from .quantifier import replace_temperature


class TextNormalizer:
    def __init__(self):
        self.SENTENCE_SPLITOR = re.compile(r"([ï¼šã€ï¼Œï¼›ã€‚ï¼Ÿï¼?;?!][?â€??)")

    def _split(self, text: str, lang="zh") -> List[str]:
        """Split long text into sentences with sentence-splitting punctuations.
        Args:
            text (str): The input text.
        Returns:
            List[str]: Sentences.
        """
        # Only for pure Chinese here
        if lang == "zh":
            text = text.replace(" ", "")
            # è¿‡æ»¤?‰ç‰¹æ®Šå­—ç¬?
            text = re.sub(r"[?”â€”ã€Šã€‹ã€ã€?>{}()ï¼ˆï¼‰#&@?œâ€?_|\\]", "", text)
        text = self.SENTENCE_SPLITOR.sub(r"\1\n", text)
        text = text.strip()
        sentences = [sentence.strip() for sentence in re.split(r"\n+", text)]
        return sentences

    def _post_replace(self, sentence: str) -> str:
        sentence = sentence.replace("/", "æ¯?)
        # sentence = sentence.replace('~', '??)
        # sentence = sentence.replace('ï½?, '??)
        sentence = sentence.replace("??, "ä¸€")
        sentence = sentence.replace("??, "äº?)
        sentence = sentence.replace("??, "ä¸?)
        sentence = sentence.replace("??, "??)
        sentence = sentence.replace("??, "äº?)
        sentence = sentence.replace("??, "??)
        sentence = sentence.replace("??, "ä¸?)
        sentence = sentence.replace("??, "??)
        sentence = sentence.replace("??, "ä¹?)
        sentence = sentence.replace("??, "??)
        sentence = sentence.replace("Î±", "?¿å°”æ³?)
        sentence = sentence.replace("Î²", "è´å¡”")
        sentence = sentence.replace("Î³", "ä¼½ç›").replace("?", "ä¼½ç›")
        sentence = sentence.replace("Î´", "å¾·å°”å¡?).replace("?", "å¾·å°”å¡?)
        sentence = sentence.replace("Îµ", "?¾æ™®è¥¿é¾™")
        sentence = sentence.replace("Î¶", "?·å¡”")
        sentence = sentence.replace("Î·", "ä¾å¡”")
        sentence = sentence.replace("Î¸", "è¥¿å¡”").replace("?", "è¥¿å¡”")
        sentence = sentence.replace("Î¹", "?¾æ¬§å¡?)
        sentence = sentence.replace("Îº", "?€å¸?)
        sentence = sentence.replace("Î»", "?‰å§†è¾?).replace("?", "?‰å§†è¾?)
        sentence = sentence.replace("Î¼", "ç¼?)
        sentence = sentence.replace("Î½", "??)
        sentence = sentence.replace("Î¾", "?‹è?").replace("?", "?‹è?")
        sentence = sentence.replace("Î¿", "æ¬§ç±³?‹ä¼¦")
        sentence = sentence.replace("?", "æ´?).replace("?", "æ´?)
        sentence = sentence.replace("?", "??)
        sentence = sentence.replace("?", "è¥¿æ ¼??).replace("Î£", "è¥¿æ ¼??).replace("?", "è¥¿æ ¼??)
        sentence = sentence.replace("?", "å¥?)
        sentence = sentence.replace("?", "å®‡æ™®è¥¿é¾™")
        sentence = sentence.replace("?", "?è‰¾").replace("Î¦", "?è‰¾")
        sentence = sentence.replace("?", "??)
        sentence = sentence.replace("?", "??µ›").replace("Î¨", "??µ›")
        sentence = sentence.replace("?", "æ¬§ç±³ä¼?).replace("Î©", "æ¬§ç±³ä¼?)
        # ?œåº•?°å?è¿ç®—ï¼Œé¡ºä¾¿å…¼å®¹æ‡’äººç”¨è¯?
        sentence = sentence.replace("+", "??)
        sentence = sentence.replace("-", "??)
        sentence = sentence.replace("Ã—", "ä¹?)
        sentence = sentence.replace("Ã·", "??)
        sentence = sentence.replace("=", "ç­?)
        # re filter special characters, have one more character "-" than line 68
        sentence = re.sub(r"[-?”â€”ã€Šã€‹ã€ã€?=>{}()ï¼ˆï¼‰#&@?œâ€?_|\\]", "", sentence)
        return sentence

    def normalize_sentence(self, sentence: str) -> str:
        # basic character conversions
        sentence = tranditional_to_simplified(sentence)
        sentence = sentence.translate(F2H_ASCII_LETTERS).translate(F2H_DIGITS).translate(F2H_SPACE)

        # number related NSW verbalization
        sentence = RE_DATE.sub(replace_date, sentence)
        sentence = RE_DATE2.sub(replace_date2, sentence)

        # range first
        sentence = RE_TIME_RANGE.sub(replace_time, sentence)
        sentence = RE_TIME.sub(replace_time, sentence)

        # å¤„ç†~æ³¢æµª?·ä½œä¸ºè‡³?„æ›¿??
        sentence = RE_TO_RANGE.sub(replace_to_range, sentence)
        sentence = RE_TEMPERATURE.sub(replace_temperature, sentence)
        sentence = replace_measure(sentence)

        # å¤„ç†?°å?è¿ç®—
        while RE_ASMD.search(sentence):
            sentence = RE_ASMD.sub(replace_asmd, sentence)
        sentence = RE_POWER.sub(replace_power, sentence)

        sentence = RE_FRAC.sub(replace_frac, sentence)
        sentence = RE_PERCENTAGE.sub(replace_percentage, sentence)
        sentence = RE_MOBILE_PHONE.sub(replace_mobile, sentence)

        sentence = RE_TELEPHONE.sub(replace_phone, sentence)
        sentence = RE_NATIONAL_UNIFORM_NUMBER.sub(replace_phone, sentence)

        sentence = RE_RANGE.sub(replace_range, sentence)

        sentence = RE_INTEGER.sub(replace_negative_num, sentence)
        sentence = RE_VERSION_NUM.sub(replace_vrsion_num, sentence)
        sentence = RE_DECIMAL_NUM.sub(replace_number, sentence)
        sentence = RE_POSITIVE_QUANTIFIERS.sub(replace_positive_quantifier, sentence)
        sentence = RE_DEFAULT_NUM.sub(replace_default_num, sentence)
        sentence = RE_NUMBER.sub(replace_number, sentence)
        sentence = self._post_replace(sentence)

        return sentence

    def normalize(self, text: str) -> List[str]:
        sentences = self._split(text)
        sentences = [self.normalize_sentence(sent) for sent in sentences]
        return sentences
