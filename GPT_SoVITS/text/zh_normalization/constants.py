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
import string

from pypinyin.constants import SUPPORT_UCS4

# ?®Ëßí?äËßíËΩ¨Êç¢
# ?±ÊñáÂ≠óÁ¨¶?®Ëßí -> ?äËßí?†Â∞ÑË°?(num: 52)
F2H_ASCII_LETTERS = {ord(char) + 65248: ord(char) for char in string.ascii_letters}

# ?±ÊñáÂ≠óÁ¨¶?äËßí -> ?®Ëßí?†Â∞ÑË°?
H2F_ASCII_LETTERS = {value: key for key, value in F2H_ASCII_LETTERS.items()}

# ?∞Â≠óÂ≠óÁ¨¶?®Ëßí -> ?äËßí?†Â∞ÑË°?(num: 10)
F2H_DIGITS = {ord(char) + 65248: ord(char) for char in string.digits}
# ?∞Â≠óÂ≠óÁ¨¶?äËßí -> ?®Ëßí?†Â∞ÑË°?
H2F_DIGITS = {value: key for key, value in F2H_DIGITS.items()}

# ?áÁÇπÁ¨?è∑?®Ëßí -> ?äËßí?†Â∞ÑË°?(num: 32)
F2H_PUNCTUATIONS = {ord(char) + 65248: ord(char) for char in string.punctuation}
# ?áÁÇπÁ¨?è∑?äËßí -> ?®Ëßí?†Â∞ÑË°?
H2F_PUNCTUATIONS = {value: key for key, value in F2H_PUNCTUATIONS.items()}

# Á©∫Ê†º (num: 1)
F2H_SPACE = {"\u3000": " "}
H2F_SPACE = {" ": "\u3000"}

# ???âÊãº?≥ÁöÑÊ±âÂ≠ó"?ÑÂ≠óÁ¨¶‰∏≤ÔºåÂèØ?®‰∫éNSW?êÂèñ
if SUPPORT_UCS4:
    RE_NSW = re.compile(
        r"(?:[^"
        r"\u3007"  # ??
        r"\u3400-\u4dbf"  # CJK?©Â±ïA:[3400-4DBF]
        r"\u4e00-\u9fff"  # CJK?∫Êú¨:[4E00-9FFF]
        r"\uf900-\ufaff"  # CJK?ºÂ?:[F900-FAFF]
        r"\U00020000-\U0002A6DF"  # CJK?©Â±ïB:[20000-2A6DF]
        r"\U0002A703-\U0002B73F"  # CJK?©Â±ïC:[2A700-2B73F]
        r"\U0002B740-\U0002B81D"  # CJK?©Â±ïD:[2B740-2B81D]
        r"\U0002F80A-\U0002FA1F"  # CJK?ºÂ??©Â±ï:[2F800-2FA1F]
        r"])+"
    )
else:
    RE_NSW = re.compile(  # pragma: no cover
        r"(?:[^"
        r"\u3007"  # ??
        r"\u3400-\u4dbf"  # CJK?©Â±ïA:[3400-4DBF]
        r"\u4e00-\u9fff"  # CJK?∫Êú¨:[4E00-9FFF]
        r"\uf900-\ufaff"  # CJK?ºÂ?:[F900-FAFF]
        r"])+"
    )
