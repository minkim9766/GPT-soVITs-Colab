# by https://github.com/Cosmo-klara

from __future__ import print_function

import re
import inflect
import unicodedata

# ?��?计量?�位?�换�?
measurement_map = {
    "m": ["meter", "meters"],
    "km": ["kilometer", "kilometers"],
    "km/h": ["kilometer per hour", "kilometers per hour"],
    "ft": ["feet", "feet"],
    "L": ["liter", "liters"],
    "tbsp": ["tablespoon", "tablespoons"],
    "tsp": ["teaspoon", "teaspoons"],
    "h": ["hour", "hours"],
    "min": ["minute", "minutes"],
    "s": ["second", "seconds"],
    "°C": ["degree celsius", "degrees celsius"],
    "°F": ["degree fahrenheit", "degrees fahrenheit"],
}


# 识别 12,000 类型
_inflect = inflect.engine()

# 转化?�字序数�?
_ordinal_number_re = re.compile(r"\b([0-9]+)\. ")

# ?�听说�??��?于数字�??�识?�其实用 \d 会�?一??

_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")

# ?�间识别
_time_re = re.compile(r"\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b")

# ?��?计量?�位识别
_measurement_re = re.compile(r"\b([0-9]+(\.[0-9]+)?(m|km|km/h|ft|L|tbsp|tsp|h|min|s|°C|°F))\b")

# ?�后 £ 识别 ( ?�了识别两边?��?边的，但??��?�道为�?么失败了??��﹏┭??)
_pounds_re_start = re.compile(r"£([0-9\.\,]*[0-9]+)")
_pounds_re_end = re.compile(r"([0-9\.\,]*[0-9]+)£")

# ?�后 $ 识别
_dollars_re_start = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_dollars_re_end = re.compile(r"([(0-9\.\,]*[0-9]+)\$")

# 小数?�识??
_decimal_number_re = re.compile(r"([0-9]+\.\s*[0-9]+)")

# ?�数识别 (形式 "3/4" )
_fraction_re = re.compile(r"([0-9]+/[0-9]+)")

# 序数词识??
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")

# ?�字处理
_number_re = re.compile(r"[0-9]+")


def _convert_ordinal(m):
    """
    ?�准?�序?�词, 例如: 1. 2. 3. 4. 5. 6.
    Examples:
        input: "1. "
        output: "1st"
    ?�后?�后?�的 _expand_ordinal, 将其转化�?first 这类??
    """
    ordinal = _inflect.ordinal(m.group(1))
    return ordinal + ", "


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_time(m):
    """
    �?24 小时?�的?�间转换�?12 小时?�的?�间表示?�式??

    Examples:
        input: "13:00 / 4:00 / 13:30"
        output: "one o'clock p.m. / four o'clock am. / one thirty p.m."
    """
    hours, minutes = map(int, m.group(1, 2))
    period = "a.m." if hours < 12 else "p.m."
    if hours > 12:
        hours -= 12

    hour_word = _inflect.number_to_words(hours)
    minute_word = _inflect.number_to_words(minutes) if minutes != 0 else ""

    if minutes == 0:
        return f"{hour_word} o'clock {period}"
    else:
        return f"{hour_word} {minute_word} {period}"


def _expand_measurement(m):
    """
    处理一些常见的测量?�位?��?, ??��??��: m, km, km/h, ft, L, tbsp, tsp, h, min, s, °C, °F
    如果要拓展的话修?? _measurement_re ??measurement_map
    """
    sign = m.group(3)
    ptr = 1
    # ?�不?�怎么?�便?�取?�字，又?�得?��??�，诶，1.2 ?��?也是复数读法，干?�直?�去??"."
    num = int(m.group(1).replace(sign, "").replace(".", ""))
    decimal_part = m.group(2)
    # 上面?�断?�漏洞，比如 0.1 ?�情?�，?�这?�排?�了
    if decimal_part == None and num == 1:
        ptr = 0
    return m.group(1).replace(sign, " " + measurement_map[sign][ptr])


def _expand_pounds(m):
    """
    没找?�特?�规?�的说明，和美元?�处?��??�，?�实??��?�两个合并在一�?
    """
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " pounds"  # Unexpected format
    pounds = int(parts[0]) if parts[0] else 0
    pence = int(parts[1].ljust(2, "0")) if len(parts) > 1 and parts[1] else 0
    if pounds and pence:
        pound_unit = "pound" if pounds == 1 else "pounds"
        penny_unit = "penny" if pence == 1 else "pence"
        return "%s %s and %s %s" % (pounds, pound_unit, pence, penny_unit)
    elif pounds:
        pound_unit = "pound" if pounds == 1 else "pounds"
        return "%s %s" % (pounds, pound_unit)
    elif pence:
        penny_unit = "penny" if pence == 1 else "pence"
        return "%s %s" % (pence, penny_unit)
    else:
        return "zero pounds"


def _expand_dollars(m):
    """
    change: 美分??100 ?�限?? 应�?要做补零?�吧
    Example:
        input: "32.3$ / $6.24"
        output: "thirty-two dollars and thirty cents" / "six dollars and twenty-four cents"
    """
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1].ljust(2, "0")) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s and %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


# 小数?�处??
def _expand_decimal_number(m):
    """
    Example:
        input: "13.234"
        output: "thirteen point two three four"
    """
    match = m.group(1)
    parts = match.split(".")
    words = []
    # ?�历字符串中?�每个字�?
    for char in parts[1]:
        if char == ".":
            words.append("point")
        else:
            words.append(char)
    return parts[0] + " point " + " ".join(words)


# ?�数?�处??
def _expend_fraction(m):
    """
    规则1: ?�子使用?�数词�?�? ?�母?�序?�词读法.
    规则2: 如果?�子大于 1, ?��??�母?�时?�使?�序?�词复数读法.
    规则3: 当分母为2?�时?? ?�母读做 half, 并且当分子大�?1 ?�时?? half 也要?�复?��?�? 读为 halves.
    Examples:

    | Written |	Said |
    |:---:|:---:|
    | 1/3 | one third |
    | 3/4 | three fourths |
    | 5/6 | five sixths |
    | 1/2 | one half |
    | 3/2 | three halves |
    """
    match = m.group(0)
    numerator, denominator = map(int, match.split("/"))

    numerator_part = _inflect.number_to_words(numerator)
    if denominator == 2:
        if numerator == 1:
            denominator_part = "half"
        else:
            denominator_part = "halves"
    elif denominator == 1:
        return f"{numerator_part}"
    else:
        denominator_part = _inflect.ordinal(_inflect.number_to_words(denominator))
        if numerator > 1:
            denominator_part += "s"

    return f"{numerator_part} {denominator_part}"


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


# ?�减乘除
RE_ASMD = re.compile(
    r"((-?)((\d+)(\.\d+)?[?�¹²³⁴?�⁶?�⁸?�ˣʸⁿ]*)|(\.\d+[?�¹²³⁴?�⁶?�⁸?�ˣʸⁿ]*)|([A-Za-z][?�¹²³⁴?�⁶?�⁸?�ˣʸⁿ]*))\s+([\+\-\×÷=])\s+((-?)((\d+)(\.\d+)?[?�¹²³⁴?�⁶?�⁸?�ˣʸⁿ]*)|(\.\d+[?�¹²³⁴?�⁶?�⁸?�ˣʸⁿ]*)|([A-Za-z][?�¹²³⁴?�⁶?�⁸?�ˣʸⁿ]*))"
)
# RE_ASMD = re.compile(
#     r"\b((-?)((\d+)(\.\d+)?[?�¹²³⁴?�⁶?�⁸?�ˣʸⁿ]*)|(\.\d+[?�¹²³⁴?�⁶?�⁸?�ˣʸⁿ]*)|([A-Za-z][?�¹²³⁴?�⁶?�⁸?�ˣʸⁿ]*))([\+\-\×÷=])((-?)((\d+)(\.\d+)?[?�¹²³⁴?�⁶?�⁸?�ˣʸⁿ]*)|(\.\d+[?�¹²³⁴?�⁶?�⁸?�ˣʸⁿ]*)|([A-Za-z][?�¹²³⁴?�⁶?�⁸?�ˣʸⁿ]*))\b"
# )

asmd_map = {"+": " plus ", "-": " minus ", "×": " times ", "÷": " divided by ", "=": " Equals "}


def replace_asmd(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    result = match.group(1) + asmd_map[match.group(8)] + match.group(9)
    return result


RE_INTEGER = re.compile(r"(?:^|\s+)(-)" r"(\d+)")


def replace_negative_num(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    number = match.group(2)
    sign: str = "negative " if sign else ""
    result = f"{sign}{number}"
    return result



def normalize(text):
    """
    !!! ?�?�的处理?��?要�?�?��输入 !!!
    ??��添加?�的处理，只?�要添?��??�表达式?��?应的处理?�数?�可
    """

    text = re.sub(_ordinal_number_re, _convert_ordinal, text)

    # 处理?��?运算
    # ?�换text = re.sub(r"(?<!\d)-|-(?!\d)", " minus ", text)
    while RE_ASMD.search(text):
        text = RE_ASMD.sub(replace_asmd, text)
    text = RE_INTEGER.sub(replace_negative_num, text)

    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_time_re, _expand_time, text)
    text = re.sub(_measurement_re, _expand_measurement, text)
    text = re.sub(_pounds_re_start, _expand_pounds, text)
    text = re.sub(_pounds_re_end, _expand_pounds, text)
    text = re.sub(_dollars_re_start, _expand_dollars, text)
    text = re.sub(_dollars_re_end, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_number, text)
    text = re.sub(_fraction_re, _expend_fraction, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)

    text = "".join(
        char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn"
    )  # Strip accents

    text = re.sub("%", " percent", text)
    text = re.sub("[^ A-Za-z'.,?!\-]", "", text)
    text = re.sub(r"(?i)i\.e\.", "that is", text)
    text = re.sub(r"(?i)e\.g\.", "for example", text)
    # 增加�?��?�单词拆??
    text = re.sub(r"(?<!^)(?<![\s])([A-Z])", r" \1", text)
    return text


if __name__ == "__main__":
    # ?�觉得其实可以把?�分结果展示?�来（只读，?�者修?�不影响传给TTS?�实?�text�?
    # ?�后让用?�确认后?�输?�给 TTS，可以�??�户检?�自己有没有不标?�的输入
    print(normalize("1. test ordinal number 1st"))
    print(normalize("32.3$, $6.24, 1.1£, £7.14."))
    print(normalize("3/23, 1/2, 3/2, 1/3, 6/1"))
    print(normalize("1st, 22nd"))
    print(normalize("a test 20h, 1.2s, 1L, 0.1km"))
    print(normalize("a test of time 4:00, 13:00, 13:30"))
    print(normalize("a test of temperature 4°F, 23°C, -19°C"))
