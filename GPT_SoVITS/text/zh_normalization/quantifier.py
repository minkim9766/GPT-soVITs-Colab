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

from .num import num2str

# 温度表达式，温度会影?�负?�的读法
# -3°C ?�下三度
RE_TEMPERATURE = re.compile(r"(-?)(\d+(\.\d+)?)(°C|??�??�氏�?")
measure_dict = {
    "cm2": "平方?�米",
    "cm²": "平方?�米",
    "cm3": "立方?�米",
    "cm³": "立方?�米",
    "cm": "?�米",
    "db": "?�贝",
    "ds": "毫秒",
    "kg": "?�克",
    "km": "?�米",
    "m2": "平方�?,
    "m²": "平方�?,
    "m³": "立方�?,
    "m3": "立方�?,
    "ml": "毫升",
    "m": "�?,
    "mm": "毫米",
    "s": "�?,
}


def replace_temperature(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    sign = match.group(1)
    temperature = match.group(2)
    unit = match.group(3)
    sign: str = "?�下" if sign else ""
    temperature: str = num2str(temperature)
    unit: str = "?�氏�? if unit == "?�氏�? else "�?
    result = f"{sign}{temperature}{unit}"
    return result


def replace_measure(sentence) -> str:
    for q_notation in measure_dict:
        if q_notation in sentence:
            sentence = sentence.replace(q_notation, measure_dict[q_notation])
    return sentence
