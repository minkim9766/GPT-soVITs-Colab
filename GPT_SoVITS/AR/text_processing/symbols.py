# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/text_processing/symbols.py
# reference: https://github.com/lifeiteng/vall-e
PAD = "_"
PUNCTUATION = ';:,.!?¡¿?��?«»?��?'
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
IPA_LETTERS = (
    "???æ??β??ç??ðʤ?????????ɡ?ɢ?ɦɧħɥ?ɨɪ?ɭɬɫɮ?ɱɯɰŋɳɲɴøɵɸθœɶ?ɹɺɾɻ??ɽ???ʧ???ⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀ???????ʼʴʰʱʲʷ?ˤ??�↑?�↗??̩'�?
)
SYMBOLS = [PAD] + list(PUNCTUATION) + list(LETTERS) + list(IPA_LETTERS)
SPACE_ID = SYMBOLS.index(" ")
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}
