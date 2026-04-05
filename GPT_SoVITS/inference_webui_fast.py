"""
?‰ä¸­?±æ··?ˆè¯†??
?‰æ—¥?±æ··?ˆè¯†??
å¤šè?ç§å¯?¨åˆ‡?†è¯†?«è?ç§?
?¨éƒ¨?‰ä¸­?‡è¯†??
?¨éƒ¨?‰è‹±?‡è¯†??
?¨éƒ¨?‰æ—¥?‡è¯†??
"""
import psutil
import os

def set_high_priority():
    """?Šå½“??Python è¿›ç¨‹è®¾ä¸º HIGH_PRIORITY_CLASS"""
    if os.name != "nt":
        return # ä»?Windows ?‰æ•ˆ
    p = psutil.Process(os.getpid())
    try:
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("å·²å°†è¿›ç¨‹ä¼˜å…ˆçº§è?ä¸?High")
    except psutil.AccessDenied:
        print("?ƒé™ä¸è¶³ï¼Œæ— æ³•ä¿®?¹ä¼˜?ˆçº§ï¼ˆè??¨ç??†å‘˜è¿è¡Œï¼?)
set_high_priority()
import json
import logging
import os
import random
import re
import sys

import torch

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)


infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
gpt_path = os.environ.get("gpt_path", None)
sovits_path = os.environ.get("sovits_path", None)
cnhubert_base_path = os.environ.get("cnhubert_base_path", None)
bert_path = os.environ.get("bert_path", None)
version = model_version = os.environ.get("version", "v2")

import gradio as gr
from TTS_infer_pack.text_segmentation_method import get_method
from TTS_infer_pack.TTS import NO_PROMPT_ERROR, TTS, TTS_Config

from tools.assets import css, js, top_html
from tools.i18n.i18n import I18nAuto, scan_language_list

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)


# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # ç¡?¿?´æ¥??Š¨?¨ç†UI?¶ä¹Ÿ?½å¤Ÿè®¾ç½®??

if torch.cuda.is_available():
    device = "cuda"
# elif torch.backends.mps.is_available():
#     device = "mps"
else:
    device = "cpu"

# is_half = False
# device = "cpu"

dict_language_v1 = {
    i18n("ä¸?–‡"): "all_zh",  # ?¨éƒ¨?‰ä¸­?‡è¯†??
    i18n("?±æ–‡"): "en",  # ?¨éƒ¨?‰è‹±?‡è¯†??######ä¸å˜
    i18n("?¥æ–‡"): "all_ja",  # ?¨éƒ¨?‰æ—¥?‡è¯†??
    i18n("ä¸?‹±æ··åˆ"): "zh",  # ?‰ä¸­?±æ··?ˆè¯†??###ä¸å˜
    i18n("?¥è‹±æ··åˆ"): "ja",  # ?‰æ—¥?±æ··?ˆè¯†??###ä¸å˜
    i18n("å¤šè?ç§æ··??): "auto",  # å¤šè?ç§å¯?¨åˆ‡?†è¯†?«è?ç§?
}
dict_language_v2 = {
    i18n("ä¸?–‡"): "all_zh",  # ?¨éƒ¨?‰ä¸­?‡è¯†??
    i18n("?±æ–‡"): "en",  # ?¨éƒ¨?‰è‹±?‡è¯†??######ä¸å˜
    i18n("?¥æ–‡"): "all_ja",  # ?¨éƒ¨?‰æ—¥?‡è¯†??
    i18n("ç²¤è?"): "all_yue",  # ?¨éƒ¨?‰ä¸­?‡è¯†??
    i18n("?©æ–‡"): "all_ko",  # ?¨éƒ¨?‰éŸ©?‡è¯†??
    i18n("ä¸?‹±æ··åˆ"): "zh",  # ?‰ä¸­?±æ··?ˆè¯†??###ä¸å˜
    i18n("?¥è‹±æ··åˆ"): "ja",  # ?‰æ—¥?±æ··?ˆè¯†??###ä¸å˜
    i18n("ç²¤è‹±æ··åˆ"): "yue",  # ?‰ç²¤?±æ··?ˆè¯†??###ä¸å˜
    i18n("?©è‹±æ··åˆ"): "ko",  # ?‰éŸ©?±æ··?ˆè¯†??###ä¸å˜
    i18n("å¤šè?ç§æ··??): "auto",  # å¤šè?ç§å¯?¨åˆ‡?†è¯†?«è?ç§?
    i18n("å¤šè?ç§æ··??ç²¤è?)"): "auto_yue",  # å¤šè?ç§å¯?¨åˆ‡?†è¯†?«è?ç§?
}
dict_language = dict_language_v1 if version == "v1" else dict_language_v2

cut_method = {
    i18n("ä¸åˆ‡"): "cut0",
    i18n("?‘å››?¥ä???): "cut1",
    i18n("??0å­—ä???): "cut2",
    i18n("?‰ä¸­?‡å¥?·ã€‚åˆ‡"): "cut3",
    i18n("?‰è‹±?‡å¥????): "cut4",
    i18n("?‰æ ‡?¹ç¬¦?·åˆ‡"): "cut5",
}

from config import change_choices, get_weights_names, name2gpt_path, name2sovits_path

SoVITS_names, GPT_names = get_weights_names()
from config import pretrained_sovits_name

path_sovits_v3 = pretrained_sovits_name["v3"]
path_sovits_v4 = pretrained_sovits_name["v4"]
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)

tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
# tts_config.version = version
tts_config.update_version(version)
if gpt_path is not None:
    if "ï¼? in gpt_path or "!" in gpt_path:
        gpt_path = name2gpt_path[gpt_path]
    tts_config.t2s_weights_path = gpt_path
if sovits_path is not None:
    if "ï¼? in sovits_path or "!" in sovits_path:
        sovits_path = name2sovits_path[sovits_path]
    tts_config.vits_weights_path = sovits_path
if cnhubert_base_path is not None:
    tts_config.cnhuhbert_base_path = cnhubert_base_path
if bert_path is not None:
    tts_config.bert_base_path = bert_path

print(tts_config)
tts_pipeline = TTS(tts_config)
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path
version = tts_config.version


def inference(
    text,
    text_lang,
    ref_audio_path,
    aux_ref_audio_paths,
    prompt_text,
    prompt_lang,
    top_k,
    top_p,
    temperature,
    text_split_method,
    batch_size,
    speed_factor,
    ref_text_free,
    split_bucket,
    fragment_interval,
    seed,
    keep_random,
    parallel_infer,
    repetition_penalty,
    sample_steps,
    super_sampling,
):
    seed = -1 if keep_random else seed
    actual_seed = seed if seed not in [-1, "", None] else random.randint(0, 2**32 - 1)
    inputs = {
        "text": text,
        "text_lang": dict_language[text_lang],
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": [item.name for item in aux_ref_audio_paths] if aux_ref_audio_paths is not None else [],
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language[prompt_lang],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size": int(batch_size),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "return_fragment": False,
        "fragment_interval": fragment_interval,
        "seed": actual_seed,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
    }
    try:
        for item in tts_pipeline.run(inputs):
            yield item, actual_seed
    except NO_PROMPT_ERROR:
        gr.Warning(i18n("V3ä¸æ”¯?æ— ?‚è€ƒæ–‡?¬æ¨¡å¼ï¼Œè¯·å¡«?™å‚?ƒæ–‡?¬ï¼"))


def custom_sort_key(s):
    # ä½¿ç”¨æ­£åˆ™è¡¨è¾¾å¼æ?–å­—ç¬¦ä¸²ä¸?š„?°å­—?¨åˆ†?Œé?°å­—?¨åˆ†
    parts = re.split("(\d+)", s)
    # å°†æ•°å­—éƒ¨?†è½¬?¢ä¸º?´æ•°ï¼Œé?°å­—?¨åˆ†ä¿æŒä¸å˜
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


if os.path.exists("./weight.json"):
    pass
else:
    with open("./weight.json", "w", encoding="utf-8") as file:
        json.dump({"GPT": {}, "SoVITS": {}}, file)

with open("./weight.json", "r", encoding="utf-8") as file:
    weight_data = file.read()
    weight_data = json.loads(weight_data)
    gpt_path = os.environ.get("gpt_path", weight_data.get("GPT", {}).get(version, GPT_names[-1]))
    sovits_path = os.environ.get("sovits_path", weight_data.get("SoVITS", {}).get(version, SoVITS_names[0]))
    if isinstance(gpt_path, list):
        gpt_path = gpt_path[0]
    if isinstance(sovits_path, list):
        sovits_path = sovits_path[0]

from process_ckpt import get_sovits_version_from_path_fast

v3v4set = {"v3", "v4"}


def change_sovits_weights(sovits_path, prompt_language=None, text_language=None):
    if "ï¼? in sovits_path or "!" in sovits_path:
        sovits_path = name2sovits_path[sovits_path]
    global version, model_version, dict_language, if_lora_v3
    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    # print(sovits_path,version, model_version, if_lora_v3)
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
    if if_lora_v3 == True and is_exist == False:
        info = path_sovits + "SoVITS %s" % model_version + i18n("åº•æ¨¡ç¼ºå¤±ï¼Œæ— æ³•åŠ è½½ç›¸åº?LoRA ?ƒé‡")
        gr.Warning(info)
        raise FileExistsError(info)
    dict_language = dict_language_v1 if version == "v1" else dict_language_v2
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = (
                {"__type__": "update"},
                {"__type__": "update", "value": prompt_language},
            )
        else:
            prompt_text_update = {"__type__": "update", "value": ""}
            prompt_language_update = {"__type__": "update", "value": i18n("ä¸?–‡")}
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = {"__type__": "update"}, {"__type__": "update", "value": text_language}
        else:
            text_update = {"__type__": "update", "value": ""}
            text_language_update = {"__type__": "update", "value": i18n("ä¸?–‡")}
        if model_version in v3v4set:
            visible_sample_steps = True
            visible_inp_refs = False
        else:
            visible_sample_steps = False
            visible_inp_refs = True
        yield (
            {"__type__": "update", "choices": list(dict_language.keys())},
            {"__type__": "update", "choices": list(dict_language.keys())},
            prompt_text_update,
            prompt_language_update,
            text_update,
            text_language_update,
            {"__type__": "update", "interactive": visible_sample_steps, "value": 32},
            {"__type__": "update", "visible": visible_inp_refs},
            {"__type__": "update", "interactive": True if model_version not in v3v4set else False},
            {"__type__": "update", "value": i18n("æ¨¡å‹? è½½ä¸?¼Œè¯·ç­‰å¾?), "interactive": False},
        )

    tts_pipeline.init_vits_weights(sovits_path)
    yield (
        {"__type__": "update", "choices": list(dict_language.keys())},
        {"__type__": "update", "choices": list(dict_language.keys())},
        prompt_text_update,
        prompt_language_update,
        text_update,
        text_language_update,
        {"__type__": "update", "interactive": visible_sample_steps, "value": 32},
        {"__type__": "update", "visible": visible_inp_refs},
        {"__type__": "update", "interactive": True if model_version not in v3v4set else False},
        {"__type__": "update", "value": i18n("?ˆæˆè¯?Ÿ³"), "interactive": True},
    )
    with open("./weight.json") as f:
        data = f.read()
        data = json.loads(data)
        data["SoVITS"][version] = sovits_path
    with open("./weight.json", "w") as f:
        f.write(json.dumps(data))


def change_gpt_weights(gpt_path):
    if "ï¼? in gpt_path or "!" in gpt_path:
        gpt_path = name2gpt_path[gpt_path]
    tts_pipeline.init_t2s_weights(gpt_path)


with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False, js=js, css=css) as app:
    gr.HTML(
        top_html.format(
            i18n("?¬è½¯ä»¶ä»¥MIT?è?å¼€æº? ä½œè€…ä¸å¯¹è½¯ä»¶å…·å¤‡ä»»ä½•æ§?¶åŠ›, ä½¿ç”¨è½?»¶?…ã€ä¼ ??½¯ä»¶å??ºçš„å£°éŸ³?…è‡ªè´Ÿå…¨è´?")
            + i18n("å¦‚ä¸è®¤å¯è¯¥æ¡æ¬? ?™ä¸?½ä½¿?¨æˆ–å¼•ç”¨è½?»¶?…å†…ä»»ä½•ä»£ç ?Œæ–‡ä»? è¯?§?¹ç›®å½•LICENSE.")
        ),
        elem_classes="markdown",
    )

    with gr.Column():
        # with gr.Group():
        gr.Markdown(value=i18n("æ¨¡å‹?‡æ¢"))
        with gr.Row():
            GPT_dropdown = gr.Dropdown(
                label=i18n("GPTæ¨¡å‹?—è¡¨"),
                choices=sorted(GPT_names, key=custom_sort_key),
                value=gpt_path,
                interactive=True,
            )
            SoVITS_dropdown = gr.Dropdown(
                label=i18n("SoVITSæ¨¡å‹?—è¡¨"),
                choices=sorted(SoVITS_names, key=custom_sort_key),
                value=sovits_path,
                interactive=True,
            )
            refresh_button = gr.Button(i18n("?·æ–°æ¨¡å‹è·?¾„"), variant="primary")
            refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])

    with gr.Row():
        with gr.Column():
            gr.Markdown(value=i18n("*è¯·ä¸Šä¼ å¹¶å¡«å†™?‚è€ƒä¿¡??))
            with gr.Row():
                inp_ref = gr.Audio(label=i18n("ä¸»å‚?ƒéŸ³é¢?è¯·ä¸Šä¼?~10ç§’å†…?‚è€ƒéŸ³é¢‘ï¼Œè¶…è¿‡ä¼šæŠ¥?™ï¼)"), type="filepath")
                inp_refs = gr.File(
                    label=i18n("è¾…å‚?ƒéŸ³é¢???€‰å¤šä¸ªï¼Œ?–ä¸??"),
                    file_count="multiple",
                    visible=True if model_version != "v3" else False,
                )
            prompt_text = gr.Textbox(label=i18n("ä¸»å‚?ƒéŸ³é¢‘çš„?‡æœ¬"), value="", lines=2)
            with gr.Row():
                prompt_language = gr.Dropdown(
                    label=i18n("ä¸»å‚?ƒéŸ³é¢‘çš„è¯?§"), choices=list(dict_language.keys()), value=i18n("ä¸?–‡")
                )
                with gr.Column():
                    ref_text_free = gr.Checkbox(
                        label=i18n("å¼€??— ?‚è€ƒæ–‡?¬æ¨¡å¼ã€‚ä¸å¡«å‚?ƒæ–‡?¬äº¦?¸å½“äºå???€?),
                        value=False,
                        interactive=True if model_version != "v3" else False,
                        show_label=True,
                    )
                    gr.Markdown(
                        i18n("ä½¿ç”¨? å‚?ƒæ–‡?¬æ¨¡å¼æ—¶å»ºè?ä½¿ç”¨å¾?°ƒ?„GPT")
                        + "<br>"
                        + i18n("?¬ä¸æ¸…å‚?ƒéŸ³é¢‘è??„å•¥(ä¸æ™“å¾—å†™????»¥å¼€?‚å???? è§†å¡«å†™?„å‚?ƒæ–‡?¬ã€?)
                    )

        with gr.Column():
            gr.Markdown(value=i18n("*è¯·å¡«?™é?è¦åˆ?çš„?? ‡?‡æœ¬?Œè?ç§æ¨¡å¼?))
            text = gr.Textbox(label=i18n("?€è¦åˆ?çš„?‡æœ¬"), value="", lines=20, max_lines=20)
            text_language = gr.Dropdown(
                label=i18n("?€è¦åˆ?çš„?‡æœ¬?„è?ç§?), choices=list(dict_language.keys()), value=i18n("ä¸?–‡")
            )

    with gr.Group():
        gr.Markdown(value=i18n("?¨ç†è®¾ç½®"))
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    batch_size = gr.Slider(
                        minimum=1, maximum=200, step=1, label=i18n("batch_size"), value=20, interactive=True
                    )
                    sample_steps = gr.Radio(
                        label=i18n("?‡æ ·æ­¥æ•°(ä»…å?V3/4?Ÿæ•ˆ)"), value=32, choices=[4, 8, 16, 32, 64, 128], visible=True
                    )
                with gr.Row():
                    fragment_interval = gr.Slider(
                        minimum=0.01, maximum=1, step=0.01, label=i18n("?†æ??´éš”(ç§?"), value=0.3, interactive=True
                    )
                    speed_factor = gr.Slider(
                        minimum=0.6, maximum=1.65, step=0.05, label="è¯?€?, value=1.0, interactive=True
                    )
                with gr.Row():
                    top_k = gr.Slider(minimum=1, maximum=100, step=1, label=i18n("top_k"), value=15, interactive=True)
                    top_p = gr.Slider(minimum=0, maximum=1, step=0.05, label=i18n("top_p"), value=1, interactive=True)
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0, maximum=1, step=0.05, label=i18n("temperature"), value=1, interactive=True
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0, maximum=2, step=0.05, label=i18n("?å¤?©ç½š"), value=1.35, interactive=True
                    )

            with gr.Column():
                with gr.Row():
                    how_to_cut = gr.Dropdown(
                        label=i18n("?ä¹ˆ??),
                        choices=[
                            i18n("ä¸åˆ‡"),
                            i18n("?‘å››?¥ä???),
                            i18n("??0å­—ä???),
                            i18n("?‰ä¸­?‡å¥?·ã€‚åˆ‡"),
                            i18n("?‰è‹±?‡å¥????),
                            i18n("?‰æ ‡?¹ç¬¦?·åˆ‡"),
                        ],
                        value=i18n("?‘å››?¥ä???),
                        interactive=True,
                        scale=1,
                    )
                    super_sampling = gr.Checkbox(
                        label=i18n("?³é¢‘è¶…é‡‡??ä»…å?V3?Ÿæ•ˆ))"), value=False, interactive=True, show_label=True
                    )

                with gr.Row():
                    parallel_infer = gr.Checkbox(label=i18n("å¹¶è¡Œ?¨ç†"), value=True, interactive=True, show_label=True)
                    split_bucket = gr.Checkbox(
                        label=i18n("?°æ®?†æ¡¶(å¹¶è¡Œ?¨ç†?¶ä¼š?ä½ä¸€?¹è?ç®—é‡)"),
                        value=True,
                        interactive=True,
                        show_label=True,
                    )

                with gr.Row():
                    seed = gr.Number(label=i18n("?æœºç§å­"), value=-1)
                    keep_random = gr.Checkbox(label=i18n("ä¿æŒ?æœº"), value=True, interactive=True, show_label=True)

                output = gr.Audio(label=i18n("è¾“å‡º?„è???))
                with gr.Row():
                    inference_button = gr.Button(i18n("?ˆæˆè¯?Ÿ³"), variant="primary")
                    stop_infer = gr.Button(i18n("ç»ˆæ??ˆæˆ"), variant="primary")

        inference_button.click(
            inference,
            [
                text,
                text_language,
                inp_ref,
                inp_refs,
                prompt_text,
                prompt_language,
                top_k,
                top_p,
                temperature,
                how_to_cut,
                batch_size,
                speed_factor,
                ref_text_free,
                split_bucket,
                fragment_interval,
                seed,
                keep_random,
                parallel_infer,
                repetition_penalty,
                sample_steps,
                super_sampling,
            ],
            [output, seed],
        )
        stop_infer.click(tts_pipeline.stop, [], [])
        SoVITS_dropdown.change(
            change_sovits_weights,
            [SoVITS_dropdown, prompt_language, text_language],
            [
                prompt_language,
                text_language,
                prompt_text,
                prompt_language,
                text,
                text_language,
                sample_steps,
                inp_refs,
                ref_text_free,
                inference_button,
            ],
        )  #
        GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])

    with gr.Group():
        gr.Markdown(
            value=i18n(
                "?‡æœ¬?‡åˆ†å·¥å…·?‚å¤ª?¿çš„?‡æœ¬?ˆæˆ?ºæ¥?ˆæœä¸ä?å®šå?ï¼Œæ?ä»¥å¤ª?¿å»ºè®?…ˆ?‡ã€‚åˆ?ä¼š?¹æ®?‡æœ¬?„æ¢è¡Œåˆ†å¼€?ˆæˆ?æ‹¼èµ·æ¥??
            )
        )
        with gr.Row():
            text_inp = gr.Textbox(label=i18n("?€è¦åˆ?çš„?‡åˆ†?æ–‡??), value="", lines=4)
            with gr.Column():
                _how_to_cut = gr.Radio(
                    label=i18n("?ä¹ˆ??),
                    choices=[
                        i18n("ä¸åˆ‡"),
                        i18n("?‘å››?¥ä???),
                        i18n("??0å­—ä???),
                        i18n("?‰ä¸­?‡å¥?·ã€‚åˆ‡"),
                        i18n("?‰è‹±?‡å¥????),
                        i18n("?‰æ ‡?¹ç¬¦?·åˆ‡"),
                    ],
                    value=i18n("?‘å››?¥ä???),
                    interactive=True,
                )
                cut_text = gr.Button(i18n("?‡åˆ†"), variant="primary")

            def to_cut(text_inp, how_to_cut):
                if len(text_inp.strip()) == 0 or text_inp == []:
                    return ""
                method = get_method(cut_method[how_to_cut])
                return method(text_inp)

            text_opt = gr.Textbox(label=i18n("?‡åˆ†?æ–‡??), value="", lines=4)
            cut_text.click(to_cut, [text_inp, _how_to_cut], [text_opt])
        gr.Markdown(value=i18n("?ç»­å°†æ”¯?è½¬?³ç´ ?æ‰‹å·¥ä¿®?¹éŸ³ç´ ã€è??³åˆ?åˆ†æ­¥æ‰§è¡Œã€?))

if __name__ == "__main__":
    app.queue().launch(  # concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=True,
        share=True,
        server_port=infer_ttswebui,
        # quiet=True,
    )
