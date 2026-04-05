import os
import sys

os.environ["version"] = version = "v2Pro"
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
import warnings

warnings.filterwarnings("ignore")
import json
import platform
import shutil
import signal

import psutil
import torch
import yaml

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if os.path.exists(tmp):
    for name in os.listdir(tmp):
        if name == "jieba.cache":
            continue
        path = "%s/%s" % (tmp, name)
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass
import site
import traceback

site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if site_packages_roots == []:
    site_packages_roots = ["%s/runtime/Lib/site-packages" % now_dir]
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    # "%s\n%s/runtime\n%s/tools\n%s/tools/asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    "%s\n%s/GPT_SoVITS/BigVGAN\n%s/tools\n%s/tools/asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    % (now_dir, now_dir, now_dir, now_dir, now_dir, now_dir)
                )
            break
        except PermissionError:
            traceback.print_exc()
import shutil
import subprocess
from subprocess import Popen

from tools.assets import css, js, top_html
from tools.i18n.i18n import I18nAuto, scan_language_list

language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else "Auto"
os.environ["language"] = language
i18n = I18nAuto(language=language)
from multiprocessing import cpu_count

from config import (
    GPU_INDEX,
    GPU_INFOS,
    IS_GPU,
    exp_root,
    infer_device,
    is_half,
    is_share,
    memset,
    python_exec,
    webui_port_infer_tts,
    webui_port_main,
    webui_port_subfix,
    webui_port_uvr5,
)
from tools import my_utils
from tools.my_utils import check_details, check_for_existance

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # å½“é‡?°mpsä¸æ”¯?çš„æ­¥éª¤?¶ä½¿?¨cpu
import gradio as gr

n_cpu = cpu_count()

set_gpu_numbers = GPU_INDEX
gpu_infos = GPU_INFOS
mem = memset
is_gpu_ok = IS_GPU

v3v4set = {"v3", "v4"}


def set_default():
    global \
        default_batch_size, \
        default_max_batch_size, \
        gpu_info, \
        default_sovits_epoch, \
        default_sovits_save_every_epoch, \
        max_sovits_epoch, \
        max_sovits_save_every_epoch, \
        default_batch_size_s1, \
        if_force_ckpt
    if_force_ckpt = False
    gpu_info = "\n".join(gpu_infos)
    if is_gpu_ok:
        minmem = min(mem)
        default_batch_size = int(minmem // 2 if version not in v3v4set else minmem // 8)
        default_batch_size_s1 = int(minmem // 2)
    else:
        default_batch_size = default_batch_size_s1 = int(psutil.virtual_memory().total / 1024 / 1024 / 1024 / 4)
    if version not in v3v4set:
        default_sovits_epoch = 8
        default_sovits_save_every_epoch = 4
        max_sovits_epoch = 25  # 40
        max_sovits_save_every_epoch = 25  # 10
    else:
        default_sovits_epoch = 2
        default_sovits_save_every_epoch = 1
        max_sovits_epoch = 16  # 40 # 3 #è®?¤ªå¤?ä½œæ?
        max_sovits_save_every_epoch = 10  # 10 # 3

    default_batch_size = max(1, default_batch_size)
    default_batch_size_s1 = max(1, default_batch_size_s1)
    default_max_batch_size = default_batch_size * 3


set_default()

gpus = "-".join(map(str, GPU_INDEX))
default_gpu_numbers = infer_device.index


def fix_gpu_number(input):  # å°†è¶Š?Œçš„numberå¼ºåˆ¶?¹åˆ°?Œå†…
    try:
        if int(input) not in set_gpu_numbers:
            return default_gpu_numbers
    except:
        return input
    return input


def fix_gpu_numbers(inputs):
    output = []
    try:
        for input in inputs.split(","):
            output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs


from config import pretrained_gpt_name, pretrained_sovits_name


def check_pretrained_is_exist(version):
    pretrained_model_list = (
        pretrained_sovits_name[version],
        pretrained_sovits_name[version].replace("s2G", "s2D"),
        pretrained_gpt_name[version],
        "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    )
    _ = ""
    for i in pretrained_model_list:
        if "s2Dv3" not in i and "s2Dv4" not in i and os.path.exists(i) == False:
            _ += f"\n    {i}"
    if _:
        print("warning: ", i18n("ä»¥ä¸‹æ¨¡å‹ä¸å­˜??") + _)


check_pretrained_is_exist(version)
for key in pretrained_sovits_name.keys():
    if os.path.exists(pretrained_sovits_name[key]) == False:
        pretrained_sovits_name[key] = ""
for key in pretrained_gpt_name.keys():
    if os.path.exists(pretrained_gpt_name[key]) == False:
        pretrained_gpt_name[key] = ""

from config import (
    GPT_weight_root,
    GPT_weight_version2root,
    SoVITS_weight_root,
    SoVITS_weight_version2root,
    change_choices,
    get_weights_names,
)

for root in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(root, exist_ok=True)
SoVITS_names, GPT_names = get_weights_names()

p_label = None
p_uvr5 = None
p_asr = None
p_denoise = None
p_tts_inference = None


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


system = platform.system()


def kill_process(pid, process_name=""):
    if system == "Windows":
        cmd = "taskkill /t /f /pid %s" % pid
        # os.system(cmd)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        kill_proc_tree(pid)
    print(process_name + i18n("è¿›ç¨‹å·²ç»ˆæ­?))


def process_info(process_name="", indicator=""):
    if indicator == "opened":
        return process_name + i18n("å·²å???)
    elif indicator == "open":
        return i18n("å¼€??) + process_name
    elif indicator == "closed":
        return process_name + i18n("å·²å…³??)
    elif indicator == "close":
        return i18n("?³é—­") + process_name
    elif indicator == "running":
        return process_name + i18n("è¿è¡Œä¸?)
    elif indicator == "occupy":
        return process_name + i18n("? ç”¨ä¸?) + "," + i18n("?€?ˆç»ˆæ­¢æ‰?½å???¸‹ä¸€æ¬¡ä»»??)
    elif indicator == "finish":
        return process_name + i18n("å·²å®Œ??)
    elif indicator == "failed":
        return process_name + i18n("å¤±è´¥")
    elif indicator == "info":
        return process_name + i18n("è¿›ç¨‹è¾“å‡ºä¿¡æ¯")
    else:
        return process_name


process_name_subfix = i18n("?³é¢‘?‡æ³¨WebUI")


def change_label(path_list):
    global p_label
    if p_label is None:
        check_for_existance([path_list])
        path_list = my_utils.clean_path(path_list)
        cmd = '"%s" -s tools/subfix_webui.py --load_list "%s" --webui_port %s --is_share %s' % (
            python_exec,
            path_list,
            webui_port_subfix,
            is_share,
        )
        yield (
            process_info(process_name_subfix, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        print(cmd)
        p_label = Popen(cmd, shell=True)
    else:
        kill_process(p_label.pid, process_name_subfix)
        p_label = None
        yield (
            process_info(process_name_subfix, "closed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )


process_name_uvr5 = i18n("äººå£°?†ç¦»WebUI")


def change_uvr5():
    global p_uvr5
    if p_uvr5 is None:
        cmd = '"%s" -s tools/uvr5/webui.py "%s" %s %s %s' % (
            python_exec,
            infer_device,
            is_half,
            webui_port_uvr5,
            is_share,
        )
        yield (
            process_info(process_name_uvr5, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        print(cmd)
        p_uvr5 = Popen(cmd, shell=True)
    else:
        kill_process(p_uvr5.pid, process_name_uvr5)
        p_uvr5 = None
        yield (
            process_info(process_name_uvr5, "closed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )


process_name_tts = i18n("TTS?¨ç†WebUI")


def change_tts_inference(bert_path, cnhubert_base_path, gpu_number, gpt_path, sovits_path, batched_infer_enabled):
    global p_tts_inference
    if batched_infer_enabled:
        cmd = '"%s" -s GPT_SoVITS/inference_webui_fast.py "%s"' % (python_exec, language)
    else:
        cmd = '"%s" -s GPT_SoVITS/inference_webui.py "%s"' % (python_exec, language)
    # #####v3?‚ä¸??Œ? é€Ÿæ¨??
    # if version=="v3":
    #     cmd = '"%s" GPT_SoVITS/inference_webui.py "%s"'%(python_exec, language)
    if p_tts_inference is None:
        os.environ["gpt_path"] = gpt_path
        os.environ["sovits_path"] = sovits_path
        os.environ["cnhubert_base_path"] = cnhubert_base_path
        os.environ["bert_path"] = bert_path
        os.environ["_CUDA_VISIBLE_DEVICES"] = str(fix_gpu_number(gpu_number))
        os.environ["is_half"] = str(is_half)
        os.environ["infer_ttswebui"] = str(webui_port_infer_tts)
        os.environ["is_share"] = str(is_share)
        yield (
            process_info(process_name_tts, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        print(cmd)
        p_tts_inference = Popen(cmd, shell=True)
    else:
        kill_process(p_tts_inference.pid, process_name_tts)
        p_tts_inference = None
        yield (
            process_info(process_name_tts, "closed"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )


from tools.asr.config import asr_dict

process_name_asr = i18n("è¯?Ÿ³è¯†åˆ«")


def open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision):
    global p_asr
    if p_asr is None:
        asr_inp_dir = my_utils.clean_path(asr_inp_dir)
        asr_opt_dir = my_utils.clean_path(asr_opt_dir)
        check_for_existance([asr_inp_dir])
        cmd = f'"{python_exec}" -s tools/asr/{asr_dict[asr_model]["path"]}'
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f" -s {asr_model_size}"
        cmd += f" -l {asr_lang}"
        cmd += f" -p {asr_precision}"
        output_file_name = os.path.basename(asr_inp_dir)
        output_folder = asr_opt_dir or "output/asr_opt"
        output_file_path = os.path.abspath(f"{output_folder}/{output_file_name}.list")
        yield (
            process_info(process_name_asr, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        print(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr = None
        yield (
            process_info(process_name_asr, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update", "value": output_file_path},
            {"__type__": "update", "value": output_file_path},
            {"__type__": "update", "value": asr_inp_dir},
        )
    else:
        yield (
            process_info(process_name_asr, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )


def close_asr():
    global p_asr
    if p_asr is not None:
        kill_process(p_asr.pid, process_name_asr)
        p_asr = None
    return (
        process_info(process_name_asr, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


process_name_denoise = i18n("è¯?Ÿ³?å™ª")


def open_denoise(denoise_inp_dir, denoise_opt_dir):
    global p_denoise
    if p_denoise == None:
        denoise_inp_dir = my_utils.clean_path(denoise_inp_dir)
        denoise_opt_dir = my_utils.clean_path(denoise_opt_dir)
        check_for_existance([denoise_inp_dir])
        cmd = '"%s" -s tools/cmd-denoise.py -i "%s" -o "%s" -p %s' % (
            python_exec,
            denoise_inp_dir,
            denoise_opt_dir,
            "float16" if is_half == True else "float32",
        )

        yield (
            process_info(process_name_denoise, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        print(cmd)
        p_denoise = Popen(cmd, shell=True)
        p_denoise.wait()
        p_denoise = None
        yield (
            process_info(process_name_denoise, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update", "value": denoise_opt_dir},
            {"__type__": "update", "value": denoise_opt_dir},
        )
    else:
        yield (
            process_info(process_name_denoise, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )


def close_denoise():
    global p_denoise
    if p_denoise is not None:
        kill_process(p_denoise.pid, process_name_denoise)
        p_denoise = None
    return (
        process_info(process_name_denoise, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


p_train_SoVITS = None
process_name_sovits = i18n("SoVITSè®?»ƒ")


def open1Ba(
    version,
    batch_size,
    total_epoch,
    exp_name,
    text_low_lr_rate,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers1Ba,
    pretrained_s2G,
    pretrained_s2D,
    if_grad_ckpt,
    lora_rank,
):
    global p_train_SoVITS
    if p_train_SoVITS == None:
        exp_name = exp_name.rstrip(" ")
        config_file = (
            "GPT_SoVITS/configs/s2.json"
            if version not in {"v2Pro", "v2ProPlus"}
            else f"GPT_SoVITS/configs/s2{version}.json"
        )
        with open(config_file) as f:
            data = f.read()
            data = json.loads(data)
        s2_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s2_%s" % (s2_dir, version), exist_ok=True)
        if check_for_existance([s2_dir], is_train=True):
            check_details([s2_dir], is_train=True)
        if is_half == False:
            data["train"]["fp16_run"] = False
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["train"]["text_low_lr_rate"] = text_low_lr_rate
        data["train"]["pretrained_s2G"] = pretrained_s2G
        data["train"]["pretrained_s2D"] = pretrained_s2D
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["save_every_epoch"] = save_every_epoch
        data["train"]["gpu_numbers"] = gpu_numbers1Ba
        data["train"]["grad_ckpt"] = if_grad_ckpt
        data["train"]["lora_rank"] = lora_rank
        data["model"]["version"] = version
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = SoVITS_weight_version2root[version]
        data["name"] = exp_name
        data["version"] = version
        tmp_config_path = "%s/tmp_s2.json" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))
        if version in ["v1", "v2", "v2Pro", "v2ProPlus"]:
            cmd = '"%s" -s GPT_SoVITS/s2_train.py --config "%s"' % (python_exec, tmp_config_path)
        else:
            cmd = '"%s" -s GPT_SoVITS/s2_train_v3_lora.py --config "%s"' % (python_exec, tmp_config_path)
        yield (
            process_info(process_name_sovits, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS = None
        SoVITS_dropdown_update, GPT_dropdown_update = change_choices()
        yield (
            process_info(process_name_sovits, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            SoVITS_dropdown_update,
            GPT_dropdown_update,
        )
    else:
        yield (
            process_info(process_name_sovits, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )


def close1Ba():
    global p_train_SoVITS
    if p_train_SoVITS is not None:
        kill_process(p_train_SoVITS.pid, process_name_sovits)
        p_train_SoVITS = None
    return (
        process_info(process_name_sovits, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


p_train_GPT = None
process_name_gpt = i18n("GPTè®?»ƒ")


def open1Bb(
    batch_size,
    total_epoch,
    exp_name,
    if_dpo,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers,
    pretrained_s1,
):
    global p_train_GPT
    if p_train_GPT == None:
        exp_name = exp_name.rstrip(" ")
        with open(
            "GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml"
        ) as f:
            data = f.read()
            data = yaml.load(data, Loader=yaml.FullLoader)
        s1_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
        if check_for_existance([s1_dir], is_train=True):
            check_details([s1_dir], is_train=True)
        if is_half == False:
            data["train"]["precision"] = "32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["pretrained_s1"] = pretrained_s1
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_dpo"] = if_dpo
        data["train"]["half_weights_save_dir"] = GPT_weight_version2root[version]
        data["train"]["exp_name"] = exp_name
        data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
        data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
        data["output_dir"] = "%s/logs_s1_%s" % (s1_dir, version)
        # data["version"]=version

        os.environ["_CUDA_VISIBLE_DEVICES"] = str(fix_gpu_numbers(gpu_numbers.replace("-", ",")))
        os.environ["hz"] = "25hz"
        tmp_config_path = "%s/tmp_s1.yaml" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" -s GPT_SoVITS/s1_train.py --config_file "%s" ' % (python_exec, tmp_config_path)
        yield (
            process_info(process_name_gpt, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT = None
        SoVITS_dropdown_update, GPT_dropdown_update = change_choices()
        yield (
            process_info(process_name_gpt, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            SoVITS_dropdown_update,
            GPT_dropdown_update,
        )
    else:
        yield (
            process_info(process_name_gpt, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
        )


def close1Bb():
    global p_train_GPT
    if p_train_GPT is not None:
        kill_process(p_train_GPT.pid, process_name_gpt)
        p_train_GPT = None
    return (
        process_info(process_name_gpt, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps_slice = []
process_name_slice = i18n("è¯?Ÿ³?‡åˆ†")


def open_slice(inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, n_parts):
    global ps_slice
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    check_for_existance([inp])
    if os.path.exists(inp) == False:
        yield (
            i18n("è¾“å…¥è·?¾„ä¸å­˜??),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return
    if os.path.isfile(inp):
        n_parts = 1
    elif os.path.isdir(inp):
        pass
    else:
        yield (
            i18n("è¾“å…¥è·?¾„å­˜åœ¨ä½†ä¸??”¨"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        return
    if ps_slice == []:
        for i_part in range(n_parts):
            cmd = '"%s" -s tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s' % (
                python_exec,
                inp,
                opt_root,
                threshold,
                min_length,
                min_interval,
                hop_size,
                max_sil_kept,
                _max,
                alpha,
                i_part,
                n_parts,
            )
            print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        yield (
            process_info(process_name_slice, "opened"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )
        for p in ps_slice:
            p.wait()
        ps_slice = []
        yield (
            process_info(process_name_slice, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update", "value": opt_root},
            {"__type__": "update", "value": opt_root},
            {"__type__": "update", "value": opt_root},
        )
    else:
        yield (
            process_info(process_name_slice, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update"},
            {"__type__": "update"},
            {"__type__": "update"},
        )


def close_slice():
    global ps_slice
    if ps_slice != []:
        for p_slice in ps_slice:
            try:
                kill_process(p_slice.pid, process_name_slice)
            except:
                traceback.print_exc()
        ps_slice = []
    return (
        process_info(process_name_slice, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1a = []
process_name_1a = i18n("?‡æœ¬?†è¯ä¸ç‰¹å¾æ??)


def open1a(inp_text, inp_wav_dir, exp_name, gpu_numbers, bert_pretrained_dir):
    global ps1a
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1a == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "bert_pretrained_dir": bert_pretrained_dir,
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                    "is_half": str(is_half),
                }
            )
            os.environ.update(config)
            cmd = '"%s" -s GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1a.append(p)
        yield (
            process_info(process_name_1a, "running"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        for p in ps1a:
            p.wait()
        opt = []
        for i_part in range(all_parts):
            txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
            with open(txt_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(txt_path)
        path_text = "%s/2-name2text.txt" % opt_dir
        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1a = []
        if len("".join(opt)) > 0:
            yield (
                process_info(process_name_1a, "finish"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
        else:
            yield (
                process_info(process_name_1a, "failed"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
    else:
        yield (
            process_info(process_name_1a, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1a():
    global ps1a
    if ps1a != []:
        for p1a in ps1a:
            try:
                kill_process(p1a.pid, process_name_1a)
            except:
                traceback.print_exc()
        ps1a = []
    return (
        process_info(process_name_1a, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
ps1b = []
process_name_1b = i18n("è¯?Ÿ³?ªç›‘?£ç‰¹å¾æ??)


def open1b(version, inp_text, inp_wav_dir, exp_name, gpu_numbers, ssl_pretrained_dir):
    global ps1b
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1b == []:
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": "%s/%s" % (exp_root, exp_name),
            "cnhubert_base_dir": ssl_pretrained_dir,
            "sv_path": sv_path,
            "is_half": str(is_half),
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                }
            )
            os.environ.update(config)
            cmd = '"%s" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1b.append(p)
        yield (
            process_info(process_name_1b, "running"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        for p in ps1b:
            p.wait()
        ps1b = []
        if "Pro" in version:
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" -s GPT_SoVITS/prepare_datasets/2-get-sv.py' % python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1b.append(p)
            for p in ps1b:
                p.wait()
            ps1b = []
        yield (
            process_info(process_name_1b, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
    else:
        yield (
            process_info(process_name_1b, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1b():
    global ps1b
    if ps1b != []:
        for p1b in ps1b:
            try:
                kill_process(p1b.pid, process_name_1b)
            except:
                traceback.print_exc()
        ps1b = []
    return (
        process_info(process_name_1b, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1c = []
process_name_1c = i18n("è¯?¹‰Token?å–")


def open1c(version, inp_text, inp_wav_dir, exp_name, gpu_numbers, pretrained_s2G_path):
    global ps1c
    inp_text = my_utils.clean_path(inp_text)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1c == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        config_file = (
            "GPT_SoVITS/configs/s2.json"
            if version not in {"v2Pro", "v2ProPlus"}
            else f"GPT_SoVITS/configs/s2{version}.json"
        )
        config = {
            "inp_text": inp_text,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "pretrained_s2G": pretrained_s2G_path,
            "s2config_path": config_file,
            "is_half": str(is_half),
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                }
            )
            os.environ.update(config)
            cmd = '"%s" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1c.append(p)
        yield (
            process_info(process_name_1c, "running"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )
        for p in ps1c:
            p.wait()
        opt = ["item_name\tsemantic_audio"]
        path_semantic = "%s/6-name2semantic.tsv" % opt_dir
        for i_part in range(all_parts):
            semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
            with open(semantic_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(semantic_path)
        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1c = []
        yield (
            process_info(process_name_1c, "finish"),
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
        )
    else:
        yield (
            process_info(process_name_1c, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1c():
    global ps1c
    if ps1c != []:
        for p1c in ps1c:
            try:
                kill_process(p1c.pid, process_name_1c)
            except:
                traceback.print_exc()
        ps1c = []
    return (
        process_info(process_name_1c, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1abc = []
process_name_1abc = i18n("è®?»ƒ?†æ ¼å¼åŒ–ä¸€??¸‰è¿?)


def open1abc(
    version,
    inp_text,
    inp_wav_dir,
    exp_name,
    gpu_numbers1a,
    gpu_numbers1Ba,
    gpu_numbers1c,
    bert_pretrained_dir,
    ssl_pretrained_dir,
    pretrained_s2G_path,
):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1abc == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        try:
            #############################1a
            path_text = "%s/2-name2text.txt" % opt_dir
            if os.path.exists(path_text) == False or (
                os.path.exists(path_text) == True
                and len(open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")) < 2
            ):
                config = {
                    "inp_text": inp_text,
                    "inp_wav_dir": inp_wav_dir,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "bert_pretrained_dir": bert_pretrained_dir,
                    "is_half": str(is_half),
                }
                gpu_names = gpu_numbers1a.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" -s GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield (
                    i18n("è¿›åº¦") + ": 1A-Doing",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
                for p in ps1abc:
                    p.wait()

                opt = []
                for i_part in range(all_parts):  # txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                assert len("".join(opt)) > 0, process_info(process_name_1a, "failed")
            yield (
                i18n("è¿›åº¦") + ": 1A-Done",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            ps1abc = []
            #############################1b
            config = {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": opt_dir,
                "cnhubert_base_dir": ssl_pretrained_dir,
                "sv_path": sv_path,
            }
            gpu_names = gpu_numbers1Ba.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py' % python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            yield (
                i18n("è¿›åº¦") + ": 1A-Done, 1B-Doing",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            for p in ps1abc:
                p.wait()
            ps1abc = []
            if "Pro" in version:
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" -s GPT_SoVITS/prepare_datasets/2-get-sv.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                for p in ps1abc:
                    p.wait()
                ps1abc = []
            yield (
                i18n("è¿›åº¦") + ": 1A-Done, 1B-Done",
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": True},
            )
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if os.path.exists(path_semantic) == False or (
                os.path.exists(path_semantic) == True and os.path.getsize(path_semantic) < 31
            ):
                config_file = (
                    "GPT_SoVITS/configs/s2.json"
                    if version not in {"v2Pro", "v2ProPlus"}
                    else f"GPT_SoVITS/configs/s2{version}.json"
                )
                config = {
                    "inp_text": inp_text,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "pretrained_s2G": pretrained_s2G_path,
                    "s2config_path": config_file,
                }
                gpu_names = gpu_numbers1c.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": str(fix_gpu_number(gpu_names[i_part])),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield (
                    i18n("è¿›åº¦") + ": 1A-Done, 1B-Done, 1C-Doing",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
                for p in ps1abc:
                    p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                yield (
                    i18n("è¿›åº¦") + ": 1A-Done, 1B-Done, 1C-Done",
                    {"__type__": "update", "visible": False},
                    {"__type__": "update", "visible": True},
                )
            ps1abc = []
            yield (
                process_info(process_name_1abc, "finish"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
        except:
            traceback.print_exc()
            close1abc()
            yield (
                process_info(process_name_1abc, "failed"),
                {"__type__": "update", "visible": True},
                {"__type__": "update", "visible": False},
            )
    else:
        yield (
            process_info(process_name_1abc, "occupy"),
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
        )


def close1abc():
    global ps1abc
    if ps1abc != []:
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid, process_name_1abc)
            except:
                traceback.print_exc()
        ps1abc = []
    return (
        process_info(process_name_1abc, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


def switch_version(version_):
    os.environ["version"] = version_
    global version
    version = version_
    if pretrained_sovits_name[version] != "" and pretrained_gpt_name[version] != "":
        ...
    else:
        gr.Warning(i18n("?ªä¸‹è½½æ¨¡??) + ": " + version.upper())
    set_default()
    return (
        {"__type__": "update", "value": pretrained_sovits_name[version]},
        {"__type__": "update", "value": pretrained_sovits_name[version].replace("s2G", "s2D")},
        {"__type__": "update", "value": pretrained_gpt_name[version]},
        {"__type__": "update", "value": pretrained_gpt_name[version]},
        {"__type__": "update", "value": pretrained_sovits_name[version]},
        {"__type__": "update", "value": default_batch_size, "maximum": default_max_batch_size},
        {"__type__": "update", "value": default_sovits_epoch, "maximum": max_sovits_epoch},
        {"__type__": "update", "value": default_sovits_save_every_epoch, "maximum": max_sovits_save_every_epoch},
        {"__type__": "update", "visible": True if version not in v3v4set else False},
        {
            "__type__": "update",
            "value": False if not if_force_ckpt else True,
            "interactive": True if not if_force_ckpt else False,
        },
        {"__type__": "update", "interactive": True, "value": False},
        {"__type__": "update", "visible": True if version in v3v4set else False},
    )  # {'__type__': 'update', "interactive": False if version in v3v4set else True, "value": False}, \ ####batch infer


if os.path.exists("GPT_SoVITS/text/G2PWModel"):
    ...
else:
    cmd = '"%s" -s GPT_SoVITS/download.py' % python_exec
    p = Popen(cmd, shell=True)
    p.wait()


def sync(text):
    return {"__type__": "update", "value": text}


with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False, js=js, css=css) as app:
    gr.HTML(
        top_html.format(
            i18n("?¬è½¯ä»¶ä»¥MIT?è?å¼€æº? ä½œè€…ä¸å¯¹è½¯ä»¶å…·å¤‡ä»»ä½•æ§?¶åŠ›, ä½¿ç”¨è½?»¶?…ã€ä¼ ??½¯ä»¶å??ºçš„å£°éŸ³?…è‡ªè´Ÿå…¨è´?")
            + i18n("å¦‚ä¸è®¤å¯è¯¥æ¡æ¬? ?™ä¸?½ä½¿?¨æˆ–å¼•ç”¨è½?»¶?…å†…ä»»ä½•ä»£ç ?Œæ–‡ä»? è¯?§?¹ç›®å½•LICENSE.")
        ),
        elem_classes="markdown",
    )

    with gr.Tabs():
        with gr.TabItem("0-" + i18n("?ç½®?°æ®?†è·?–å·¥??)):  # ?å‰?æœº?‡ç‰‡?²æ?uvr5?†å†…å­?>uvr5->slicer->asr->?“æ ‡
            with gr.Accordion(label="0a-" + i18n("UVR5äººå£°ä¼´å¥?†ç¦»&?»æ··?å»å»¶è¿Ÿå·¥å…·")):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            uvr5_info = gr.Textbox(label=process_info(process_name_uvr5, "info"))
                    open_uvr5 = gr.Button(
                        value=process_info(process_name_uvr5, "open"), variant="primary", visible=True
                    )
                    close_uvr5 = gr.Button(
                        value=process_info(process_name_uvr5, "close"), variant="primary", visible=False
                    )

            with gr.Accordion(label="0b-" + i18n("è¯?Ÿ³?‡åˆ†å·¥å…·")):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            slice_inp_path = gr.Textbox(label=i18n("?³é¢‘?ªåŠ¨?‡åˆ†è¾“å…¥è·?¾„ï¼Œå¯?‡ä»¶??–‡ä»¶å¤¹"), value="")
                            slice_opt_root = gr.Textbox(
                                label=i18n("?‡åˆ†?çš„å­éŸ³é¢‘çš„è¾“å‡º?¹ç›®å½?), value="output/slicer_opt"
                            )
                        with gr.Row():
                            threshold = gr.Textbox(
                                label=i18n("threshold:?³é‡å°äºè¿™ä¸ª?¼è§†ä½œé™?³çš„å¤‡é€‰åˆ‡?²ç‚¹"), value="-34"
                            )
                            min_length = gr.Textbox(
                                label=i18n("min_length:æ¯æ??€å°å¤š?¿ï¼Œå¦‚æœç¬¬ä?æ®µå¤ª????´å’Œ?é¢æ®µè¿èµ·æ¥?´åˆ°è¶…è¿‡è¿™ä¸ª??),
                                value="4000",
                            )
                            min_interval = gr.Textbox(label=i18n("min_interval:?€??ˆ‡?²é—´??), value="300")
                            hop_size = gr.Textbox(
                                label=i18n("hop_size:?ä¹ˆç®—éŸ³?æ›²çº¿ï¼Œè¶Šå°ç²¾åº¦è¶Šå¤§è®¡ç®—?è¶Šé«˜ï¼ˆä¸æ˜¯ç²¾åº¦è¶Šå¤§?ˆæœè¶Šå?ï¼?),
                                value="10",
                            )
                            max_sil_kept = gr.Textbox(label=i18n("max_sil_kept:?‡å®Œ?é™?³æ?å¤šç•™å¤šé•¿"), value="500")
                        with gr.Row():
                            _max = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                label=i18n("max:å½’ä??–å?€å¤§å€¼å¤šå°?),
                                value=0.9,
                                interactive=True,
                            )
                            alpha = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                label=i18n("alpha_mix:æ··å¤šå°‘æ¯”ä¾‹å½’ä¸€?–å?³é¢‘è¿›æ¥"),
                                value=0.25,
                                interactive=True,
                            )
                        with gr.Row():
                            n_process = gr.Slider(
                                minimum=1,
                                maximum=n_cpu,
                                step=1,
                                label=i18n("?‡å‰²ä½¿ç”¨?„è¿›ç¨‹æ•°"),
                                value=4,
                                interactive=True,
                            )
                            slicer_info = gr.Textbox(label=process_info(process_name_slice, "info"))
                    open_slicer_button = gr.Button(
                        value=process_info(process_name_slice, "open"), variant="primary", visible=True
                    )
                    close_slicer_button = gr.Button(
                        value=process_info(process_name_slice, "close"), variant="primary", visible=False
                    )

            # gr.Markdown(value="0bb-" + i18n("è¯?Ÿ³?å™ªå·¥å…·")+i18n("(ä¸ç¨³å®šï¼Œ?ˆåˆ«?¨ï¼Œ??ƒ½?£åŒ–æ¨¡å‹?ˆæœï¼?"))
            with gr.Row(visible=False):
                with gr.Column(scale=3):
                    with gr.Row():
                        denoise_input_dir = gr.Textbox(label=i18n("è¾“å…¥?‡ä»¶å¤¹è·¯å¾?), value="")
                        denoise_output_dir = gr.Textbox(label=i18n("è¾“å‡º?‡ä»¶å¤¹è·¯å¾?), value="output/denoise_opt")
                    with gr.Row():
                        denoise_info = gr.Textbox(label=process_info(process_name_denoise, "info"))
                open_denoise_button = gr.Button(
                    value=process_info(process_name_denoise, "open"), variant="primary", visible=True
                )
                close_denoise_button = gr.Button(
                    value=process_info(process_name_denoise, "close"), variant="primary", visible=False
                )

            with gr.Accordion(label="0c-" + i18n("è¯?Ÿ³è¯†åˆ«å·¥å…·")):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            asr_inp_dir = gr.Textbox(
                                label=i18n("è¾“å…¥?‡ä»¶å¤¹è·¯å¾?), value="D:\\GPT-SoVITS\\raw\\xxx", interactive=True
                            )
                            asr_opt_dir = gr.Textbox(
                                label=i18n("è¾“å‡º?‡ä»¶å¤¹è·¯å¾?), value="output/asr_opt", interactive=True
                            )
                        with gr.Row():
                            asr_model = gr.Dropdown(
                                label=i18n("ASR æ¨¡å‹"),
                                choices=list(asr_dict.keys()),
                                interactive=True,
                                value="è¾¾æ‘© ASR (ä¸?–‡)",
                            )
                            asr_size = gr.Dropdown(
                                label=i18n("ASR æ¨¡å‹å°ºå?"), choices=["large"], interactive=True, value="large"
                            )
                            asr_lang = gr.Dropdown(
                                label=i18n("ASR è¯??è®¾ç½®"), choices=["zh", "yue"], interactive=True, value="zh"
                            )
                            asr_precision = gr.Dropdown(
                                label=i18n("?°æ®ç±»å‹ç²¾åº¦"), choices=["float32"], interactive=True, value="float32"
                            )
                        with gr.Row():
                            asr_info = gr.Textbox(label=process_info(process_name_asr, "info"))
                    open_asr_button = gr.Button(
                        value=process_info(process_name_asr, "open"), variant="primary", visible=True
                    )
                    close_asr_button = gr.Button(
                        value=process_info(process_name_asr, "close"), variant="primary", visible=False
                    )

                def change_lang_choices(key):  # ?¹æ®?‰æ‹©?„æ¨¡?‹ä¿®?¹å¯?‰çš„è¯??
                    return {"__type__": "update", "choices": asr_dict[key]["lang"], "value": asr_dict[key]["lang"][0]}

                def change_size_choices(key):  # ?¹æ®?‰æ‹©?„æ¨¡?‹ä¿®?¹å¯?‰çš„æ¨¡å‹å°ºå?
                    return {"__type__": "update", "choices": asr_dict[key]["size"], "value": asr_dict[key]["size"][-1]}

                def change_precision_choices(key):  # ?¹æ®?‰æ‹©?„æ¨¡?‹ä¿®?¹å¯?‰çš„è¯??
                    if key == "Faster Whisper (å¤šè?ç§?":
                        if default_batch_size <= 4:
                            precision = "int8"
                        elif is_half:
                            precision = "float16"
                        else:
                            precision = "float32"
                    else:
                        precision = "float32"
                    return {"__type__": "update", "choices": asr_dict[key]["precision"], "value": precision}

                asr_model.change(change_lang_choices, [asr_model], [asr_lang])
                asr_model.change(change_size_choices, [asr_model], [asr_size])
                asr_model.change(change_precision_choices, [asr_model], [asr_precision])

            with gr.Accordion(label="0d-" + i18n("è¯?Ÿ³?‡æœ¬?¡å??‡æ³¨å·¥å…·")):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            path_list = gr.Textbox(
                                label=i18n("?‡æ³¨?‡ä»¶è·?¾„ (?«æ–‡ä»¶åç¼€ *.list)"),
                                value="D:\\RVC1006\\GPT-SoVITS\\raw\\xxx.list",
                                interactive=True,
                            )
                            label_info = gr.Textbox(label=process_info(process_name_subfix, "info"))
                    open_label = gr.Button(
                        value=process_info(process_name_subfix, "open"), variant="primary", visible=True
                    )
                    close_label = gr.Button(
                        value=process_info(process_name_subfix, "close"), variant="primary", visible=False
                    )

                open_label.click(change_label, [path_list], [label_info, open_label, close_label])
                close_label.click(change_label, [path_list], [label_info, open_label, close_label])
                open_uvr5.click(change_uvr5, [], [uvr5_info, open_uvr5, close_uvr5])
                close_uvr5.click(change_uvr5, [], [uvr5_info, open_uvr5, close_uvr5])

        with gr.TabItem(i18n("1-GPT-SoVITS-TTS")):
            with gr.Accordion(i18n("å¾?°ƒæ¨¡å‹ä¿¡æ¯")):
                with gr.Row():
                    with gr.Row(equal_height=True):
                        exp_name = gr.Textbox(
                            label=i18n("*å®éªŒ/æ¨¡å‹??),
                            value="xxx",
                            interactive=True,
                            scale=3,
                        )
                        gpu_info_box = gr.Textbox(
                            label=i18n("?¾å¡ä¿¡æ¯"),
                            value=gpu_info,
                            visible=True,
                            interactive=False,
                            scale=5,
                        )
                        version_checkbox = gr.Radio(
                            label=i18n("è®?»ƒæ¨¡å‹?„ç‰ˆ??),
                            value=version,
                            choices=["v1", "v2", "v4", "v2Pro", "v2ProPlus"],
                            scale=5,
                        )
            with gr.Accordion(label=i18n("é¢„è?ç»ƒæ¨¡?‹è·¯å¾?), open=False):
                with gr.Row():
                    with gr.Row(equal_height=True):
                        pretrained_s1 = gr.Textbox(
                            label=i18n("é¢„è?ç»ƒGPTæ¨¡å‹è·?¾„"),
                            value=pretrained_gpt_name[version],
                            interactive=True,
                            lines=1,
                            max_lines=1,
                            scale=3,
                        )
                        pretrained_s2G = gr.Textbox(
                            label=i18n("é¢„è?ç»ƒSoVITS-Gæ¨¡å‹è·?¾„"),
                            value=pretrained_sovits_name[version],
                            interactive=True,
                            lines=1,
                            max_lines=1,
                            scale=5,
                        )
                        pretrained_s2D = gr.Textbox(
                            label=i18n("é¢„è?ç»ƒSoVITS-Dæ¨¡å‹è·?¾„"),
                            value=pretrained_sovits_name[version].replace("s2G", "s2D"),
                            interactive=True,
                            lines=1,
                            max_lines=1,
                            scale=5,
                        )

            with gr.TabItem("1A-" + i18n("è®?»ƒ?†æ ¼å¼åŒ–å·¥å…·")):
                with gr.Accordion(label=i18n("è¾“å‡ºlogs/å®éªŒ?ç›®å½•ä¸‹åº”æœ‰23456å¼€å¤´çš„?‡ä»¶?Œæ–‡ä»¶å¤¹")):
                    with gr.Row():
                        with gr.Row():
                            inp_text = gr.Textbox(
                                label=i18n("*?‡æœ¬?‡æ³¨?‡ä»¶"),
                                value=r"D:\RVC1006\GPT-SoVITS\raw\xxx.list",
                                interactive=True,
                                scale=10,
                            )
                        with gr.Row():
                            inp_wav_dir = gr.Textbox(
                                label=i18n("*è®?»ƒ?†éŸ³é¢‘æ–‡ä»¶ç›®å½?),
                                # value=r"D:\RVC1006\GPT-SoVITS\raw\xxx",
                                interactive=True,
                                placeholder=i18n(
                                    "å¡«åˆ‡?²å?³é¢‘?€?¨ç›®å½•ï¼è¯»å–?„éŸ³é¢‘æ–‡ä»¶å®Œ?´è·¯å¾?è¯¥ç›®å½??¼æ¥-list?‡ä»¶?Œæ³¢å½¢å?åº”çš„?‡ä»¶?ï¼ˆä¸æ˜¯?¨è·¯å¾„ï¼‰?‚å¦‚?œç•™ç©ºåˆ™ä½¿ç”¨.list?‡ä»¶?Œçš„ç»å??¨è·¯å¾„ã€?
                                ),
                                scale=10,
                            )

                with gr.Accordion(label="1Aa-" + process_name_1a):
                    with gr.Row():
                        with gr.Row():
                            gpu_numbers1a = gr.Textbox(
                                label=i18n("GPU?¡å·ä»??†å‰²ï¼Œæ¯ä¸ªå¡?·ä?ä¸ªè¿›ç¨?),
                                value="%s-%s" % (gpus, gpus),
                                interactive=True,
                            )
                        with gr.Row():
                            bert_pretrained_dir = gr.Textbox(
                                label=i18n("é¢„è?ç»ƒä¸­?‡BERTæ¨¡å‹è·?¾„"),
                                value="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                                interactive=False,
                                lines=2,
                            )
                        with gr.Row():
                            button1a_open = gr.Button(
                                value=process_info(process_name_1a, "open"), variant="primary", visible=True
                            )
                            button1a_close = gr.Button(
                                value=process_info(process_name_1a, "close"), variant="primary", visible=False
                            )
                        with gr.Row():
                            info1a = gr.Textbox(label=process_info(process_name_1a, "info"))

                with gr.Accordion(label="1Ab-" + process_name_1b):
                    with gr.Row():
                        with gr.Row():
                            gpu_numbers1Ba = gr.Textbox(
                                label=i18n("GPU?¡å·ä»??†å‰²ï¼Œæ¯ä¸ªå¡?·ä?ä¸ªè¿›ç¨?),
                                value="%s-%s" % (gpus, gpus),
                                interactive=True,
                            )
                        with gr.Row():
                            cnhubert_base_dir = gr.Textbox(
                                label=i18n("é¢„è?ç»ƒSSLæ¨¡å‹è·?¾„"),
                                value="GPT_SoVITS/pretrained_models/chinese-hubert-base",
                                interactive=False,
                                lines=2,
                            )
                        with gr.Row():
                            button1b_open = gr.Button(
                                value=process_info(process_name_1b, "open"), variant="primary", visible=True
                            )
                            button1b_close = gr.Button(
                                value=process_info(process_name_1b, "close"), variant="primary", visible=False
                            )
                        with gr.Row():
                            info1b = gr.Textbox(label=process_info(process_name_1b, "info"))

                with gr.Accordion(label="1Ac-" + process_name_1c):
                    with gr.Row():
                        with gr.Row():
                            gpu_numbers1c = gr.Textbox(
                                label=i18n("GPU?¡å·ä»??†å‰²ï¼Œæ¯ä¸ªå¡?·ä?ä¸ªè¿›ç¨?),
                                value="%s-%s" % (gpus, gpus),
                                interactive=True,
                            )
                        with gr.Row():
                            pretrained_s2G_ = gr.Textbox(
                                label=i18n("é¢„è?ç»ƒSoVITS-Gæ¨¡å‹è·?¾„"),
                                value=pretrained_sovits_name[version],
                                interactive=False,
                                lines=2,
                            )
                        with gr.Row():
                            button1c_open = gr.Button(
                                value=process_info(process_name_1c, "open"), variant="primary", visible=True
                            )
                            button1c_close = gr.Button(
                                value=process_info(process_name_1c, "close"), variant="primary", visible=False
                            )
                        with gr.Row():
                            info1c = gr.Textbox(label=process_info(process_name_1c, "info"))

                with gr.Accordion(label="1Aabc-" + process_name_1abc):
                    with gr.Row():
                        with gr.Row():
                            button1abc_open = gr.Button(
                                value=process_info(process_name_1abc, "open"), variant="primary", visible=True
                            )
                            button1abc_close = gr.Button(
                                value=process_info(process_name_1abc, "close"), variant="primary", visible=False
                            )
                        with gr.Row():
                            info1abc = gr.Textbox(label=process_info(process_name_1abc, "info"))

            pretrained_s2G.change(sync, [pretrained_s2G], [pretrained_s2G_])
            open_asr_button.click(
                open_asr,
                [asr_inp_dir, asr_opt_dir, asr_model, asr_size, asr_lang, asr_precision],
                [asr_info, open_asr_button, close_asr_button, path_list, inp_text, inp_wav_dir],
            )
            close_asr_button.click(close_asr, [], [asr_info, open_asr_button, close_asr_button])
            open_slicer_button.click(
                open_slice,
                [
                    slice_inp_path,
                    slice_opt_root,
                    threshold,
                    min_length,
                    min_interval,
                    hop_size,
                    max_sil_kept,
                    _max,
                    alpha,
                    n_process,
                ],
                [slicer_info, open_slicer_button, close_slicer_button, asr_inp_dir, denoise_input_dir, inp_wav_dir],
            )
            close_slicer_button.click(close_slice, [], [slicer_info, open_slicer_button, close_slicer_button])
            open_denoise_button.click(
                open_denoise,
                [denoise_input_dir, denoise_output_dir],
                [denoise_info, open_denoise_button, close_denoise_button, asr_inp_dir, inp_wav_dir],
            )
            close_denoise_button.click(close_denoise, [], [denoise_info, open_denoise_button, close_denoise_button])

            button1a_open.click(
                open1a,
                [inp_text, inp_wav_dir, exp_name, gpu_numbers1a, bert_pretrained_dir],
                [info1a, button1a_open, button1a_close],
            )
            button1a_close.click(close1a, [], [info1a, button1a_open, button1a_close])
            button1b_open.click(
                open1b,
                [version_checkbox, inp_text, inp_wav_dir, exp_name, gpu_numbers1Ba, cnhubert_base_dir],
                [info1b, button1b_open, button1b_close],
            )
            button1b_close.click(close1b, [], [info1b, button1b_open, button1b_close])
            button1c_open.click(
                open1c,
                [version_checkbox, inp_text, inp_wav_dir, exp_name, gpu_numbers1c, pretrained_s2G],
                [info1c, button1c_open, button1c_close],
            )
            button1c_close.click(close1c, [], [info1c, button1c_open, button1c_close])
            button1abc_open.click(
                open1abc,
                [
                    version_checkbox,
                    inp_text,
                    inp_wav_dir,
                    exp_name,
                    gpu_numbers1a,
                    gpu_numbers1Ba,
                    gpu_numbers1c,
                    bert_pretrained_dir,
                    cnhubert_base_dir,
                    pretrained_s2G,
                ],
                [info1abc, button1abc_open, button1abc_close],
            )
            button1abc_close.click(close1abc, [], [info1abc, button1abc_open, button1abc_close])

            with gr.TabItem("1B-" + i18n("å¾?°ƒè®?»ƒ")):
                with gr.Accordion(label="1Ba-" + i18n("SoVITS è®?»ƒ: æ¨¡å‹?ƒé‡?‡ä»¶??SoVITS_weights/")):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                batch_size = gr.Slider(
                                    minimum=1,
                                    maximum=default_max_batch_size,
                                    step=1,
                                    label=i18n("æ¯å¼ ?¾å¡?„batch_size"),
                                    value=default_batch_size,
                                    interactive=True,
                                )
                                total_epoch = gr.Slider(
                                    minimum=1,
                                    maximum=max_sovits_epoch,
                                    step=1,
                                    label=i18n("?»è?ç»ƒè½®?°total_epochï¼Œä¸å»ºè?å¤ªé«˜"),
                                    value=default_sovits_epoch,
                                    interactive=True,
                                )
                            with gr.Row():
                                text_low_lr_rate = gr.Slider(
                                    minimum=0.2,
                                    maximum=0.6,
                                    step=0.05,
                                    label=i18n("?‡æœ¬æ¨¡å—å­¦ä¹ ?‡æƒ??),
                                    value=0.4,
                                    visible=True if version not in v3v4set else False,
                                )  # v3v4 not need
                                lora_rank = gr.Radio(
                                    label=i18n("LoRAç§?),
                                    value="32",
                                    choices=["16", "32", "64", "128"],
                                    visible=True if version in v3v4set else False,
                                )  # v1v2 not need
                                save_every_epoch = gr.Slider(
                                    minimum=1,
                                    maximum=max_sovits_save_every_epoch,
                                    step=1,
                                    label=i18n("ä¿å­˜é¢‘ç‡save_every_epoch"),
                                    value=default_sovits_save_every_epoch,
                                    interactive=True,
                                )
                        with gr.Column():
                            with gr.Column():
                                if_save_latest = gr.Checkbox(
                                    label=i18n("??¦ä»…ä¿å­˜æ??°çš„?ƒé‡?‡ä»¶ä»¥èŠ‚?ç¡¬?˜ç©º??),
                                    value=True,
                                    interactive=True,
                                    show_label=True,
                                )
                                if_save_every_weights = gr.Checkbox(
                                    label=i18n("??¦?¨æ¯æ¬¡ä¿å­˜æ—¶?´ç‚¹å°†æ?ç»ˆå°æ¨¡å‹ä¿å­˜?³weights?‡ä»¶å¤?),
                                    value=True,
                                    interactive=True,
                                    show_label=True,
                                )
                                if_grad_ckpt = gr.Checkbox(
                                    label="v3??¦å¼€??¢¯åº???¥ç‚¹?‚çœ?¾å­˜? ç”¨",
                                    value=False,
                                    interactive=True if version in v3v4set else False,
                                    show_label=True,
                                    visible=False,
                                )  # ?ªæœ‰V3s2??»¥??
                            with gr.Row():
                                gpu_numbers1Ba = gr.Textbox(
                                    label=i18n("GPU?¡å·ä»??†å‰²ï¼Œæ¯ä¸ªå¡?·ä?ä¸ªè¿›ç¨?),
                                    value="%s" % (gpus),
                                    interactive=True,
                                )
                    with gr.Row():
                        with gr.Row():
                            button1Ba_open = gr.Button(
                                value=process_info(process_name_sovits, "open"), variant="primary", visible=True
                            )
                            button1Ba_close = gr.Button(
                                value=process_info(process_name_sovits, "close"), variant="primary", visible=False
                            )
                        with gr.Row():
                            info1Ba = gr.Textbox(label=process_info(process_name_sovits, "info"))
                with gr.Accordion(label="1Bb-" + i18n("GPT è®?»ƒ: æ¨¡å‹?ƒé‡?‡ä»¶??GPT_weights/")):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                batch_size1Bb = gr.Slider(
                                    minimum=1,
                                    maximum=40,
                                    step=1,
                                    label=i18n("æ¯å¼ ?¾å¡?„batch_size"),
                                    value=default_batch_size_s1,
                                    interactive=True,
                                )
                                total_epoch1Bb = gr.Slider(
                                    minimum=2,
                                    maximum=50,
                                    step=1,
                                    label=i18n("?»è?ç»ƒè½®?°total_epoch"),
                                    value=15,
                                    interactive=True,
                                )
                            with gr.Row():
                                save_every_epoch1Bb = gr.Slider(
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    label=i18n("ä¿å­˜é¢‘ç‡save_every_epoch"),
                                    value=5,
                                    interactive=True,
                                )
                                if_dpo = gr.Checkbox(
                                    label=i18n("??¦å¼€?¯DPOè®?»ƒ?‰é¡¹(å®éªŒ??"),
                                    value=False,
                                    interactive=True,
                                    show_label=True,
                                )
                        with gr.Column():
                            with gr.Column():
                                if_save_latest1Bb = gr.Checkbox(
                                    label=i18n("??¦ä»…ä¿å­˜æ??°çš„?ƒé‡?‡ä»¶ä»¥èŠ‚?ç¡¬?˜ç©º??),
                                    value=True,
                                    interactive=True,
                                    show_label=True,
                                )
                                if_save_every_weights1Bb = gr.Checkbox(
                                    label=i18n("??¦?¨æ¯æ¬¡ä¿å­˜æ—¶?´ç‚¹å°†æ?ç»ˆå°æ¨¡å‹ä¿å­˜?³weights?‡ä»¶å¤?),
                                    value=True,
                                    interactive=True,
                                    show_label=True,
                                )
                            with gr.Row():
                                gpu_numbers1Bb = gr.Textbox(
                                    label=i18n("GPU?¡å·ä»??†å‰²ï¼Œæ¯ä¸ªå¡?·ä?ä¸ªè¿›ç¨?),
                                    value="%s" % (gpus),
                                    interactive=True,
                                )
                    with gr.Row():
                        with gr.Row():
                            button1Bb_open = gr.Button(
                                value=process_info(process_name_gpt, "open"), variant="primary", visible=True
                            )
                            button1Bb_close = gr.Button(
                                value=process_info(process_name_gpt, "close"), variant="primary", visible=False
                            )
                        with gr.Row():
                            info1Bb = gr.Textbox(label=process_info(process_name_gpt, "info"))

            button1Ba_close.click(close1Ba, [], [info1Ba, button1Ba_open, button1Ba_close])
            button1Bb_close.click(close1Bb, [], [info1Bb, button1Bb_open, button1Bb_close])

            with gr.TabItem("1C-" + i18n("?¨ç†")):
                gr.Markdown(
                    value=i18n(
                        "?‰æ‹©è®?»ƒå®Œå­˜?¾åœ¨SoVITS_weights?ŒGPT_weightsä¸‹çš„æ¨¡å‹?‚é»˜è®¤çš„? ä¸ª??º•æ¨¡ï¼Œä½“éªŒ5ç§’Zero Shot TTSä¸è?ç»ƒæ¨?†ç”¨??
                    )
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            GPT_dropdown = gr.Dropdown(
                                label=i18n("GPTæ¨¡å‹?—è¡¨"),
                                choices=GPT_names,
                                value=GPT_names[-1],
                                interactive=True,
                            )
                            SoVITS_dropdown = gr.Dropdown(
                                label=i18n("SoVITSæ¨¡å‹?—è¡¨"),
                                choices=SoVITS_names,
                                value=SoVITS_names[0],
                                interactive=True,
                            )
                    with gr.Column(scale=2):
                        with gr.Row():
                            gpu_number_1C = gr.Textbox(
                                label=i18n("GPU?¡å·,?ªèƒ½å¡?ä¸ªæ•´??), value=gpus, interactive=True
                            )
                            refresh_button = gr.Button(i18n("?·æ–°æ¨¡å‹è·?¾„"), variant="primary")
                    refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
                with gr.Row(equal_height=True):
                    with gr.Row():
                        batched_infer_enabled = gr.Checkbox(
                            label=i18n("??”¨å¹¶è¡Œ?¨ç†?ˆæœ¬"), value=False, interactive=True, show_label=True
                        )
                        open_tts = gr.Button(
                            value=process_info(process_name_tts, "open"), variant="primary", visible=True
                        )
                        close_tts = gr.Button(
                            value=process_info(process_name_tts, "close"), variant="primary", visible=False
                        )
                    with gr.Column():
                        tts_info = gr.Textbox(label=process_info(process_name_tts, "info"), scale=2)
                    open_tts.click(
                        change_tts_inference,
                        [
                            bert_pretrained_dir,
                            cnhubert_base_dir,
                            gpu_number_1C,
                            GPT_dropdown,
                            SoVITS_dropdown,
                            batched_infer_enabled,
                        ],
                        [tts_info, open_tts, close_tts],
                    )
                    close_tts.click(
                        change_tts_inference,
                        [
                            bert_pretrained_dir,
                            cnhubert_base_dir,
                            gpu_number_1C,
                            GPT_dropdown,
                            SoVITS_dropdown,
                            batched_infer_enabled,
                        ],
                        [tts_info, open_tts, close_tts],
                    )
            button1Ba_open.click(
                open1Ba,
                [
                    version_checkbox,
                    batch_size,
                    total_epoch,
                    exp_name,
                    text_low_lr_rate,
                    if_save_latest,
                    if_save_every_weights,
                    save_every_epoch,
                    gpu_numbers1Ba,
                    pretrained_s2G,
                    pretrained_s2D,
                    if_grad_ckpt,
                    lora_rank,
                ],
                [info1Ba, button1Ba_open, button1Ba_close, SoVITS_dropdown, GPT_dropdown],
            )
            button1Bb_open.click(
                open1Bb,
                [
                    batch_size1Bb,
                    total_epoch1Bb,
                    exp_name,
                    if_dpo,
                    if_save_latest1Bb,
                    if_save_every_weights1Bb,
                    save_every_epoch1Bb,
                    gpu_numbers1Bb,
                    pretrained_s1,
                ],
                [info1Bb, button1Bb_open, button1Bb_close, SoVITS_dropdown, GPT_dropdown],
            )
            version_checkbox.change(
                switch_version,
                [version_checkbox],
                [
                    pretrained_s2G,
                    pretrained_s2D,
                    pretrained_s1,
                    GPT_dropdown,
                    SoVITS_dropdown,
                    batch_size,
                    total_epoch,
                    save_every_epoch,
                    text_low_lr_rate,
                    if_grad_ckpt,
                    batched_infer_enabled,
                    lora_rank,
                ],
            )

        with gr.TabItem(i18n("2-GPT-SoVITS-?˜å£°")):
            gr.Markdown(value=i18n("?½å·¥ä¸?¼Œè¯·é™?™ä½³??))

    app.queue().launch(  # concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=True,
        share=True,
        server_port=webui_port_main,
        # quiet=True,
    )
