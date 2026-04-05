import logging
import os
import traceback

import gradio as gr

from tools.i18n.i18n import I18nAuto
from tools.my_utils import clean_path

i18n = I18nAuto()

logger = logging.getLogger(__name__)
import sys

import ffmpeg
import torch
from bsroformer import Roformer_Loader
from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho

weight_uvr5_root = "tools/uvr5/uvr5_weights"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or name.endswith(".ckpt") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", "").replace(".ckpt", ""))

device = sys.argv[1]
is_half = eval(sys.argv[2])
webui_port_uvr5 = int(sys.argv[3])
is_share = eval(sys.argv[4])


def html_left(text, label="p"):
    return f"""<div style="text-align: left; margin: 0; padding: 0;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""


def html_center(text, label="p"):
    return f"""<div style="text-align: center; margin: 100; padding: 50;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""


def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    infos = []
    try:
        inp_root = clean_path(inp_root)
        save_root_vocal = clean_path(save_root_vocal)
        save_root_ins = clean_path(save_root_ins)
        is_hp3 = "HP3" in model_name
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        elif "roformer" in model_name.lower():
            func = Roformer_Loader
            pre_fun = func(
                model_path=os.path.join(weight_uvr5_root, model_name + ".ckpt"),
                config_path=os.path.join(weight_uvr5_root, model_name + ".yaml"),
                device=device,
                is_half=is_half,
            )
            if not os.path.exists(os.path.join(weight_uvr5_root, model_name + ".yaml")):
                infos.append(
                    "Warning: You are using a model without a configuration file. The program will automatically use the default configuration file. However, the default configuration file cannot guarantee that all models will run successfully. You can manually place the model configuration file into 'tools/uvr5/uvr5w_weights' and ensure that the configuration file is named as '<model_name>.yaml' then try it again. (For example, the configuration file corresponding to the model 'bs_roformer_ep_368_sdr_12.9628.ckpt' should be 'bs_roformer_ep_368_sdr_12.9628.yaml'.) Or you can just ignore this warning."
                )
                yield "\n".join(infos)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=device,
                is_half=is_half,
            )
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            if os.path.isfile(inp_path) == False:
                continue
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100":
                    need_reformat = 0
                    pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0, is_hp3)
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                os.system(f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y')
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0, is_hp3)
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append("%s->%s" % (os.path.basename(inp_path), traceback.format_exc()))
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    yield "\n".join(infos)


with gr.Blocks(title="UVR5 WebUI", analytics_enabled=False) as app:
    gr.Markdown(
        value=i18n("?�软件以MIT?��?开�? 作者不对软件具备任何控?�力, 使用�?��?�、传??��件�??�的声音?�自负全�?")
        + "<br>"
        + i18n("如不认可该条�? ?�不?�使?�或引用�?��?�内任何代码?�文�? �?��?�目录LICENSE.")
    )
    with gr.Group():
        gr.Markdown(html_center(i18n("伴奏人声?�离&?�混???�回�?), "h2"))
        with gr.Group():
            gr.Markdown(
                value=html_left(
                    i18n("人声伴奏?�离?�量处理�?使用UVR5模型??)
                    + "<br>"
                    + i18n(
                        "?�格?�文件夹�?��?�式举例�?E:\\codes\\py39\\vits_vc_gpu\\?�鹭?�华测试?�例(?�文件�??�器?��??�拷就行�???
                    )
                    + "<br>"
                    + i18n("模型?�为三类�?)
                    + "<br>"
                    + i18n(
                        "1?�保?�人声：不带?�声?�音频选这个，对主人声保留比HP5?��??�内置HP2?�HP3两个模型，HP3??��轻微漏伴奏但对主人声保留比HP2稍微好�?丁点�?
                    )
                    + "<br>"
                    + i18n("2?�仅保留主人声：�?��声的?�频?�这个，对主人声??��?�削弱。内置HP5一个模?�；")
                    + "<br>"
                    + i18n("3?�去混响?�去延迟模型（by FoxJoy）：")
                    + "<br>?��?
                    + i18n("(1)MDX-Net(onnx_dereverb):对于?�通道混响???好的?�择，不?�去?�单?�道混响�?)
                    + "<br>&emsp;"
                    + i18n(
                        "(234)DeEcho:?�除延迟?�果?�Aggressive比Normal?�除得更彻底，DeReverb额外?�除混响，可?�除?�声?�混?�，但是对高频重?�板式混?�去不干?�??
                    )
                    + "<br>"
                    + i18n("?�混???�延迟，?�：")
                    + "<br>"
                    + i18n("1?�DeEcho-DeReverb模型?�耗时??���?个DeEcho模型?�接�??�；")
                    + "<br>"
                    + i18n("2?�MDX-Net-Dereverb模型?�慢?�；")
                    + "<br>"
                    + i18n("3?�个人推?�的?�干�??�配�?��?�MDX-Net?�DeEcho-Aggressive??),
                    "h4",
                )
            )
            with gr.Row():
                with gr.Column():
                    model_choose = gr.Dropdown(label=i18n("模型"), choices=uvr5_names)
                    dir_wav_input = gr.Textbox(
                        label=i18n("输入待处?�音频文件夹�?��"),
                        placeholder="C:\\Users\\Desktop\\todo-songs",
                    )
                    wav_inputs = gr.File(
                        file_count="multiple", label=i18n("也可?�量输入?�频?�件, 二选�?, 优先读文件夹")
                    )
                with gr.Column():
                    agg = gr.Slider(
                        minimum=0,
                        maximum=20,
                        step=1,
                        label=i18n("人声?�取激进程�?),
                        value=10,
                        interactive=True,
                        visible=False,  # ?�不开?�调??
                    )
                    opt_vocal_root = gr.Textbox(label=i18n("?�定输出主人声文件夹"), value="output/uvr5_opt")
                    opt_ins_root = gr.Textbox(label=i18n("?�定输出?�主人声?�件�?), value="output/uvr5_opt")
                    format0 = gr.Radio(
                        label=i18n("导出?�件?�式"),
                        choices=["wav", "flac", "mp3", "m4a"],
                        value="flac",
                        interactive=True,
                    )
                    with gr.Column():
                        with gr.Row():
                            but2 = gr.Button(i18n("转换"), variant="primary")
                        with gr.Row():
                            vc_output4 = gr.Textbox(label=i18n("输出信息"), lines=3)
                but2.click(
                    uvr,
                    [
                        model_choose,
                        dir_wav_input,
                        opt_vocal_root,
                        wav_inputs,
                        opt_ins_root,
                        agg,
                        format0,
                    ],
                    [vc_output4],
                    api_name="uvr_convert",
                )
app.queue().launch(  # concurrency_count=511, max_size=1022
    server_name="0.0.0.0",
    inbrowser=True,
    share=True,
    server_port=webui_port_uvr5,
    # quiet=True,
)
