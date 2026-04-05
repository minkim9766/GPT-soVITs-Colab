js = """
function deleteTheme() {

const params = new URLSearchParams(window.location.search);
if (params.has('__theme')) {
    params.delete('__theme');
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.location.replace(newUrl);
}

}
"""

css = """
/* CSSStyleRule */
.markdown {
    padding: 6px 10px;
}

@media (prefers-color-scheme: light) {
    .markdown {
        background-color: lightblue;
        color: #000;
    }
}

@media (prefers-color-scheme: dark) {
    .markdown {
        background-color: #4b4b4b;
        color: rgb(244, 244, 245);
    }
}

::selection {
    background: #ffc078 !important;
}

footer {
    height: 50px !important;           /* 设置页脚高度 */
    background-color: transparent !important; /* ?�景?�明 */
    display: flex;
    justify-content: center;           /* 居中对齐 */
    align-items: center;               /* ?�直居中 */
}

footer * {
    display: none !important;          /* ?�藏?�?�子?�素 */
}

"""

top_html = """
<div align="center">
    <div style="margin-bottom: 5px; font-size: 15px;">{}</div>
    <div style="display: flex; gap: 60px; justify-content: center;">
        <a href="https://github.com/RVC-Boss/GPT-SoVITS" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-GPT--SoVITS-blue.svg?style=for-the-badge&logo=github" style="width: auto; height: 30px;">
        </a>
        <a href="https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e" target="_blank">
            <img src="https://img.shields.io/badge/简体中???��??�档-blue?style=for-the-badge&logo=googledocs&logoColor=white" style="width: auto; height: 30px;">
        </a>
        <a href="https://lj1995-gpt-sovits-proplus.hf.space/" target="_blank">
            <img src="https://img.shields.io/badge/?�费?�线体验-free_online_demo-yellow.svg?style=for-the-badge&logo=huggingface" style="width: auto; height: 30px;">
        </a>
        <a href="https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e" target="_blank">
            <img src="https://img.shields.io/badge/English-READ%20DOCS-blue?style=for-the-badge&logo=googledocs&logoColor=white" style="width: auto; height: 30px;">
        </a>
        <a href="https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE" target="_blank">
            <img src="https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge&logo=opensourceinitiative" style="width: auto; height: 30px;">
        </a>
    </div>
</div>
"""
