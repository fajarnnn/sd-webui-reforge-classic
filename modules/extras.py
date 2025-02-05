from modules import images
from modules.ui_common import plaintext_to_html


def run_pnginfo(image):
    if image is None:
        return "", "", ""

    geninfo, items = images.read_info_from_image(image)
    items = {**{"parameters": geninfo}, **items}

    info = ""
    for key, text in items.items():
        info += (
            f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()
            + "\n"
        )

    if len(info) == 0:
        info = f"<div><p>Image contains no metadata...</p></div>"

    return "", geninfo, info
