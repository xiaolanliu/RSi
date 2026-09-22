"""Inspect the exact multimodal request offline; this never calls a provider."""
import argparse
import html
import json
from pathlib import Path


def report(request, output):
    payload = json.loads(Path(request).read_text())
    cards, texts, label = [], [], "Image"
    for message in payload["input"]:
        for part in message.get("content", []):
            if part["type"] == "input_text":
                label = part["text"]
                if label.startswith("Current robot context: "):
                    label = json.dumps(json.loads(label.removeprefix("Current robot context: ")), indent=2)
                texts.append(f"<pre>{html.escape(label)}</pre>")
            elif part["type"] == "input_image":
                url = part["image_url"]
                if not url.startswith(("data:image/jpeg;base64,", "data:image/png;base64,")):
                    raise ValueError("Offline inspection only accepts embedded PNG/JPEG images")
                cards.append(f'<figure><figcaption>{html.escape(label)}</figcaption><img src="{html.escape(url, quote=True)}"></figure>')
    metadata = {key: value for key, value in payload.items() if key not in ("input", "instructions")}
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>恢复请求上下文检查</title><style>
body{{max-width:1300px;margin:30px auto;padding:0 16px;font:16px/1.6 sans-serif;background:#f5f7fa;color:#182638}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}figure{{margin:0;background:white;padding:12px}}img{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}}.note{{padding:16px;background:#fff1c4}}
</style><h1>恢复请求上下文检查</h1>
<p class="note">本页只展示实际请求材料，共 {len(cards)} 张图像；不是 GPT 回复或纠正成功证明。生成此页面不会调用 API。</p>
<div class="gallery">{''.join(cards)}</div>
<details><summary>完整文字上下文</summary>{''.join(texts)}</details>
<details><summary>系统指令</summary><pre>{html.escape(payload.get('instructions',''))}</pre></details>
<details><summary>请求元信息</summary><pre>{html.escape(json.dumps(metadata,indent=2))}</pre></details></html>'''
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report(args.request, args.output)
