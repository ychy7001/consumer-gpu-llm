#!/usr/bin/env python3
"""合成中文 OCR 测试图（带标准答案）：A4 扫描页、截图（含表格）、手机拍照风格。
用法: gen_ocr_set.py <out_dir> <md 文件...>
输出: out_dir/NN-kind.(png|jpg) 与 out_dir/manifest.jsonl {"file","kind","w","h","text"}
"""
import json, random, re, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SANS = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
SERIF = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
CJK = re.compile(r"[一-鿿]")


def corpus(files):
    out = []
    for f in files:
        t = Path(f).read_text(encoding="utf-8")
        t = re.sub(r"```.*?```", "", t, flags=re.S)
        for ln in t.splitlines():
            ln = re.sub(r"`[^`]*`|\[([^\]]*)\]\([^)]*\)|https?://\S+|[*#>|_]+", lambda m: m.group(1) or "", ln).strip(" -")
            ln = re.sub(r"\s+", " ", ln).strip()
            if len(ln) >= 20 and len(CJK.findall(ln)) / len(ln) > 0.5:
                out.append(ln)
    return out


def wrap(draw, text, font, width):
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) > width:
            lines.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def page(paras, w, h, font_path, size, margin, table=None):
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(font_path, size)
    y, used = margin, []
    lh = int(size * 1.6)
    for p in paras:
        ls = wrap(d, p, f, w - 2 * margin)
        if y + lh * len(ls) > h - margin - (lh * 6 if table else 0):
            break
        for ln in ls:
            d.text((margin, y), ln, font=f, fill="black"); y += lh
        used.append("".join(ls)); y += lh // 2
    if table:
        cols = len(table[0]); cw = (w - 2 * margin) // cols
        for r, row in enumerate(table):
            for c, cell in enumerate(row):
                x0, y0 = margin + c * cw, y + r * lh
                d.rectangle([x0, y0, x0 + cw, y0 + lh], outline="black", width=2)
                d.text((x0 + size // 3, y0 + size // 4), cell, font=f, fill="black")
        used.append("".join("".join(r) for r in table))
    return img, "\n".join(used)


def photoize(img, rng):
    img = img.rotate(rng.uniform(-2.5, 2.5), expand=True, fillcolor=(235, 232, 225))
    px = img.load()
    for _ in range(img.width * img.height // 400):
        x, y = rng.randrange(img.width), rng.randrange(img.height)
        g = rng.randint(150, 230); px[x, y] = (g, g, g)
    return img.filter(ImageFilter.GaussianBlur(0.8))


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261031)
    paras = corpus(sys.argv[2:]); rng.shuffle(paras)
    pos = [0]
    def take(n):  # 循环取段落；每页起点错开，页与页内容不同
        s = pos[0]; pos[0] += n
        return [paras[(s + k) % len(paras)] for k in range(n)]
    table = [["指标", "基准", "新方案", "变化"], ["吞吐", "2547", "3268", "+28%"],
             ["首字延迟", "16.8秒", "11.6秒", "-31%"], ["显存池", "35万", "90万", "2.5倍"]]
    specs = (
        [("scan", 2480, 3508, SERIF, 30, 200, None)] * 4 +    # A4 300dpi，小字号（约 10.5pt）
        [("scan", 2480, 3508, SANS, 40, 200, None)] * 4 +     # A4 300dpi，正文字号
        [("shot", 1920, 1080, SANS, 22, 60, table)] * 3 +     # 1080p 截图带表格
        [("shot", 2560, 1440, SANS, 24, 80, table)] * 3 +     # 2K 截图带表格
        [("photo", 3024, 4032, SANS, 46, 260, None)] * 6      # 手机竖拍
    )
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for i, (kind, w, h, fp, sz, mg, tb) in enumerate(specs):
            img, text = page(take(40), w, h, fp, sz, mg, tb)
            if kind == "photo":
                img = photoize(img, rng); name = f"{i:02d}-{kind}.jpg"; img.save(out / name, quality=85)
            else:
                name = f"{i:02d}-{kind}.png"; img.save(out / name, optimize=True)
            mf.write(json.dumps({"file": name, "kind": kind, "w": img.width, "h": img.height, "text": text}, ensure_ascii=False) + "\n")
            print(name, img.size, len(text))


if __name__ == "__main__":
    main()
