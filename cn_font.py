# -*- coding: utf-8 -*-
"""中文字形安全字体层（v12.7）

问题背景：
Windows 日文字体（Meiryo / MS Gothic）缺少大量简体中文字形（这、说、么、们、
开、脏、兮、个 ...）。旧版 app.py 的 font_for() 把 meiryo.ttc 放在第一位，
于是这些字被画成 .notdef（□ 方框），看起来像「OCR 识别错」，其实是绘制阶段选错字体的锅。

本模块的做法：
1. 字体链以「简体中文字形完整」的字体优先（微软雅黑 → 黑体 → 等线 → 宋体 → 日文字体兜底）。
2. 真正判断一个字体是否含某个字形：把字渲染出来，和该字体的 .notdef（私用区码位）
   位图逐字节比较 —— 相同即缺字，不靠字体名猜。
3. 整句能被同一字体覆盖时整句一次性绘制（字形/字距最好）；否则逐字回退，
   每个字都挑第一个真正含有它的字体，保证不再出现 □。
4. 缺字查询带缓存，绘制一页的额外开销可以忽略。
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = os.environ.get("JPMT_FONT_DIR") or os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")

# 顺序即优先级：先保证简体中文覆盖完整，再考虑日漫风格，最后是万能兜底字体。
FONT_CHAINS = {
    False: [
        "msyh.ttc", "msyh.ttf", "simhei.ttf", "Deng.ttf", "simsun.ttc",
        "SimSun-ExtB.ttf", "meiryo.ttc", "msgothic.ttc", "arialuni.ttf", "seguisym.ttf", "seguiemj.ttf", "arial.ttf",
    ],
    True: [
        "msyhbd.ttc", "msyh.ttc", "simhei.ttf", "Dengb.ttf", "simsun.ttc",
        "SimSun-ExtB.ttf", "meiryob.ttc", "msgothicb.ttc", "seguisym.ttf", "seguiemj.ttf", "arialbd.ttf",
    ],
}

# 允许用户强制指定字体（可选，例如 JPMT_FONT="C:\\Windows\\Fonts\\simhei.ttf"）
ENV_FONT = os.environ.get("JPMT_FONT")
ENV_FONT_BOLD = os.environ.get("JPMT_FONT_BOLD")

_PROBE_SIZE = 64
_PROBE_PAD = 8
_NOTDEF_CODES = ("\ue05e", "\U000f0000")

_font_cache = {}
_coverage_cache = {}
_notdef_cache = {}

NO_LINE_START = set("，。、！？：；）］｝」』】》〉…‥ー～,.;:!?)]}%‰℃°")
NO_LINE_END = set("（［｛「『【《〈([{")


def _resolve(name):
    if not name:
        return None
    if os.path.isabs(name):
        return name if os.path.exists(name) else None
    p = os.path.join(FONT_DIR, name)
    return p if os.path.exists(p) else None


def chain_paths(bold=False):
    forced = ENV_FONT_BOLD if bold else ENV_FONT
    paths = []
    p = _resolve(forced)
    if p:
        paths.append(p)
    if bold and ENV_FONT:
        p2 = _resolve(ENV_FONT)
        if p2 and p2 not in paths:
            paths.append(p2)
    for name in FONT_CHAINS[True if bold else False]:
        p = _resolve(name)
        if p and p not in paths:
            paths.append(p)
    return paths


def load_font(path, size):
    key = (path, int(size))
    f = _font_cache.get(key)
    if f is None:
        f = ImageFont.truetype(path, size=int(size))
        _font_cache[key] = f
    return f


def _render(path, size, ch):
    img = Image.new("L", (_PROBE_SIZE, _PROBE_SIZE), 0)
    ImageDraw.Draw(img).text((_PROBE_PAD, _PROBE_PAD), ch, font=load_font(path, size), fill=255)
    return img.tobytes()


def _notdef(path, size):
    key = (path, int(size))
    b = _notdef_cache.get(key)
    if b is None:
        b = _render(path, size, _NOTDEF_CODES[0])
        _notdef_cache[key] = b
    return b


def supports(path, size, ch):
    """这个字体是否真的有这个字符的字形。"""
    if ch in " \t\r\n\u3000":
        return True
    key = (path, int(size), ch)
    v = _coverage_cache.get(key)
    if v is not None:
        return v
    try:
        b = _render(path, size, ch)
        ok = b != _notdef(path, size) and b != bytes(len(b))
    except Exception:
        ok = False
    _coverage_cache[key] = ok
    return ok


class TextFont(object):
    """一档字号下的字体集合：测量、换行、绘制都走它，按字回退。"""

    def __init__(self, size, bold=False):
        size = max(1, int(round(size)))
        self.size = size
        self.bold = bool(bold)
        self._entries = []
        for p in chain_paths(self.bold):
            try:
                self._entries.append((p, load_font(p, size)))
            except Exception:
                continue
        if not self._entries:
            try:
                self._entries.append(("<default>", ImageFont.load_default(size=size)))
            except TypeError:
                self._entries.append(("<default>", ImageFont.load_default()))

    def font_for_text(self, text):
        """整句都能覆盖的字体；没有则 None（需要逐字回退）。"""
        need = {ch for ch in text if ch not in " \t\r\n\u3000"}
        if not need:
            return self._entries[0]
        for path, font in self._entries:
            if path == "<default>":
                continue
            if all(supports(path, self.size, ch) for ch in need):
                return path, font
        return None

    def font_for_char(self, ch):
        for path, font in self._entries:
            if path == "<default>" or supports(path, self.size, ch):
                return path, font
        return self._entries[0]

    def missing_chars(self, text):
        """所有字体都缺的字（正常情况应为空）。"""
        out = []
        real = [p for p, _ in self._entries if p != "<default>"]
        for ch in dict.fromkeys(text):
            if ch in " \t\r\n\u3000":
                continue
            if not real or all(not supports(p, self.size, ch) for p in real):
                out.append(ch)
        return out

    def width(self, draw, text, spacing=0):
        if not text:
            return 0.0
        whole = self.font_for_text(text)
        if whole is not None:
            return float(draw.textlength(text, font=whole[1]) + spacing * (len(text) - 1))
        total = 0.0
        for ch in text:
            total += float(draw.textlength(ch, font=self.font_for_char(ch)[1])) + spacing
        return max(0.0, total - spacing)

    def draw(self, draw, xy, text, fill=(0, 0, 0), stroke_width=0, stroke_fill=None, spacing=0):
        if not text:
            return xy[0]
        whole = self.font_for_text(text)
        if whole is not None:
            draw.text(xy, text, font=whole[1], fill=fill,
                      stroke_width=stroke_width, stroke_fill=stroke_fill)
            return xy[0] + float(draw.textlength(text, font=whole[1])) + spacing * (len(text) - 1)
        x, y = float(xy[0]), float(xy[1])
        for ch in text:
            f = self.font_for_char(ch)[1]
            draw.text((x, y), ch, font=f, fill=fill,
                      stroke_width=stroke_width, stroke_fill=stroke_fill)
            x += float(draw.textlength(ch, font=f)) + spacing
        return x - spacing

    def wrap(self, draw, text, max_width, spacing=0):
        """按像素宽度换行，保留显式换行符。"""
        text = str(text or "")
        if not text:
            return []
        lines, cur = [], ""
        for ch in text:
            if ch == "\n":
                if cur.strip():
                    lines.append(cur.strip())
                cur = ""
                continue
            if not cur:
                cur = ch
                continue
            if self.width(draw, cur + ch, spacing) > max_width:
                if ch in NO_LINE_START and len(cur) >= 2:
                    cur += ch
                elif len(cur) >= 2 and cur[-1] in NO_LINE_END:
                    lines.append(cur[:-1].strip())
                    cur = cur[-1] + ch
                else:
                    lines.append(cur.strip())
                    cur = ch
            else:
                cur += ch
        if cur.strip():
            lines.append(cur.strip())
        return lines


def render_self_test(text=None, out_png=None):
    """自检：把一段中文画出来，返回缺字列表。"""
    text = text or "这说么们开关门问题现样点还过对时间学习电脑脏兮兮闭嘴保密哦咯啦呀呢吧辛苦到那里去如果陪我玩全身真的快死了"
    img = Image.new("RGB", (900, 260), (255, 255, 255))
    d = ImageDraw.Draw(img)
    y = 20
    missing = []
    for size, bold in ((48, False), (48, True)):
        tf = TextFont(size, bold=bold)
        missing += tf.missing_chars(text)
        d.rectangle((0, y, 900, y + 70), fill=(30, 30, 30))
        tf.draw(d, (12, y + 8), text, fill=(255, 255, 255))
        y += 110
    if out_png:
        img.save(out_png)
    return missing


if __name__ == "__main__":
    miss = render_self_test(out_png=os.path.join(os.path.dirname(os.path.abspath(__file__)), "font_selftest.png"))
    print("fonts in chain:")
    for p in chain_paths(False):
        print("  ", p)
    print("missing chars:", "".join(miss) if miss else "(none)")
