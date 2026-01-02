from typing import Dict, TypeGuard
import unicodedata

import pandas as pd
from pandas._typing import FilePath, ReadCsvBuffer


def read_dict(
    filepath_or_buffer: FilePath | ReadCsvBuffer[str] | ReadCsvBuffer[bytes],
    key: str,
    value: str,
) -> Dict[str, str]:
    """
    CSVファイルを読み込み、指定したカラムをキーと値にして辞書を作成する。

    Parameters
    ----------
    filepath_or_buffer
        読み込むCSVファイル
    key
        辞書のキーとなるカラム名
    value
        辞書の値となるカラム名
    """

    df = pd.read_csv(filepath_or_buffer, dtype=str)
    return dict(zip(df[key], df[value]))


def is_float(x: float | None) -> TypeGuard[float]:
    return x is not None


def cowsay(text: str) -> str:
    """
    メッセージをcowsay風にフォーマットする

    Parameters
    ----------
    text
        メッセージ

    Returns
    -------
    str
        cowsay風にフォーマットされたメッセージ
    """

    max_line_width = 50
    raw_lines = text.splitlines() or [""]
    min_cow_left_pad = 4
    _cow = r"""
\   ^__^
 \  (oo)\_______
    (__)\       )\/\
        ||----w |
        ||     ||
"""

    def _char_width(ch: str) -> int:
        if unicodedata.combining(ch):
            return 0
        return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1

    def _display_width(s: str) -> int:
        return sum(_char_width(ch) for ch in s)

    def _is_ascii_word_char(ch: str) -> bool:
        return ch.isascii() and (ch.isalnum() or ch in "_-")

    def _wrap_line(line: str) -> list[str]:
        if not line:
            return [""]
        chunks: list[str] = []
        current: list[str] = []
        width = 0
        just_wrapped = False
        i = 0
        n = len(line)
        while i < n:
            ch = line[i]
            if just_wrapped and ch.isspace():
                i += 1
                continue
            if _is_ascii_word_char(ch):
                j = i + 1
                while j < n and _is_ascii_word_char(line[j]):
                    j += 1
                word = line[i:j]
                word_width = _display_width(word)
                if width and width + word_width > max_line_width:
                    chunks.append("".join(current))
                    current = []
                    width = 0
                    just_wrapped = True
                if word_width > max_line_width:
                    for wch in word:
                        chw = _char_width(wch)
                        if width + chw > max_line_width and current:
                            chunks.append("".join(current))
                            current = []
                            width = 0
                            just_wrapped = True
                        current.append(wch)
                        width += chw
                        just_wrapped = False
                else:
                    current.append(word)
                    width += word_width
                    just_wrapped = False
                i = j
                continue

            chw = _char_width(ch)
            if width + chw > max_line_width and current:
                chunks.append("".join(current))
                current = []
                width = 0
                just_wrapped = True
                if ch.isspace():
                    i += 1
                    continue
            current.append(ch)
            width += chw
            just_wrapped = False
            i += 1
        if current or not chunks:
            chunks.append("".join(current))
        return chunks

    lines: list[str] = []
    for line in raw_lines:
        lines.extend(_wrap_line(line))

    widths = [_display_width(line) for line in lines]
    max_width = max(widths)

    bubble: list[str] = []
    if len(lines) == 1:
        bubble.append(f" {'_' * (max_width + 2)}")
        bubble.append(f"< {lines[0]}{' ' * (max_width - widths[0])} >")
        bubble.append(f" {'-' * (max_width + 2)}")
    else:
        bubble.append(f" {'_' * (max_width + 2)}")
        for idx, (line, width) in enumerate(zip(lines, widths)):
            pad = " " * (max_width - width)
            if idx == 0:
                bubble.append(f"/ {line}{pad} \\")
            elif idx == len(lines) - 1:
                bubble.append(f"\\ {line}{pad} /")
            else:
                bubble.append(f"| {line}{pad} |")
        bubble.append(f" {'-' * (max_width + 2)}")

    cow_lines = _cow.strip("\n").splitlines()
    bubble_width = max(_display_width(line) for line in bubble) if bubble else 0
    cow_width = max(_display_width(line) for line in cow_lines) if cow_lines else 0

    left_pad = max((bubble_width - cow_width) // 2, min_cow_left_pad)
    padded_cow = [(" " * left_pad) + line for line in cow_lines]

    return "\n".join(bubble + padded_cow)
