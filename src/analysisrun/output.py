"""Figure output contracts and local destinations."""

from pathlib import Path
from typing import Any, LiteralString, Protocol

import matplotlib.figure as fig
import matplotlib.pyplot as plt


class Output(Protocol):
    """
    matplotlib.figure.Figureを保存する。

    Parameters
    ----------
    fig
        保存するFigure。
    name
        保存するファイル名。
    image_type
        画像タイプ。
        実際の画像保存処理のヒントとなります。
    kwargs
        savefigに渡すキーワード引数。
    """

    def __call__(
        self, fig: fig.Figure, name: str, image_type: LiteralString, **kwargs: Any
    ) -> None: ...


class _FileOutput(Output):
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def __call__(
        self, fig: fig.Figure, name: str, image_type: str, **kwargs: Any
    ) -> None:
        path = self.base_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, **kwargs)
        fig.clear()
        plt.close(fig)


class _NullOutput(Output):
    def __call__(
        self, fig: fig.Figure, name: str, image_type: str, **kwargs: Any
    ) -> None:
        fig.clear()
        plt.close(fig)


def _save_image_bytes_to_dir(name: str, image_bytes: bytes, output_dir: Path) -> None:
    path = output_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(image_bytes)
