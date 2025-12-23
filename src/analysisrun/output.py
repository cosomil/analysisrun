from typing import LiteralString, Optional, Protocol
from pathlib import Path

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
        self, fig: fig.Figure, name: str, image_type: LiteralString, **kwargs
    ) -> None: ...


class DefaultOutput(Output):
    """
    matplotlib.figure.Figureを保存する。
    show=Trueの場合、保存後にNotebookへの表示を実行する。
    """

    def __init__(self, show: bool = False, parent_dir: Optional[Path] = None):
        self._show = show
        self._dir = parent_dir

    def __call__(
        self, fig: fig.Figure, name: str, image_type: LiteralString, **kwargs
    ) -> None:
        # 画像を指定の名前で保存し、さらにNotebook上に表示する。
        filename = name if self._dir is None else self._dir / name
        fig.savefig(filename, **kwargs)
        if self._show:
            plt.show(False)
        fig.clear()
        plt.close(fig)
