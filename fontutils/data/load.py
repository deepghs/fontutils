from os import PathLike
from typing import Union, Tuple, Callable

from fontTools.ttLib import TTFont

FontTyping = Union[str, PathLike, TTFont]


def load_font(font: FontTyping) -> TTFont:
    if isinstance(font, TTFont):
        return font
    else:
        return TTFont(file=font)


def load_font_with_soft_close(font: FontTyping) -> Tuple[TTFont, Callable[[], None]]:
    if isinstance(font, TTFont):
        return font, lambda: None
    else:
        font = TTFont(file=font)
        return font, font.close
