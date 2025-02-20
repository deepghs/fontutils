from typing import Union, Tuple

from PIL import Image, ImageDraw, ImageFont


def render_text_with_font(
        text: str,
        font_file: str,
        font_size: int = 24,
        font_color: Union[str, Tuple[int, int, int], Tuple[int, int, int, int]] = 'black',
        line_interval: int = 0, char_interval: int = 0,
):
    custom_font = ImageFont.truetype(font_file, font_size)
    default_font = ImageFont.load_default(font_size)

    lines = text.splitlines(keepends=False)
    supported, unsupported = set(), set()
    line_heights, max_line_width = [], 0
    for line in lines:
        line_width, line_height = 0, 0
        for char in line:
            try:
                bbox = custom_font.getbbox(char)
                if (char.strip() and ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > 0)) or \
                        (not char.strip() and bbox[2] - bbox[0] > 0):
                    supported.add(char)
                else:
                    unsupported.add(char)
            except:
                bbox = None
                unsupported.add(char)

            font = custom_font if char in supported else default_font
            left, top, right, bottom = font.getbbox(char)
            line_width += (0 if line_width == 0 else char_interval) + (right - left)
            line_height = max(line_height, font.getmetrics()[0])

        max_line_width = max(max_line_width, line_width)
        line_heights.append(line_height)

    temp_image = Image.new("RGBA", (
        max_line_width + 50,
        sum(line_heights) + (len(line_heights) - 1) * line_interval + 50
    ), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp_image)

    y = 0
    for line_idx, line in enumerate(lines):
        x = 0
        line_height = line_heights[line_idx]
        for char in line:
            font = custom_font if char in supported else default_font
            draw.text((x, y), char, fill=font_color, font=font)
            left, _, right, _ = font.getbbox(char)
            x += (right - left) + char_interval
        y += line_height + line_interval

    bbox = temp_image.getbbox()
    if not bbox:
        return Image.new("RGBA", (0, 0)), supported, unsupported
    cropped = temp_image.crop(bbox)
    return cropped, supported, unsupported
