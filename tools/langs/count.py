import os.path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from fontutils.cmap import __file__ as _cmap_file

writing_system_mapping = {
    'af': 'latin',
    'am': 'ethiopic',
    'ar': 'arabic',
    'as': 'bengali',
    'az': 'latin',
    'be': 'cyrillic',
    'bg': 'cyrillic',
    'bn': 'bengali',
    'bn_rom': 'latin',
    'br': 'latin',
    'bs': 'latin',
    'ca': 'latin',
    'cs': 'latin',
    'cy': 'latin',
    'da': 'latin',
    'de': 'latin',
    'el': 'greek',
    'en': 'latin',
    'eo': 'latin',
    'es': 'latin',
    'et': 'latin',
    'eu': 'latin',
    'fa': 'arabic',
    'ff': 'latin',
    'fi': 'latin',
    'fr': 'latin',
    'fy': 'latin',
    'ga': 'latin',
    'gd': 'latin',
    'gl': 'latin',
    'gn': 'latin',
    'gu': 'gujarati',
    'ha': 'latin',
    'he': 'hebrew',
    'hi': 'devanagari',
    'hi_rom': 'latin',
    'hr': 'latin',
    'ht': 'latin',
    'hu': 'latin',
    'hy': 'armenian',
    'id': 'latin',
    'ig': 'latin',
    'is': 'latin',
    'it': 'latin',
    'ja': 'japanese',
    'jv': 'latin',
    'ka': 'georgian',
    'kk': 'cyrillic',
    'km': 'khmer',
    'kn': 'kannada',
    'ko': 'hangeul',
    'ku': 'latin',
    'ky': 'cyrillic',
    'la': 'latin',
    'lg': 'latin',
    'li': 'latin',
    'ln': 'latin',
    'lo': 'lao',
    'lt': 'latin',
    'lv': 'latin',
    'mg': 'latin',
    'mk': 'cyrillic',
    'ml': 'malayalam',
    'mn': 'cyrillic',
    'mr': 'devanagari',
    'ms': 'latin',
    # 'my': 'myanmar_unicode',
    # 'my_zaw': 'myanmar_zawgyi',
    'my': 'myanmar',
    'my_zaw': 'myanmar',
    'ne': 'devanagari',
    'nl': 'latin',
    'no': 'latin',
    'ns': 'latin',
    'om': 'latin',
    'or': 'oriya',
    'pa': 'gurmukhi',
    'pl': 'latin',
    'ps': 'arabic',
    'pt': 'latin',
    'qu': 'latin',
    'rm': 'latin',
    'ro': 'latin',
    'ru': 'cyrillic',
    'sa': 'devanagari',
    'sc': 'latin',
    'sd': 'arabic',
    'si': 'sinhala',
    'sk': 'latin',
    'sl': 'latin',
    'so': 'latin',
    'sq': 'latin',
    'sr': 'cyrillic',
    'ss': 'latin',
    'su': 'latin',
    'sv': 'latin',
    'sw': 'latin',
    'ta': 'tamil',
    'ta_rom': 'latin',
    'te': 'telugu',
    'te_rom': 'latin',
    'th': 'thai',
    'tl': 'latin',
    'tn': 'latin',
    'tr': 'latin',
    'ug': 'arabic',
    'uk': 'cyrillic',
    'ur': 'arabic',
    'ur_rom': 'latin',
    'uz': 'latin',
    'vi': 'latin',
    'wo': 'latin',
    'xh': 'latin',
    'yi': 'hebrew',
    'yo': 'latin',
    # 'zh-Hans': 'han_simplified',
    # 'zh-Hant': 'han_traditional',
    'zh-Hans': 'han',
    'zh-Hant': 'han',
    'zu': 'latin'
}

_CMAP_DIR = os.path.dirname(os.path.join(_cmap_file))
_CPREFIX = 'clang_'


def sync_clangs_file():
    df_chars = pd.read_parquet(hf_hub_download(
        repo_id='deepghs/cc100_char_freq',
        repo_type='dataset',
        filename='clangs.parquet'
    ))
    del df_chars['char']

    dst_chars_csv_file = os.path.join(_CMAP_DIR, 'langs_count.csv.gz')
    df_chars.to_csv(dst_chars_csv_file, compression='gzip', index=False)

    d_systems = {}
    for column in df_chars.columns:
        if not column.startswith(_CPREFIX):
            d_systems[column] = df_chars[column]
        else:
            new_name = 'slang_' + writing_system_mapping[column[len(_CPREFIX):]]
            if new_name not in d_systems:
                d_systems[new_name] = np.array(df_chars[column])
            else:
                d_systems[new_name] = d_systems[new_name] + np.array(df_chars[column])
    df_systems = pd.DataFrame(d_systems)
    dst_systems_csv_file = os.path.join(_CMAP_DIR, 'systems_count.csv.gz')
    df_systems.to_csv(dst_systems_csv_file, compression='gzip', index=False)


if __name__ == '__main__':
    sync_clangs_file()
