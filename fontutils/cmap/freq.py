from typing import Optional, Set

import numpy as np
import pandas as pd

from ..data import load_font_with_soft_close, FontTyping


class BayesianFreqAnalysis:
    def __init__(self, freq_csv_gz_file: str, prefix: str):
        self.df = pd.read_csv(freq_csv_gz_file, compression='gzip')
        self.prefix = prefix
        self._d_cache_select = {}
        self._d_cache_cinfo = {}

    def _load_count_select(self, select: Optional[str] = None):
        if select not in self._d_cache_select:
            df = self.df
            if select:
                if len(select) == 1:
                    df = df[df['type'] == select]
                elif len(select) == 2:
                    df = df[df['category'] == select]
                else:
                    raise ValueError(f'Unknown character selector - {select!r}.')
            self._d_cache_select[select] = df
        return self._d_cache_select[select]

    def _load_counts_info(self, select: Optional[str] = None):
        if select not in self._d_cache_cinfo:
            df = self._load_count_select(select)
            lang_names, count_list = [], []
            for column in df.columns:
                if column.startswith(self.prefix):
                    lang_name = column[len(self.prefix):]
                    counts = np.array(df[column])
                    lang_names.append(lang_name)
                    count_list.append(counts)

            lang_names = np.array(lang_names)  # L
            char_ids = np.array(df['id'])  # C
            count_list = np.stack(count_list)
            char_lang_freq = count_list / count_list.sum(axis=-1, keepdims=True)  # LxC

            lang_counts = count_list.sum(axis=-1)
            lang_ratios = lang_counts / lang_counts.sum()
            char_counts = count_list.sum(axis=0)
            char_ratios = char_counts / char_counts.sum()

            # bayesian formula, LxC
            char_lang_prob = char_lang_freq * lang_ratios[..., None] / char_ratios[None, ...]
            self._d_cache_cinfo[select] = (lang_names, char_ids, char_lang_freq, char_lang_prob)
        return self._d_cache_cinfo[select]

    def _load_cmap_mask(self, cmap_set: Set[int], select: Optional[str] = None):
        df = self._load_count_select(select)
        id_list = df['id']
        included_list = id_list.map(lambda x: x in cmap_set)
        return np.array(included_list)

    def font_comp(self, font: FontTyping, select: Optional[str] = None):
        font, fn_close = load_font_with_soft_close(font)
        try:
            char_mask = self._load_cmap_mask(set(font.getBestCmap().keys()), select=select).astype(np.float32)
            lang_names, _, freq_list, _ = self._load_counts_info(select=select)
            ratios = (char_mask[None, ...] * freq_list).sum(axis=-1)
            # noinspection PyTypeChecker
            d_ratios = {lang_name: ratio for lang_name, ratio in zip(lang_names.tolist(), ratios.tolist())}
            return d_ratios
        finally:
            fn_close()

    def font_prob(self, font: FontTyping, select: Optional[str] = None, topk: int = 5):
        font, fn_close = load_font_with_soft_close(font)
        try:
            char_mask = self._load_cmap_mask(set(font.getBestCmap().keys()), select=select)
            lang_names, _, _, freq_prob = self._load_counts_info(select=select)
            masked_freq_prob = freq_prob[:, char_mask]
            mean_freq_prob = masked_freq_prob.mean(axis=-1)
            zero_mask = mean_freq_prob > 0.0
            lang_names = lang_names[zero_mask]
            mean_freq_prob = mean_freq_prob[zero_mask]

            sort_ids = np.argsort(-mean_freq_prob)
            if sort_ids.shape[0] > topk:
                sort_ids = sort_ids[:topk]
            lang_names = lang_names[sort_ids]
            mean_freq_prob = mean_freq_prob[sort_ids]
            d_probs = {lang_name: prob for lang_name, prob in zip(lang_names.tolist(), mean_freq_prob.tolist())}
            return d_probs
        finally:
            fn_close()
