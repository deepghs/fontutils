import os.path
from functools import lru_cache
from typing import Optional

from .freq import BayesianFreqAnalysis
from ..data import FontTyping


@lru_cache()
def _load_bayesian():
    return BayesianFreqAnalysis(
        freq_csv_gz_file=os.path.join(os.path.dirname(__file__), 'systems_count.csv.gz'),
        prefix='slang_',
    )


def font_systems_comp(font: FontTyping, select: Optional[str] = None):
    return _load_bayesian().font_comp(font=font, select=select)


def font_systems_prob(font: FontTyping, select: Optional[str] = None, topk: int = 5):
    return _load_bayesian().font_prob(
        font=font,
        select=select,
        topk=topk,
    )
