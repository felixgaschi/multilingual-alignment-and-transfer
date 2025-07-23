from typing import List, Union
import pycountry

from datasets import load_dataset, get_dataset_config_names

from multilingual_eval.datasets.token_classification import get_token_classification_getter


get_xtreme_udpos = get_token_classification_getter(
    lambda lang, cache_dir=None: load_dataset(
        "xtreme",
        f"udpos.{pycountry.languages.get(alpha_2=lang).name}",
        cache_dir=cache_dir,
    ),
    "pos_tags",
)

get_xtreme_r_udpos = get_token_classification_getter(
    lambda lang, cache_dir=None: load_dataset(
        "izhx/xtreme-r-udpos",
        f"{lang}",
        cache_dir=cache_dir,
    ),
    "pos_tags",
)

# missing: bn, mk, ms
wuetal_subsets = {
    "en": "en_ewt",
    "af": "af_afribooms",
    "ar": "ar_padt",
    "bg": "bg_btb",
    "ca": "ca_ancora",
    "cs": "cs_pdt",
    "da": "da_ddt",
    "de": "de_gsd",
    "el": "el_gdt",
    "es": "es_gsd",
    "fa": "fa_perdt",
    "fi": "fi_pud",
    "fr": "fr_gsd",
    "he": "he_htb",
    "hi": "hi_hdtb",
    "hu": "hu_szeged",
    "it": "it_isdt",
    "ja": "ja_gsd",
    "ko": "ko_gsd",
    "lv": "lv_lvtb",
    "lt": "lt_alksnis",
    "no": "no_bokmaal",
    "pl": "pl_pdb",
    "pt": "pt_gsd",
    "ro": "ro_rrt",
    "ru": "ru_gsd",
    "sk": "sk_snk",
    "sl": "sl_ssj",
    "sv": "sv_pud",
    "ta": "ta_ttb",
    "th": "th_pud",
    "tr": "tr_pud",
    "uk": "uk_iu",
    "vi": "vi_vtb",
    "zh": "zh_gsd",
}

get_wuetal_udpos = get_token_classification_getter(
    lambda lang, cache_dir=None: load_dataset(
        "universal_dependencies", wuetal_subsets[lang], cache_dir=cache_dir
    ),
    "upos",
)

def get_xtreme_udpos_langs():
    """
    Get the list of availabel languages in xtreme.udpos
    """
    return list(
        map(
            lambda x: "el"
            if x.split(".")[1] == "Greek"
            else pycountry.languages.get(name=x.split(".")[1]).alpha_2,
            filter(lambda x: x.startswith("udpos."), get_dataset_config_names("xtreme")),
        )
    )
