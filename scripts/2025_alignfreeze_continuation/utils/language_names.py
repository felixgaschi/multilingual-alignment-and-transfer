import pycountry

def get_alpha2_from_alpha3(code: str):
    """
    Get the 2-letter code for a language, given its 3-letter code (e.g. "eng" -> "en")
    if the 2-letter code doesn't exist, returns the original 3-letter code
    """
    lang = pycountry.languages.get(alpha_3=code)
    if hasattr(lang, "alpha_2"):
        return lang.alpha_2
    return code