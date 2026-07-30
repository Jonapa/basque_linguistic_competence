from typing import List

# Allow up to 20 choices (covers ARC variants and similar MC tasks)
letters = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"
]

_LANG_PROMPTS = {
    "eu": ("Galdera", "Erantzuna"),      # Basque
    "en": ("Question", "Answer"),        # English
    "ca": ("Pregunta", "Resposta"),      # Catalan
    "es": ("Pregunta", "Respuesta"),     # Spanish
    "gl": ("Pregunta", "Resposta"),      # Galician
}

def _format_doc(doc, lang: str) -> str:
    """Internal helper to build a multiple-choice prompt dynamically.

    Args:
        doc: Sample dictionary with 'question' and 'choices'.
        lang: Language code key from _LANG_PROMPTS.
    """
    if lang not in _LANG_PROMPTS:
        raise ValueError(f"Unsupported language code: {lang}")
    question_label, answer_label = _LANG_PROMPTS[lang]
    candidates = doc["choices"]["text"]
    num_choices = len(candidates)
    if num_choices < 2:
        raise ValueError("Invalid number of candidates")
    choice_letters = letters[:num_choices]
    formatted_choices = "\n".join(
        f"{choice}: {candidates[i]}" for i, choice in enumerate(choice_letters)
    )
    return f"{question_label}: {doc['question']}\n{formatted_choices}\n{answer_label}:"

# Language-specific entry points expected by task YAMLs
def doc_to_text_eu(doc) -> str:
    return _format_doc(doc, "eu")

def doc_to_text_en(doc) -> str:
    return _format_doc(doc, "en")

def doc_to_text_ca(doc) -> str:
    return _format_doc(doc, "ca")

def doc_to_text_es(doc) -> str:
    return _format_doc(doc, "es")

def doc_to_text_gl(doc) -> str:
    return _format_doc(doc, "gl")

# Backwards-compatible generic (defaults to Basque for existing references)
def doc_to_text(doc) -> str:
    return _format_doc(doc, "eu")

def doc_to_choice(doc) -> List[str]:
    num_choices = len(doc["choices"]["text"])
    if num_choices < 2:
        raise ValueError("Invalid number of candidates")
    return letters[:num_choices]
