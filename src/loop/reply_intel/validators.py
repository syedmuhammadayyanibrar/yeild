import re

from langdetect import DetectorFactory, detect

from loop.schema import ReplyClass

DetectorFactory.seed = 0

def check_language_matches(draft: str, expected_lang: str) -> bool:
    if not draft or not draft.strip():
        return True
    try:
        detected = detect(draft)
        if expected_lang == "en" and detected == "en":
            return True
        if expected_lang == "fr" and detected == "fr":
            return True
        if expected_lang == "de" and detected == "de":
            return True
        if expected_lang == "es" and detected == "es":
            return True
        return False
    except Exception:
        return True

def check_word_count_limit(draft: str, max_words: int = 90) -> bool:
    words = draft.strip().split()
    return len(words) <= max_words

def check_interested_slots_or_calendar(draft: str) -> bool:
    draft_lower = draft.lower()
    if "calendly" in draft_lower or "calendar" in draft_lower or "link" in draft_lower or "cal.com" in draft_lower or "[calendar]" in draft_lower or "agenda" in draft_lower:
        return True
    time_matches = re.findall(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|h)?|\b(?:monday|tuesday|wednesday|thursday|friday|lundi|mardi|mercredi|jeudi|vendredi|montag|dienstag|mittwoch|donnerstag|freitag|lunes|martes|miércoles|jueves|viernes))\b", draft_lower)
    return len(time_matches) >= 2

def check_unsubscribe_removal_no_pitch(draft: str) -> bool:
    draft_lower = draft.lower()
    confirm_words = ["removed", "unsubscribed", "opted out", "supprimé", "désinscrit", "ausgetragen", "gelöscht", "eliminado", "desuscrito"]
    has_confirm = any(w in draft_lower for w in confirm_words)
    pitch_words = ["special offer", "discount", "demo", "buy now", "platform features", "notre offre", "rabatt", "descuento"]
    has_pitch = any(w in draft_lower for w in pitch_words)
    return has_confirm and not has_pitch

def check_wrong_person_ask_contact(draft: str) -> bool:
    draft_lower = draft.lower()
    ask_words = ["who", "right person", "correct contact", "point me", "qui", "bonne personne", "wer", "richtige ansprechpartner", "quién", "contacto adecuado", "recomendar"]
    return any(w in draft_lower for w in ask_words)

def check_referral_thanks_and_contact(
    draft: str,
    referral_name: str | None = None,
    referral_email: str | None = None,
) -> bool:
    draft_lower = draft.lower()
    thanks_words = ["thank", "thanks", "merci", "danke", "gracias"]
    has_thanks = any(w in draft_lower for w in thanks_words)
    if not has_thanks:
        return False
    if referral_name and referral_name.lower() in draft_lower:
        return True
    if referral_email and referral_email.lower() in draft_lower:
        return True
    return any(w in draft_lower for w in ["contact", "loop", "reach out", "connecter", "wenden", "comunicaré"])

def check_no_hallucinated_claims(draft: str, context_text: str) -> bool:
    draft_numbers = set(re.findall(r"\b\d{3,}\b", draft))
    context_numbers = set(re.findall(r"\b\d{3,}\b", context_text))
    return draft_numbers.issubset(context_numbers)

def run_all_validators(
    draft: str | None,
    reply_class: ReplyClass,
    language: str,
    context_text: str,
    referral_name: str | None = None,
    referral_email: str | None = None,
) -> dict[str, bool]:
    if reply_class == "out_of_office" or draft is None:
        return {
            "language_matches": True,
            "word_count_limit": True,
            "class_specific_rule": True,
            "no_hallucinations": True,
        }

    lang_pass = check_language_matches(draft, language)
    words_pass = check_word_count_limit(draft, 90)
    claims_pass = check_no_hallucinated_claims(draft, context_text)

    class_rule_pass = True
    if reply_class == "interested":
        class_rule_pass = check_interested_slots_or_calendar(draft)
    elif reply_class == "unsubscribe":
        class_rule_pass = check_unsubscribe_removal_no_pitch(draft)
    elif reply_class == "wrong_person":
        class_rule_pass = check_wrong_person_ask_contact(draft)
    elif reply_class == "referral":
        class_rule_pass = check_referral_thanks_and_contact(draft, referral_name, referral_email)

    return {
        "language_matches": lang_pass,
        "word_count_limit": words_pass,
        "class_specific_rule": class_rule_pass,
        "no_hallucinations": claims_pass,
    }
