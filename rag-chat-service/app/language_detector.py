import re
from typing import Dict, List, Set

# Comprehensive Roman Urdu keywords & grammatical markers
ROMAN_URDU_KEYWORDS: Set[str] = {
    # Question words
    "kya", "kia", "kyun", "kyu", "kaise", "kese", "kab", "kahan", "kidhar", "kitna", "kitne", "kitni",
    "kis", "kisko", "kisi", "kaun", "kon", "konsa", "konsi", "konse",
    # Pronouns
    "mujhe", "mera", "meri", "mere", "humein", "hum", "humara", "humari", "humare",
    "aap", "ap", "aapko", "apko", "aapka", "apka", "aapki", "apki", "aapke", "apke",
    "tum", "tumhe", "tumhara", "woh", "wo", "unka", "unki", "unke", "unko", "inhe", "inka", "inki", "inke",
    # Auxiliary verbs / Verbs / Modals
    "hai", "hain", "hoga", "hogi", "honge", "chahiye", "chahta", "chahti", "chahte",
    "milta", "milti", "milte", "milega", "milegi", "milegay", "milenge",
    "karna", "karein", "kare", "karti", "karte", "karta", "karo", "kariye",
    "hona", "hoti", "hota", "hote", "raha", "rahi", "rahe",
    "batao", "batayein", "bataiye", "bata", "dena", "deta", "deti", "dete", "dega", "degi",
    "pata", "samajh", "chale", "rakhe", "sakta", "sakti", "sakte", "chahiye",
    # Prepositions / Conjunctions / Particles
    "mein", "me", "main", "par", "pe", "se", "ko", "ki", "ka", "ke", "aur", "ya",
    "lekin", "magar", "agar", "to", "toh", "bhi", "nahi", "nahin", "na", "mat",
    "wala", "wali", "wale", "kuch", "bohot", "zyada", "kam", "sirf", "bhi",
    # Common HR & Domain vocabulary in Roman Urdu
    "chutti", "chuttiyan", "chutiyan", "ijazat", "tanwah", "tankhwah", "mulazim", "mulazmeen", "daftari"
}

# Regex to detect Arabic/Urdu native script characters
URDU_SCRIPT_REGEX = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def detect_language(text: str) -> str:
    """
    Detect language/script of query text:
    - 'urdu': Native Arabic/Urdu script
    - 'roman_urdu': Latin script with Urdu vocabulary and grammar
    - 'english': Default fallback
    
    Known heuristic limitations (documented honestly per Phase 10):
    1. Short ambiguous keywords like 'leave policy' (pure English) vs 'leave policy kya hai' (Roman Urdu).
    2. Single-word queries like 'timings' (English) vs 'chuttiyan' (Roman Urdu).
    3. Code switching / mixed queries: if at least 1 strong Roman Urdu functional word is present,
       it classifies as 'roman_urdu'.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return "english"

    # 1. Native Urdu script check
    urdu_chars = URDU_SCRIPT_REGEX.findall(cleaned)
    if len(urdu_chars) >= 2 or (len(cleaned) <= 5 and len(urdu_chars) >= 1):
        return "urdu"

    # 2. Roman Urdu vocabulary match check
    # Tokenize into clean lowercase words
    tokens = re.findall(r"\b[a-zA-Z]+\b", cleaned.lower())
    if not tokens:
        return "english"

    matched_keywords = [t for t in tokens if t in ROMAN_URDU_KEYWORDS]
    match_count = len(matched_keywords)
    match_ratio = match_count / len(tokens)

    # If sentence has strong Roman Urdu markers (>= 1 strong keyword in short queries, or >= 12% ratio, or >= 2 keywords)
    if (match_count >= 1 and (len(tokens) <= 6 or match_ratio >= 0.12)) or match_count >= 2:
        return "roman_urdu"

    return "english"


# Localized Fallback / Grounding Messages
LOCALIZED_FALLBACK_MESSAGES: Dict[str, str] = {
    "english": "This isn't covered in company documents. Would you like this flagged to HR?",
    "roman_urdu": "Company ke documents mein is baare mein koi information mojood nahi hai. Kya aap chahte hain ke ise HR ko flag kar diya jaye?",
    "urdu": "کمپنی کی دستاویزات میں اس بارے میں معلومات موجود نہیں ہیں۔ کیا آپ چاہتے ہیں کہ اسے ایچ آر (HR) کو بھیج دیا جائے؟",
}

# Localized Guardrail / Refusal Messages
LOCALIZED_GUARDRAIL_REFUSALS: Dict[str, str] = {
    "english": "I am the official HR AI Assistant. I can only assist with company HR policies, leave rules, and employee guidelines.",
    "roman_urdu": "Main official HR AI Assistant hoon. Main sirf company ki HR policies, leave rules, aur employee guidelines ke mutaliq madad kar sakta hoon.",
    "urdu": "میں آفیشل ایچ آر اے آئی اسسٹنٹ ہوں۔ میں صرف کمپنی کی ایچ آر پالیسیوں، چھٹیوں کے قوانین اور ملازمین کے رہنما خطوط میں مدد کر سکتا ہوں۔",
}

# Localized Scope Isolation / Permission Messages
LOCALIZED_SCOPE_REFUSALS: Dict[str, str] = {
    "english": "You do not have permission to view policy documents belonging to another branch.",
    "roman_urdu": "Aapke paas doosri branch ke policy documents dekhne ki permission nahi hai.",
    "urdu": "آپ کے پاس کسی دوسری برانچ کی پالیسی دستاویزات دیکھنے کی اجازت نہیں ہے۔",
}


def get_localized_fallback_message(lang: str) -> str:
    """Retrieve localized fallback message when information is not in docs."""
    return LOCALIZED_FALLBACK_MESSAGES.get(lang.lower(), LOCALIZED_FALLBACK_MESSAGES["english"])


def get_localized_guardrail_refusal(lang: str) -> str:
    """Retrieve localized guardrail refusal message."""
    return LOCALIZED_GUARDRAIL_REFUSALS.get(lang.lower(), LOCALIZED_GUARDRAIL_REFUSALS["english"])


def get_localized_scope_refusal(lang: str) -> str:
    """Retrieve localized scope permission refusal message."""
    return LOCALIZED_SCOPE_REFUSALS.get(lang.lower(), LOCALIZED_SCOPE_REFUSALS["english"])
