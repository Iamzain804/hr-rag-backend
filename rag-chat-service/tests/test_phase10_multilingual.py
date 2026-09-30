import pytest
from app.language_detector import (
    detect_language,
    get_localized_fallback_message,
    get_localized_guardrail_refusal,
    get_localized_scope_refusal,
    LOCALIZED_FALLBACK_MESSAGES,
)
from app.rag_engine import build_system_prompt
from app.schemas import UserContext


def test_language_detection_urdu_script():
    """Verify detection of native Arabic/Urdu script."""
    assert detect_language("مجھے سالانہ کتنی چھٹیاں ملتی ہیں؟") == "urdu"
    assert detect_language("کمپنی کی پالیسی کیا ہے؟") == "urdu"
    assert detect_language("دفتر کے اوقات کیا ہیں؟") == "urdu"


def test_language_detection_roman_urdu():
    """Verify detection of Latin script Roman Urdu queries."""
    assert detect_language("Mujhe kitni annual leaves milti hain?") == "roman_urdu"
    assert detect_language("Mera duty timings kya hai?") == "roman_urdu"
    assert detect_language("Leave policy ke baare mein batao please") == "roman_urdu"
    assert detect_language("Medical insurance kaise claim karein?") == "roman_urdu"
    assert detect_language("Branch 1 ki timing kab shuru hoti hai?") == "roman_urdu"


def test_language_detection_english_default():
    """Verify default fallback to English for standard English questions."""
    assert detect_language("How many days of annual leave do I get?") == "english"
    assert detect_language("What is the company health insurance policy?") == "english"
    assert detect_language("Can I carry forward unused leaves to next year?") == "english"


def test_localized_fallback_messages():
    """Verify localized fallback messages in all 3 languages."""
    assert "isn't covered in company documents" in get_localized_fallback_message("english")
    assert "Company ke documents mein is baare mein koi information" in get_localized_fallback_message("roman_urdu")
    assert "کمپنی کی دستاویزات میں اس بارے میں معلومات موجود نہیں" in get_localized_fallback_message("urdu")


def test_localized_refusal_messages():
    """Verify localized guardrail and scope refusal messages."""
    assert "Main official HR AI Assistant hoon" in get_localized_guardrail_refusal("roman_urdu")
    assert "میں آفیشل ایچ آر اے آئی اسسٹنٹ ہوں" in get_localized_guardrail_refusal("urdu")
    assert "Aapke paas doosri branch ke policy documents dekhne ki permission nahi hai" in get_localized_scope_refusal("roman_urdu")


def test_multilingual_system_prompt_structure():
    """Verify system prompt includes language matching & structured output formatting."""
    user = UserContext(
        user_id=1,
        full_name="Zain Ali",
        email="zain@techcorp.com",
        role="Employee",
        branch_id=1,
        branch_name="London",
        department_id=1,
        department_name="Engineering",
    )
    prompt = build_system_prompt(user, detected_language="roman_urdu")
    
    assert "LANGUAGE & SCRIPT MATCHING" in prompt
    assert "STRUCTURED OUTPUT FORMATTING" in prompt
    assert "Markdown Tables" in prompt
    assert "Bullet Points" in prompt
    assert "Target Response Language for this query: ROMAN_URDU" in prompt
