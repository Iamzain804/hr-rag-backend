import re
import logging
from typing import List, Set
from fastapi import HTTPException, status

from app.config import settings

logger = logging.getLogger("rag_chat_service.guardrails")

# Known prompt injection & jailbreak signature patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"override\s+(the\s+)?system\s+prompt",
    r"you\s+are\s+now\s+(a\s+)?dan\b",
    r"jailbreak",
    r"system\s*:\s*override",
    r"bypass\s+(safety|content)\s+filters",
    r"pretend\s+to\s+be\s+an\s+unrestricted",
    r"act\s+as\s+an\s+unaligned\s+ai",
    r"reveal\s+(your\s+)?system\s+prompt",
]

COMPILED_INJECTION_REGEX = [
    re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS
]

# Stop words for groundedness term overlap
STOP_WORDS: Set[str] = {
    "the", "and", "is", "in", "to", "of", "it", "with", "as", "for", "on", "at",
    "by", "this", "that", "from", "are", "be", "an", "or", "which", "will", "can",
    "should", "would", "have", "has", "had", "not", "but", "you", "your", "our",
    "we", "they", "them", "their", "all", "any", "some", "what", "how", "when",
    "where", "why", "who", "whom", "about", "after", "before", "during", "company",
    "document", "documents", "policy", "policies", "based", "according", "stated",
    "employee", "employees", "information", "verified", "unverified"
}


def validate_input_message(message: str) -> None:
    """
    Validate incoming user chat message for safety, length, and injection.

    Capabilities:
    - Rejects empty or whitespace-only messages.
    - Rejects excessively long messages exceeding MAX_MESSAGE_LENGTH.
    - Catches common prompt injection phrases (e.g. 'ignore previous instructions',
      'jailbreak', 'DAN mode', 'system prompt override').

    Limitations:
    - Does NOT guarantee catching sophisticated multi-turn semantic jailbreaks,
      heavily encoded/obfuscated ciphers (e.g. Base64, ROT13), or novel zero-day
      adversarial suffixes.
    """
    if not message or not message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chat message cannot be empty.",
        )

    clean_message = message.strip()
    if len(clean_message) > settings.MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Message exceeds maximum allowed length of {settings.MAX_MESSAGE_LENGTH} characters.",
        )

    for regex in COMPILED_INJECTION_REGEX:
        if regex.search(clean_message):
            logger.warning(f"Rejected prompt injection attempt matching pattern: {regex.pattern}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input detected. Prompt injection and instruction overrides are strictly prohibited.",
            )


def validate_output_message(output: str) -> str:
    """
    Ensure generated response conforms to length and safety bounds.
    """
    if not output or not output.strip():
        return settings.FALLBACK_MESSAGE

    if len(output) > settings.MAX_OUTPUT_LENGTH:
        logger.warning(f"Generated output exceeded length cap ({len(output)}). Truncating.")
        return output[: settings.MAX_OUTPUT_LENGTH] + "\n\n...[Response truncated for length]"

    return output


def check_groundedness_heuristic(generated_text: str, context_text: str) -> bool:
    """
    Best-effort post-generation heuristic to verify that key terminology in the
    generated answer is anchored in the provided context (company documents or attachment).

    Capabilities:
    - Extracts significant alphanumeric tokens (> 3 characters) from the generated answer.
    - Calculates term overlap percentage against the combined context.
    - Flags answers that drift entirely into unsupported topics.

    Limitations:
    - This is a statistical token-overlap heuristic, NOT a formal logical entailment engine.
    - May allow subtle factual inversions (e.g. 'is' vs 'is not') if words match.
    - Treats numbers and proper nouns with high weight.
    """
    if not generated_text:
        return False

    # If the response is the standard fallback or refusal, it is grounded by definition
    if (
        settings.FALLBACK_MESSAGE in generated_text
        or "don't have this information" in generated_text.lower()
        or "not mentioned in" in generated_text.lower()
    ):
        return True

    if not context_text or not context_text.strip():
        return False

    # Extract words
    gen_words = re.findall(r"\b[a-zA-Z0-9_\$]{4,}\b", generated_text.lower())
    context_words = set(re.findall(r"\b[a-zA-Z0-9_\$]{4,}\b", context_text.lower()))

    # Filter stop words
    significant_gen_words = [w for w in gen_words if w not in STOP_WORDS]
    if not significant_gen_words:
        return True

    overlap_count = sum(1 for w in significant_gen_words if w in context_words)
    overlap_ratio = overlap_count / len(significant_gen_words)

    # If overlap is extremely low (< 15%), answer is likely hallucinated
    if overlap_ratio < 0.15:
        logger.warning(
            f"Groundedness check failed: overlap ratio {overlap_ratio:.2f} ({overlap_count}/{len(significant_gen_words)} terms)"
        )
        return False

    return True
