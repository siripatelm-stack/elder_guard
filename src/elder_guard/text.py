"""Turn splitting. Training and the web app MUST both use this exact function."""
import re

MAX_TURNS = 30
_SPEAKER = r"(?=\b(?:caller|receiver|agent|customer|scammer|victim|suspect|innocent)\s*:)"


def split_turns(text):
    if not isinstance(text, str) or not text.strip():
        return []
    if re.search(_SPEAKER, text, flags=re.IGNORECASE):
        parts = re.split(_SPEAKER, text, flags=re.IGNORECASE)
    else:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---- Rule layer: a safety net that does not depend on the trained model ----
# A real bank or police officer never needs these things, so asking for them is a red flag.
RISK_RULES = [
    ("Asks for OTP / PIN / CVV",
     re.compile(r"\b(tell|give|share|send|read|provide|confirm)\b[^.?!]{0,40}\b(otp|one[- ]time[- ]password|cvv|pin)\b", re.I)),
    ("Digital-arrest / don't-hang-up pressure",
     re.compile(r"\bdigital arrest\b|\barrest warrant\b|\bdo not (disconnect|hang up)\b|\bstay on the (call|line)\b", re.I)),
    ("Remote-access request",
     re.compile(r"\b(anydesk|teamviewer|quick ?support|remote access)\b", re.I)),
    ("Gift card / safe-account payment",
     re.compile(r"\bgift cards?\b|\b(safe|secure) account\b", re.I)),
    ("Account-block / KYC threat",
     re.compile(r"\bkyc\b[^.?!]{0,40}\b(update|verify|expire|block|suspend)|\byour account (will be|has been|is) (blocked|suspended|frozen)\b", re.I)),
]


def rule_flags(text):
    """Names of every red-flag rule this turn matches."""
    return [name for name, pattern in RISK_RULES if pattern.search(text)]