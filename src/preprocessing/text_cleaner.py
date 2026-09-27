"""
Step 3 — Conservative text preprocessing for disaster social media posts.

Design principle: this is intentionally CONSERVATIVE. Words like "help",
"trapped", "injured", "urgent", "rescue" carry the signal the whole project
depends on downstream (actionability, urgency, priority), so we never run
stopword removal, stemming, or lemmatization here — those would blur or
delete exactly the words that matter most.

What this module does:
  - strips URLs
  - normalizes @mentions -> keeps a marker (does not delete the fact that
    someone was mentioned, since "@RedCross please help" is meaningful)
  - splits/keeps hashtags as words (#SendHelp -> SendHelp), since hashtags
    frequently carry the actionable content in crisis tweets
  - demojizes emojis to short text descriptions (keeps the signal, e.g. an
    SOS emoji, instead of just deleting it)
  - collapses excessive whitespace
  - strips HTML artifacts (&amp; etc.)
  - normalizes repeated characters (helllllp -> hellp, capped at 2 repeats)
    without destroying the word
  - keeps normal punctuation that can matter for meaning (e.g. "!" for
    urgency) but strips truly noisy special characters

Always preserves the original text alongside the cleaned text so nothing
is lost for later re-inspection or for models that want raw text.
"""

import re
import html
import unicodedata

try:
    import emoji as _emoji_lib
    _HAS_EMOJI_LIB = True
except ImportError:
    _HAS_EMOJI_LIB = False


URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@(\w+)")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")  # 3+ repeats -> collapse to 2
MULTI_SPACE_RE = re.compile(r"\s+")
# Characters we consider noise: keep letters/numbers/basic punctuation
# that can signal urgency or meaning (. , ! ? : ; ' " - / and whitespace)
ALLOWED_PUNCT = ".,!?:;'\"-/()%"
NOISY_CHAR_RE = re.compile(
    r"[^\w\s" + re.escape(ALLOWED_PUNCT) + "]"
)

# Words whose signal must never be stripped by any future aggressive step.
# Documented here so anyone editing this file later sees the constraint.
PROTECTED_KEYWORDS = {
    "help", "trapped", "injured", "urgent", "rescue", "emergency",
    "dying", "critical", "stranded", "evacuate", "evacuation", "flood",
    "fire", "earthquake", "damage", "missing", "dead", "died", "sos",
}


def strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def normalize_html(text: str) -> str:
    return html.unescape(text)


def handle_mentions(text: str, mode: str = "keep_username") -> str:
    """mode='keep_username' turns @RedCross -> RedCross (drops the @ but
    keeps the referenced entity, which can matter, e.g. who help is
    directed at). mode='remove' drops mentions entirely."""
    if mode == "remove":
        return MENTION_RE.sub(" ", text)
    return MENTION_RE.sub(r"\1", text)


def handle_hashtags(text: str) -> str:
    """#SendHelpNow -> SendHelpNow (kept as a token; downstream tokenizers/
    transformers can still split camelCase if desired, but we don't
    destroy the content by deleting the tag)."""
    return HASHTAG_RE.sub(r"\1", text)


def handle_emojis(text: str) -> str:
    """Convert emojis to short text descriptions rather than deleting them,
    since e.g. an ambulance or fire emoji is informative. Falls back to
    leaving emojis in place if the `emoji` package isn't installed."""
    if _HAS_EMOJI_LIB:
        return _emoji_lib.demojize(text, delimiters=(" ", " "))
    return text


def normalize_repeated_chars(text: str) -> str:
    """helllllp -> hellp (cap runs at 2 repeats) — keeps the word
    recognizable/spell-correctable instead of destroying it."""
    return REPEATED_CHAR_RE.sub(r"\1\1", text)


def strip_noisy_chars(text: str) -> str:
    return NOISY_CHAR_RE.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    return MULTI_SPACE_RE.sub(" ", text).strip()


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def clean_text(
    text: str,
    remove_mentions: bool = False,
    demojize: bool = True,
    lowercase: bool = True,
) -> str:
    """
    Full conservative cleaning pipeline for a single post.

    Parameters
    ----------
    text : str
        Raw post text.
    remove_mentions : bool
        If True, delete @mentions entirely instead of keeping the username.
    demojize : bool
        If True, convert emojis to text descriptions instead of leaving
        raw emoji characters (helps downstream TF-IDF/tokenizers).
    lowercase : bool
        If True, lowercase the result. Kept optional because transformer
        models (e.g. cased BERT variants) may want case preserved.

    Returns
    -------
    str : cleaned text. Never raises on empty/None input (returns "").
    """
    if text is None:
        return ""
    text = str(text)

    text = normalize_unicode(text)
    text = normalize_html(text)
    text = strip_urls(text)
    text = handle_mentions(text, mode="remove" if remove_mentions else "keep_username")
    text = handle_hashtags(text)
    if demojize:
        text = handle_emojis(text)
    text = normalize_repeated_chars(text)
    text = strip_noisy_chars(text)
    text = collapse_whitespace(text)

    if lowercase:
        text = text.lower()

    return text


def add_clean_text_columns(df, text_column: str, clean_column: str = "text_clean",
                            original_column: str = "text_original", **clean_kwargs):
    """
    Adds `text_original` (verbatim copy of the source column) and
    `text_clean` (output of clean_text) to a dataframe, without mutating
    the source column itself.
    """
    df = df.copy()
    df[original_column] = df[text_column]
    df[clean_column] = df[text_column].apply(lambda t: clean_text(t, **clean_kwargs))
    return df


if __name__ == "__main__":
    # Quick manual smoke test with representative crisis-tweet-style examples.
    samples = [
        "RT @CNN: People are trraaapped after the flood in Houston!! "
        "Send help ASAP https://t.co/abc123 #HoustonFlood #SendHelp",
        "URGENT: injured survivors near the collapsed bridge, need rescue "
        "immediately!!! @RedCross please respond 🚨🚨",
        "Just a nice sunny day here, no news to report &amp; all is well :) https://example.com",
        None,
        "",
    ]
    for s in samples:
        print("RAW  :", repr(s))
        print("CLEAN:", repr(clean_text(s)))
        print("-" * 60)
