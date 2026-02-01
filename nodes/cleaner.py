"""
Text cleaning node - sanitizes text for TTS by removing/replacing problematic characters.

Ensures all output is clean ASCII suitable for TTS inference:
- Converts Unicode to ASCII equivalents
- Fixes RTF conversion artifacts
- Removes non-speakable characters
- Expands abbreviations and symbols to spoken form
"""

import re
import unicodedata
from typing import Optional

from ..state import AudiobookState, WorkflowStage


# ============================================================================
# CHARACTER MAPPINGS
# ============================================================================

# Characters to remove entirely (control chars, zero-width, etc.)
REMOVE_CHARS = set([
    "\x00", "\x01", "\x02", "\x03", "\x04", "\x05", "\x06", "\x07", "\x08",
    "\x0b", "\x0c", "\x0e", "\x0f", "\x10", "\x11", "\x12", "\x13", "\x14",
    "\x15", "\x16", "\x17", "\x18", "\x19", "\x1a", "\x1b", "\x1c", "\x1d",
    "\x1e", "\x1f", "\x7f",
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\u2060",  # Word joiner
    "\ufeff",  # BOM
    "\ufffc",  # Object replacement character
    "\ufffd",  # Replacement character
])

# Unicode to ASCII replacements
UNICODE_TO_ASCII = {
    # === Quotes ===
    "\u2018": "'",    # Left single quote '
    "\u2019": "'",    # Right single quote '
    "\u201a": "'",    # Single low-9 quote ‚
    "\u201b": "'",    # Single high-reversed-9 quote ‛
    "\u201c": '"',    # Left double quote "
    "\u201d": '"',    # Right double quote "
    "\u201e": '"',    # Double low-9 quote „
    "\u201f": '"',    # Double high-reversed-9 quote ‟
    "\u00ab": '"',    # Left guillemet «
    "\u00bb": '"',    # Right guillemet »
    "\u2039": "'",    # Single left guillemet ‹
    "\u203a": "'",    # Single right guillemet ›
    "\u0060": "'",    # Grave accent `
    "\u00b4": "'",    # Acute accent ´
    "\u02bc": "'",    # Modifier letter apostrophe ʼ
    "\u02bb": "'",    # Modifier letter turned comma ʻ

    # === Dashes and Hyphens ===
    "\u2010": "-",    # Hyphen ‐
    "\u2011": "-",    # Non-breaking hyphen ‑
    "\u2012": "-",    # Figure dash ‒
    "\u2013": "-",    # En dash –
    "\u2014": " - ",  # Em dash — (with spaces for TTS pause)
    "\u2015": " - ",  # Horizontal bar ―
    "\u2212": "-",    # Minus sign −
    "\u2e3a": " - ",  # Two-em dash ⸺
    "\u2e3b": " - ",  # Three-em dash ⸻
    "\u00ad": "",     # Soft hyphen (remove)

    # === Ellipsis and Dots ===
    "\u2026": "...",  # Horizontal ellipsis …
    "\u22ef": "...",  # Midline horizontal ellipsis ⋯

    # === Spaces ===
    "\u00a0": " ",    # Non-breaking space
    "\u2002": " ",    # En space
    "\u2003": " ",    # Em space
    "\u2004": " ",    # Three-per-em space
    "\u2005": " ",    # Four-per-em space
    "\u2006": " ",    # Six-per-em space
    "\u2007": " ",    # Figure space
    "\u2008": " ",    # Punctuation space
    "\u2009": " ",    # Thin space
    "\u200a": " ",    # Hair space
    "\u202f": " ",    # Narrow no-break space
    "\u205f": " ",    # Medium mathematical space
    "\u3000": " ",    # Ideographic space

    # === Bullets and List Markers ===
    "\u2022": "-",    # Bullet •
    "\u2023": "-",    # Triangular bullet ‣
    "\u2043": "-",    # Hyphen bullet ⁃
    "\u204c": "-",    # Black leftwards bullet ⁌
    "\u204d": "-",    # Black rightwards bullet ⁍
    "\u2219": "-",    # Bullet operator ∙
    "\u25aa": "-",    # Black small square ▪
    "\u25cf": "-",    # Black circle ●
    "\u25e6": "-",    # White bullet ◦
    "\u2619": "-",    # Reversed rotated floral heart bullet ☙

    # === Arrows (convert to words) ===
    "\u2192": " to ",       # Right arrow →
    "\u2190": " from ",     # Left arrow ←
    "\u2194": " and ",      # Left-right arrow ↔
    "\u21d2": " implies ",  # Double right arrow ⇒
    "\u21d0": " from ",     # Double left arrow ⇐
    "\u21d4": " if and only if ",  # Double left-right arrow ⇔

    # === Math Symbols (convert to words) ===
    "\u00d7": " times ",           # Multiplication ×
    "\u00f7": " divided by ",      # Division ÷
    "\u00b1": " plus or minus ",   # Plus-minus ±
    "\u2264": " less than or equal to ",     # ≤
    "\u2265": " greater than or equal to ",  # ≥
    "\u2260": " not equal to ",    # ≠
    "\u2248": " approximately ",   # ≈
    "\u221e": " infinity ",        # ∞
    "\u2211": " sum of ",          # ∑
    "\u220f": " product of ",      # ∏
    "\u221a": " square root of ",  # √
    "\u2261": " is equivalent to ", # ≡
    "\u2203": " there exists ",    # ∃
    "\u2200": " for all ",         # ∀
    "\u2208": " in ",              # ∈
    "\u2209": " not in ",          # ∉
    "\u2282": " subset of ",       # ⊂
    "\u2283": " superset of ",     # ⊃

    # === Currency (convert to words) ===
    "\u00a2": " cents ",     # Cent ¢
    "\u00a3": " pounds ",    # British pound £
    "\u00a5": " yen ",       # Yen ¥
    "\u20ac": " euros ",     # Euro €
    "\u20a3": " francs ",    # French franc ₣
    "\u20b9": " rupees ",    # Indian rupee ₹
    "\u20bf": " bitcoin ",   # Bitcoin ₿

    # === Fractions (convert to words) ===
    "\u00bc": " one quarter ",
    "\u00bd": " one half ",
    "\u00be": " three quarters ",
    "\u2153": " one third ",
    "\u2154": " two thirds ",
    "\u2155": " one fifth ",
    "\u2156": " two fifths ",
    "\u2157": " three fifths ",
    "\u2158": " four fifths ",
    "\u2159": " one sixth ",
    "\u215a": " five sixths ",
    "\u215b": " one eighth ",
    "\u215c": " three eighths ",
    "\u215d": " five eighths ",
    "\u215e": " seven eighths ",

    # === Superscripts and Subscripts ===
    "\u00b2": " squared ",   # ²
    "\u00b3": " cubed ",     # ³
    "\u00b9": "1",           # ¹
    "\u2070": "0",           # ⁰
    "\u2074": "4",           # ⁴
    "\u2075": "5",           # ⁵
    "\u2076": "6",           # ⁶
    "\u2077": "7",           # ⁷
    "\u2078": "8",           # ⁸
    "\u2079": "9",           # ⁹
    "\u207a": "+",           # ⁺
    "\u207b": "-",           # ⁻
    "\u2080": "0",           # ₀
    "\u2081": "1",           # ₁
    "\u2082": "2",           # ₂
    "\u2083": "3",           # ₃
    "\u2084": "4",           # ₄
    "\u2085": "5",           # ₅
    "\u2086": "6",           # ₆
    "\u2087": "7",           # ₇
    "\u2088": "8",           # ₈
    "\u2089": "9",           # ₉

    # === Degrees and Units ===
    "\u00b0": " degrees ",   # Degree °
    "\u2103": " degrees Celsius ",   # ℃
    "\u2109": " degrees Fahrenheit ", # ℉
    "\u00b5": " micro ",     # Micro µ
    "\u2126": " ohms ",      # Ohm Ω

    # === Special Symbols ===
    "\u00a9": " copyright ", # ©
    "\u00ae": " registered ", # ®
    "\u2122": " trademark ", # ™
    "\u00a7": " section ",   # §
    "\u00b6": "",            # Pilcrow ¶ (remove)
    "\u2020": "",            # Dagger † (remove)
    "\u2021": "",            # Double dagger ‡ (remove)
    "\u00a6": "|",           # Broken bar ¦
    "\u00ac": " not ",       # Not sign ¬
    "\u2030": " per mille ", # ‰
    "\u2031": " per ten thousand ", # ‱

    # === Ligatures (expand) ===
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u0153": "oe",          # œ
    "\u0152": "OE",          # Œ
    "\u00e6": "ae",          # æ
    "\u00c6": "AE",          # Æ
    "\u0133": "ij",          # ĳ
    "\u0132": "IJ",          # Ĳ

    # === Common Accented Characters (to ASCII) ===
    # A variants
    "\u00c0": "A", "\u00c1": "A", "\u00c2": "A", "\u00c3": "A", "\u00c4": "A", "\u00c5": "A",
    "\u00e0": "a", "\u00e1": "a", "\u00e2": "a", "\u00e3": "a", "\u00e4": "a", "\u00e5": "a",
    "\u0100": "A", "\u0101": "a", "\u0102": "A", "\u0103": "a", "\u0104": "A", "\u0105": "a",
    # C variants
    "\u00c7": "C", "\u00e7": "c", "\u0106": "C", "\u0107": "c", "\u0108": "C", "\u0109": "c",
    "\u010c": "C", "\u010d": "c",
    # D variants
    "\u010e": "D", "\u010f": "d", "\u0110": "D", "\u0111": "d", "\u00d0": "D", "\u00f0": "d",
    # E variants
    "\u00c8": "E", "\u00c9": "E", "\u00ca": "E", "\u00cb": "E",
    "\u00e8": "e", "\u00e9": "e", "\u00ea": "e", "\u00eb": "e",
    "\u0112": "E", "\u0113": "e", "\u0114": "E", "\u0115": "e", "\u0116": "E", "\u0117": "e",
    "\u0118": "E", "\u0119": "e", "\u011a": "E", "\u011b": "e",
    # G variants
    "\u011c": "G", "\u011d": "g", "\u011e": "G", "\u011f": "g", "\u0120": "G", "\u0121": "g",
    "\u0122": "G", "\u0123": "g",
    # H variants
    "\u0124": "H", "\u0125": "h", "\u0126": "H", "\u0127": "h",
    # I variants
    "\u00cc": "I", "\u00cd": "I", "\u00ce": "I", "\u00cf": "I",
    "\u00ec": "i", "\u00ed": "i", "\u00ee": "i", "\u00ef": "i",
    "\u0128": "I", "\u0129": "i", "\u012a": "I", "\u012b": "i", "\u012c": "I", "\u012d": "i",
    "\u012e": "I", "\u012f": "i", "\u0130": "I", "\u0131": "i",
    # J variants
    "\u0134": "J", "\u0135": "j",
    # K variants
    "\u0136": "K", "\u0137": "k", "\u0138": "k",
    # L variants
    "\u0139": "L", "\u013a": "l", "\u013b": "L", "\u013c": "l", "\u013d": "L", "\u013e": "l",
    "\u013f": "L", "\u0140": "l", "\u0141": "L", "\u0142": "l",
    # N variants
    "\u00d1": "N", "\u00f1": "n", "\u0143": "N", "\u0144": "n", "\u0145": "N", "\u0146": "n",
    "\u0147": "N", "\u0148": "n", "\u0149": "n", "\u014a": "N", "\u014b": "n",
    # O variants
    "\u00d2": "O", "\u00d3": "O", "\u00d4": "O", "\u00d5": "O", "\u00d6": "O", "\u00d8": "O",
    "\u00f2": "o", "\u00f3": "o", "\u00f4": "o", "\u00f5": "o", "\u00f6": "o", "\u00f8": "o",
    "\u014c": "O", "\u014d": "o", "\u014e": "O", "\u014f": "o", "\u0150": "O", "\u0151": "o",
    # R variants
    "\u0154": "R", "\u0155": "r", "\u0156": "R", "\u0157": "r", "\u0158": "R", "\u0159": "r",
    # S variants
    "\u015a": "S", "\u015b": "s", "\u015c": "S", "\u015d": "s", "\u015e": "S", "\u015f": "s",
    "\u0160": "S", "\u0161": "s", "\u00df": "ss",  # German sharp S ß
    # T variants
    "\u0162": "T", "\u0163": "t", "\u0164": "T", "\u0165": "t", "\u0166": "T", "\u0167": "t",
    "\u00de": "Th", "\u00fe": "th",  # Thorn
    # U variants
    "\u00d9": "U", "\u00da": "U", "\u00db": "U", "\u00dc": "U",
    "\u00f9": "u", "\u00fa": "u", "\u00fb": "u", "\u00fc": "u",
    "\u0168": "U", "\u0169": "u", "\u016a": "U", "\u016b": "u", "\u016c": "U", "\u016d": "u",
    "\u016e": "U", "\u016f": "u", "\u0170": "U", "\u0171": "u", "\u0172": "U", "\u0173": "u",
    # W variants
    "\u0174": "W", "\u0175": "w",
    # Y variants
    "\u00dd": "Y", "\u00fd": "y", "\u00ff": "y", "\u0176": "Y", "\u0177": "y", "\u0178": "Y",
    # Z variants
    "\u0179": "Z", "\u017a": "z", "\u017b": "Z", "\u017c": "z", "\u017d": "Z", "\u017e": "z",
}


# ============================================================================
# CLEANING FUNCTIONS
# ============================================================================

def fix_rtf_artifacts(text: str) -> str:
    """
    Fix common RTF conversion artifacts.

    RTF conversion often produces:
    - Lone ? where special chars should be (em dash, copyright, etc.)
    - SPECIALIMAGE placeholders
    - Broken character sequences
    """
    # Remove SPECIALIMAGE placeholders entirely
    text = re.sub(r'SPECIAL_?IMAGE[^\s]*', '', text, flags=re.IGNORECASE)

    # Fix "? Author Name" pattern (should be "by Author Name" or just remove the ?)
    # This happens when em dash before attribution gets converted to ?
    text = re.sub(r'\s+\?\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)', r' - \1', text)

    # Fix lone ? at start of quote attribution (common in epigraphs)
    text = re.sub(r'"\s*\?\s*([A-Z])', r'" - \1', text)

    # Fix "Copyright ?" -> "Copyright"
    text = re.sub(r'Copyright\s*\?', 'Copyright', text, flags=re.IGNORECASE)

    # Fix multiple question marks that aren't intentional
    text = re.sub(r'\?{2,}', '?', text)

    # Fix "word?word" (missing space, ? should be dash or removed)
    text = re.sub(r'(\w)\?(\w)', r'\1 \2', text)

    # Fix " ? " (lone question mark between spaces - likely was a dash)
    text = re.sub(r'\s+\?\s+', ' - ', text)

    # Fix "?s" pattern (possessive that got mangled)
    text = re.sub(r"\?s\b", "'s", text)

    # Fix "?t" pattern (contraction like "don't" -> "don?t")
    text = re.sub(r"\?t\b", "'t", text)

    # Fix "?ll" pattern (contraction like "I'll" -> "I?ll")
    text = re.sub(r"\?ll\b", "'ll", text)

    # Fix "?re" pattern (contraction like "they're" -> "they?re")
    text = re.sub(r"\?re\b", "'re", text)

    # Fix "?ve" pattern (contraction like "I've" -> "I?ve")
    text = re.sub(r"\?ve\b", "'ve", text)

    # Fix "?d" pattern (contraction like "I'd" -> "I?d")
    text = re.sub(r"\?d\b", "'d", text)

    return text


def unicode_to_ascii(text: str) -> str:
    """
    Convert all Unicode characters to ASCII equivalents.
    """
    # First apply explicit mappings
    for unicode_char, ascii_equiv in UNICODE_TO_ASCII.items():
        text = text.replace(unicode_char, ascii_equiv)

    # Remove characters from remove set
    text = "".join(c for c in text if c not in REMOVE_CHARS)

    # Use unicodedata to decompose remaining accented characters
    # NFD decomposition separates base char from combining marks
    text = unicodedata.normalize("NFD", text)
    # Remove combining diacritical marks (category 'Mn')
    text = "".join(c for c in text if unicodedata.category(c) != 'Mn')

    return text


def ensure_ascii(text: str) -> str:
    """
    Final pass to ensure only ASCII characters remain.
    Any remaining non-ASCII is replaced with space or removed.
    """
    result = []
    for char in text:
        if ord(char) < 128:
            result.append(char)
        elif ord(char) in (0x0a, 0x0d):  # Keep newlines
            result.append(char)
        else:
            # Log unexpected character for debugging
            # Replace with space to maintain word boundaries
            result.append(' ')

    return ''.join(result)


def clean_markdown_for_tts(text: str) -> str:
    """Remove markdown formatting that shouldn't be spoken."""
    # Remove markdown headings but keep the text
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)  # ***bold italic***
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)      # **bold**
    text = re.sub(r"\*(.+?)\*", r"\1", text)          # *italic*
    text = re.sub(r"___(.+?)___", r"\1", text)        # ___bold italic___
    text = re.sub(r"__(.+?)__", r"\1", text)          # __bold__
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)  # _italic_ (not mid_word_underscores)

    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)

    # Remove links but keep the text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Remove bare URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove images entirely
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", text)

    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove blockquote markers but keep text
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)

    # Remove list markers but keep text
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)

    return text


def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations for better TTS pronunciation."""
    abbreviations = [
        # Titles (must have space after to avoid false matches)
        (r"\bDr\.\s", "Doctor "),
        (r"\bMr\.\s", "Mister "),
        (r"\bMrs\.\s", "Missus "),
        (r"\bMs\.\s", "Miss "),
        (r"\bProf\.\s", "Professor "),
        (r"\bSt\.\s", "Saint "),
        (r"\bGen\.\s", "General "),
        (r"\bCol\.\s", "Colonel "),
        (r"\bLt\.\s", "Lieutenant "),
        (r"\bSgt\.\s", "Sergeant "),
        (r"\bCpt\.\s", "Captain "),
        (r"\bRev\.\s", "Reverend "),

        # Common abbreviations
        (r"\bvs\.", "versus"),
        (r"\betc\.", "et cetera"),
        (r"\bi\.e\.", "that is"),
        (r"\be\.g\.", "for example"),
        (r"\bcf\.", "compare"),
        (r"\bno\.\s", "number "),
        (r"\bNo\.\s", "Number "),
        (r"\bvol\.", "volume"),
        (r"\bVol\.", "Volume"),
        (r"\bpp\.", "pages"),
        (r"\bp\.\s", "page "),
        (r"\bfig\.", "figure"),
        (r"\bFig\.", "Figure"),
        (r"\bch\.", "chapter"),
        (r"\bCh\.", "Chapter"),
        (r"\bapprox\.", "approximately"),
        (r"\bca\.", "circa"),

        # Units
        (r"\bkm\b", "kilometers"),
        (r"\bcm\b", "centimeters"),
        (r"\bmm\b", "millimeters"),
        (r"\bkg\b", "kilograms"),
        (r"\bmg\b", "milligrams"),
        (r"\bml\b", "milliliters"),
        (r"\bHz\b", "hertz"),
        (r"\bkHz\b", "kilohertz"),
        (r"\bMHz\b", "megahertz"),
        (r"\bGHz\b", "gigahertz"),
        (r"\bGB\b", "gigabytes"),
        (r"\bMB\b", "megabytes"),
        (r"\bKB\b", "kilobytes"),
        (r"\bTB\b", "terabytes"),

        # Time
        (r"\ba\.m\.", "A M"),
        (r"\bp\.m\.", "P M"),
        (r"\bAM\b", "A M"),
        (r"\bPM\b", "P M"),

        # Misc
        (r"\bAI\b", "A I"),
        (r"\bUI\b", "U I"),
        (r"\bAPI\b", "A P I"),
        (r"\bURL\b", "U R L"),
        (r"\bUSA\b", "U S A"),
        (r"\bUK\b", "U K"),
        (r"\bUN\b", "U N"),
        (r"\bEU\b", "E U"),
        (r"\bCEO\b", "C E O"),
        (r"\bCTO\b", "C T O"),
        (r"\bDNA\b", "D N A"),
        (r"\bRNA\b", "R N A"),
    ]

    for pattern, replacement in abbreviations:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def expand_numbers_and_symbols(text: str) -> str:
    """Expand numeric expressions and symbols for better TTS."""
    # Ordinals
    ordinal_map = {
        "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
        "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
        "9th": "ninth", "10th": "tenth", "11th": "eleventh", "12th": "twelfth",
        "13th": "thirteenth", "14th": "fourteenth", "15th": "fifteenth",
        "16th": "sixteenth", "17th": "seventeenth", "18th": "eighteenth",
        "19th": "nineteenth", "20th": "twentieth", "21st": "twenty-first",
        "22nd": "twenty-second", "23rd": "twenty-third", "30th": "thirtieth",
        "31st": "thirty-first", "40th": "fortieth", "50th": "fiftieth",
        "100th": "hundredth",
    }

    for abbr, full in ordinal_map.items():
        text = re.sub(rf"\b{abbr}\b", full, text, flags=re.IGNORECASE)

    # Expand & to "and"
    text = re.sub(r'\s*&\s*', ' and ', text)

    # Expand @ in context
    text = re.sub(r'\s*@\s*', ' at ', text)

    # Expand common symbols
    text = text.replace('#', ' number ')
    text = text.replace('%', ' percent')
    text = text.replace('+', ' plus ')
    text = text.replace('=', ' equals ')

    # Clean up multiple spaces from expansions
    text = re.sub(r' {2,}', ' ', text)

    return text


def clean_whitespace(text: str) -> str:
    """Normalize whitespace for natural TTS reading."""
    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Normalize multiple spaces to single space
    text = re.sub(r" {2,}", " ", text)

    # Normalize multiple newlines to double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing whitespace from lines
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Remove leading/trailing whitespace
    return text.strip()


def clean_text_for_tts(text: str) -> str:
    """
    Apply all cleaning operations to prepare text for TTS.

    Order matters:
    1. Fix RTF artifacts (before Unicode conversion)
    2. Convert Unicode to ASCII
    3. Clean markdown
    4. Expand abbreviations
    5. Expand numbers and symbols
    6. Final ASCII check
    7. Clean whitespace

    Args:
        text: Raw text content

    Returns:
        Cleaned ASCII text suitable for TTS
    """
    if not text:
        return ""

    # 1. Fix RTF conversion artifacts
    text = fix_rtf_artifacts(text)

    # 2. Convert Unicode to ASCII
    text = unicode_to_ascii(text)

    # 3. Clean markdown formatting
    text = clean_markdown_for_tts(text)

    # 4. Expand abbreviations
    text = expand_abbreviations(text)

    # 5. Expand numbers and symbols
    text = expand_numbers_and_symbols(text)

    # 6. Final ASCII enforcement
    text = ensure_ascii(text)

    # 7. Clean whitespace
    text = clean_whitespace(text)

    return text


# ============================================================================
# LANGGRAPH NODE
# ============================================================================

def clean_text(state: AudiobookState) -> AudiobookState:
    """
    LangGraph node: Clean chapter text for TTS.

    Args:
        state: Current workflow state with chapters

    Returns:
        Updated state with cleaned_content in each chapter
    """
    state.stage = WorkflowStage.CLEANING

    if not state.chapters:
        state.errors.append("No chapters to clean")
        state.stage = WorkflowStage.FAILED
        return state

    try:
        print(f"[DEBUG CLEAN] Cleaning {len(state.chapters)} chapters")
        for chapter in state.chapters:
            original_len = len(chapter.content) if chapter.content else 0
            print(f"[DEBUG CLEAN] Chapter {chapter.number}: content length before clean = {original_len}")

            chapter.cleaned_content = clean_text_for_tts(chapter.content)

            cleaned_len = len(chapter.cleaned_content) if chapter.cleaned_content else 0
            print(f"[DEBUG CLEAN] Chapter {chapter.number}: cleaned_content length = {cleaned_len}")

            # Verify ASCII
            non_ascii = [c for c in chapter.cleaned_content if ord(c) >= 128]
            if non_ascii:
                print(f"[DEBUG CLEAN] WARNING: Chapter {chapter.number} has {len(non_ascii)} non-ASCII chars remaining")
                # Show first few for debugging
                print(f"[DEBUG CLEAN]   First non-ASCII: {[hex(ord(c)) for c in non_ascii[:5]]}")

        state.stage = WorkflowStage.CHUNKING

    except Exception as e:
        import traceback
        print(f"[DEBUG CLEAN] Error: {e}")
        print(traceback.format_exc())
        state.errors.append(f"Text cleaning error: {str(e)}")
        state.stage = WorkflowStage.FAILED

    return state


def get_cleaning_stats(state: AudiobookState) -> dict:
    """Get statistics about text cleaning."""
    if not state.chapters:
        return {"error": "No chapters"}

    stats = []
    for ch in state.chapters:
        if ch.cleaned_content:
            original_len = len(ch.content)
            cleaned_len = len(ch.cleaned_content)
            stats.append({
                "chapter": ch.number,
                "original_chars": original_len,
                "cleaned_chars": cleaned_len,
                "reduction_pct": round((1 - cleaned_len / original_len) * 100, 1) if original_len > 0 else 0,
            })

    return {
        "chapters_cleaned": len([s for s in stats if s]),
        "chapter_stats": stats,
    }
