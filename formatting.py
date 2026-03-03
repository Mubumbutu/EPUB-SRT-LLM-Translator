# formatting.py
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ALL_QUOTES_CHARS = '"\'\u2018\u2019\u201C\u201D\u201E\u201F\u00AB\u00BB\u2039\u203A\u201A\u201B\u2032\u2033\u301D\u301E\u301F\uFF02'
DOUBLE_QUOTES_CHARS = '"\u201C\u201D\u201E\u201F\u00AB\u00BB\u301D\u301E\u301F\uFF02'
SINGLE_QUOTES_CHARS = '\'\u2018\u2019\u201A\u201B\u2032\u2039\u203A'

ALL_QUOTES = r'["\'\u2018\u2019\u201C\u201D\u201E\u201F\u00AB\u00BB\u2039\u203A\u201A\u201B\u2032\u2033\u301D\u301E\u301F\uFF02]'
DOUBLE_QUOTES = r'["\u201C\u201D\u201E\u201F\u00AB\u00BB\u301D\u301E\u301F\uFF02]'
SINGLE_QUOTES = r'[\'\u2018\u2019\u201A\u201B\u2032\u2039\u203A]'

def count_quotes(text: str) -> int:
    return len(re.findall(ALL_QUOTES, text))

def count_double_quotes(text: str) -> int:
    return len(re.findall(DOUBLE_QUOTES, text))

def count_all_quotes(text: str) -> int:
    return sum(1 for ch in text if ch in DOUBLE_QUOTES_CHARS)

def has_quote_at_start(text: str) -> bool:
    text = text.lstrip()
    if not text:
        return False

    text_no_placeholders = re.sub(r'^(</?p_\d{2}>|<id_\d{2}>|<nt_\d{2}/>|\s)+', '', text)

    if not text_no_placeholders:
        return False

    if re.match(ALL_QUOTES, text_no_placeholders):
        return True

    if text_no_placeholders.startswith('- ') or text_no_placeholders.startswith('– ') or text_no_placeholders.startswith('— '):
        return True

    return False

def has_quote_at_end(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False

    text_no_placeholders = re.sub(r'(\s|</?p_\d{2}>|<id_\d{2}>|<nt_\d{2}/>)+$', '', text).rstrip()

    if not text_no_placeholders:
        return False

    return bool(re.match(ALL_QUOTES, text_no_placeholders[-1]))

def contains_dash(text: str) -> bool:
    return '-' in text or '–' in text or '—' in text

def count_external_quotes(text: str) -> int:
    count = 0

    text_start = text.lstrip()
    text_start_clean = re.sub(r'^(</?p_\d{2}>|<id_\d{2}>|<nt_\d{2}/>|\s)+', '', text_start)

    if text_start_clean and re.match(ALL_QUOTES, text_start_clean):
        count += 1

    text_end = text.rstrip()
    text_end_clean = re.sub(r'(\s|</?p_\d{2}>|<id_\d{2}>|<nt_\d{2}/>)+$', '', text_end).rstrip()

    if text_end_clean and re.match(ALL_QUOTES, text_end_clean[-1]):
        count += 1

    return count

def add_quote_to_start(text: str) -> str:
    stripped = text.lstrip()
    leading_ws = text[:len(text) - len(stripped)]

    match = re.match(r'^(</?p_\d{2}>|<id_\d{2}>|<nt_\d{2}/>|\s)*', stripped)
    if match:
        placeholders = match.group(0)
        content = stripped[len(placeholders):]
        return leading_ws + placeholders + '\u201e' + content
    else:
        return leading_ws + '\u201e' + stripped

def add_quote_to_end(text: str) -> str:
    stripped = text.rstrip()
    trailing_ws = text[len(stripped):]

    match = re.search(r'(</?p_\d{2}>|<id_\d{2}>|<nt_\d{2}/>|\s)*$', stripped)
    if match:
        placeholders = match.group(0)
        content = stripped[:len(stripped) - len(placeholders)]
        return content + '\u201d' + placeholders + trailing_ws
    else:
        return stripped + '\u201d' + trailing_ws

def remove_quote_from_start(text: str, force: bool = False) -> str:
    stripped = text.lstrip()
    leading_ws = text[:len(text) - len(stripped)]

    if not stripped:
        return text

    match = re.match(r'^(</?p_\d{2}>|<id_\d{2}>|<nt_\d{2}/>|\s)*', stripped)
    if match:
        placeholders = match.group(0)
        content = stripped[len(placeholders):]

        if not content:
            return text

        if re.match(ALL_QUOTES, content[0]):
            if not force:
                original_count = count_quotes(text)
                if original_count % 2 == 0:
                    return text
            return leading_ws + placeholders + content[1:]

    return text

def remove_quote_from_end(text: str, force: bool = False) -> str:
    stripped = text.rstrip()
    trailing_ws = text[len(stripped):]

    if not stripped:
        return text

    match = re.search(r'(</?p_\d{2}>|<id_\d{2}>|<nt_\d{2}/>|\s)*$', stripped)
    if match:
        placeholders = match.group(0)
        content = stripped[:len(stripped) - len(placeholders)]

        if not content:
            return text

        if re.match(ALL_QUOTES, content[-1]):
            if not force:
                original_count = count_quotes(text)
                if original_count % 2 == 0:
                    return text
            return content[:-1] + placeholders + trailing_ws

    return text

def get_first_letter_info(text: str) -> Tuple[Optional[int], Optional[bool]]:
    text_work = text.lstrip()

    if text_work.startswith('- '):
        text_work = text_work[2:].lstrip()

    text_work = re.sub(f'^{ALL_QUOTES}+', '', text_work).lstrip()

    if not text_work:
        return None, None

    in_tag = False
    for i, char in enumerate(text_work):
        if char == '<':
            in_tag = True
        elif char == '>':
            in_tag = False
        elif char.isalpha() and not in_tag:
            return i, char.isupper()

    return None, None

def get_ending_punctuation(text: str) -> Tuple[Optional[str], Optional[int]]:
    text_stripped = text.rstrip()
    if not text_stripped:
        return None, None

    text_no_quotes = re.sub(f'{ALL_QUOTES}+$', '', text_stripped).rstrip()

    if not text_no_quotes:
        return None, None

    last_char = text_no_quotes[-1]

    if last_char == '.':
        return 'period', len(text_no_quotes) - 1
    elif last_char == ',':
        return 'comma', len(text_no_quotes) - 1
    elif last_char == '!':
        return 'exclamation', len(text_no_quotes) - 1
    elif last_char == '?':
        return 'question', len(text_no_quotes) - 1
    elif last_char == '…':
        return 'ellipsis', len(text_no_quotes) - 1

    return None, None

def add_punctuation(text: str, punct_type: str) -> str:
    text_stripped = text.rstrip()
    trailing_ws = text[len(text_stripped):]

    text_no_quotes = re.sub(f'{ALL_QUOTES}+$', '', text_stripped)
    trailing_quotes = text_stripped[len(text_no_quotes):]

    text_core = text_no_quotes.rstrip()
    quotes_ws = text_no_quotes[len(text_core):]

    punct = '.' if punct_type == 'period' else ','
    return text_core + punct + quotes_ws + trailing_quotes + trailing_ws

def remove_punctuation(text: str, punct_type: str) -> str:
    text_stripped = text.rstrip()
    trailing_ws = text[len(text_stripped):]

    text_no_quotes = re.sub(f'{ALL_QUOTES}+$', '', text_stripped)
    trailing_quotes = text_stripped[len(text_no_quotes):]

    text_core = text_no_quotes.rstrip()

    if not text_core:
        return text

    punct = '.' if punct_type == 'period' else ','
    if text_core.endswith(punct):
        text_core = text_core[:-1]

    return text_core + trailing_quotes + trailing_ws

def replace_punctuation(text: str, from_punct: str, to_punct: str) -> str:
    text_stripped = text.rstrip()
    trailing_ws = text[len(text_stripped):]

    text_no_quotes = re.sub(f'{ALL_QUOTES}+$', '', text_stripped)
    trailing_quotes = text_stripped[len(text_no_quotes):]

    text_core = text_no_quotes.rstrip()

    if not text_core:
        return text

    from_char = '.' if from_punct == 'period' else ','
    to_char = '.' if to_punct == 'period' else ','

    if text_core.endswith(from_char):
        text_core = text_core[:-1] + to_char

    return text_core + trailing_quotes + trailing_ws

class MismatchChecker:
    DEFAULT_CHECKS = {
        "paragraphs": True,
        "first_char": True,
        "last_char": True,
        "length": True,
        "quote_parity": True,
        "untranslated": True,
        "reserve_elements": True,
        "inline_formatting": True,
        "nt_markers": True,
        "ps_markers": True,
    }
    DEFAULT_THRESHOLDS = {
        "length_ratio_short": 1.6,
        "length_ratio_medium": 1.4,
        "length_ratio_long": 1.3,
        "untranslated_ratio": 0.3,
        "position_shift_threshold": 0.15,
    }

    def __init__(self, file_type: str, processing_mode: str = "inline", mismatch_settings: Dict = None):
        self.file_type = file_type
        self.processing_mode = processing_mode

        if mismatch_settings:
            checks = self.DEFAULT_CHECKS.copy()
            checks.update(mismatch_settings.get("mismatch_checks", {}))
            self.checks = checks

            thresholds = self.DEFAULT_THRESHOLDS.copy()
            thresholds.update(mismatch_settings.get("mismatch_thresholds", {}))
            self.thresholds = thresholds
        else:
            self.checks = self.DEFAULT_CHECKS.copy()
            self.thresholds = self.DEFAULT_THRESHOLDS.copy()

        logger.debug(
            f"MismatchChecker init: file_type={file_type}, mode={processing_mode}, "
            f"checks={self.checks}, thresholds={self.thresholds}"
        )

    def check_mismatch(self, para: Dict) -> Tuple[bool, Dict]:
        processing_mode = para.get('processing_mode', self.processing_mode)
        if processing_mode == 'legacy':
            return self._check_mismatch_legacy(para)
        else:
            return self._check_mismatch_inline(para)

    def _dynamic_length_mismatch(self, a: str, b: str) -> bool:
        if not a or not b:
            return False
        a_len, b_len = len(a), len(b)
        shorter = min(a_len, b_len)
        longer = max(a_len, b_len)
        if shorter == 0:
            return True

        if a_len <= 20 and b_len <= 20:
            return False
        ratio = longer / shorter
        if a_len <= 100 and b_len <= 100:
            threshold = self.thresholds.get("length_ratio_short", 1.6)
        elif a_len <= 500 and b_len <= 500:
            threshold = self.thresholds.get("length_ratio_medium", 1.4)
        else:
            threshold = self.thresholds.get("length_ratio_long", 1.3)
        return ratio > threshold

    def _check_untranslated_with_threshold(self, orig: str, trans: str) -> bool:
        if orig.strip() != trans.strip():
            return False
        text = orig.strip()
        if not text:
            return False
        url_pattern = re.compile(
            r'^(https?://\S+|www\.\S+|\S+@\S+\.\S+|[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(/\S*)?)$',
            re.IGNORECASE
        )
        if url_pattern.match(text):
            return False
        if ' ' not in text and '\n' not in text:
            return False
        words = re.findall(r'[^\W\d_]+', text, re.UNICODE)
        if not words:
            return False
        if len(words) <= 2:
            return False
        lowercase_words = [w for w in words if w[0].islower()]
        ratio = len(lowercase_words) / len(words)
        threshold = self.thresholds.get("untranslated_ratio", 0.3)
        return ratio > threshold

    def _check_mismatch_inline(self, para: Dict) -> Tuple[bool, Dict]:
        if para.get("ignore_mismatch", False):
            return False, {}
        if para.get("force_mismatch", False):
            return True, {"manual": True}
        if not para.get("is_translated"):
            return False, {}

        orig_raw = para.get("original_text", "")
        trans_raw = para.get("translated_text", "")

        if 'auto_wrap_tags' in para and trans_raw:
            trans_raw = self._extract_text_without_auto_wrap(trans_raw, para['auto_wrap_tags'])

        reserve_ok, reserve_errors = self._check_reserve_elements_integrity(orig_raw, trans_raw)
        nt_ok, nt_errors           = self._check_nt_markers_integrity(orig_raw, trans_raw)
        inline_ok, inline_errors   = self._check_inline_formatting_integrity(orig_raw, trans_raw)
        ps_ok, ps_errors           = self._check_ps_markers_integrity(orig_raw, trans_raw)

        orig  = self._remove_all_placeholders(orig_raw)
        trans = self._remove_all_placeholders(trans_raw)

        if not trans.strip() or orig.strip() == trans.strip():
            mismatch_flags = {}
            if not reserve_ok and self.checks.get("reserve_elements", True):
                mismatch_flags["reserve_elements"] = reserve_errors
            if not nt_ok and self.checks.get("nt_markers", True):
                mismatch_flags["nt_markers"] = nt_errors
            if not inline_ok and self.checks.get("inline_formatting", True):
                mismatch_flags["inline_formatting"] = inline_errors
            if not ps_ok and self.checks.get("ps_markers", True):
                mismatch_flags["ps_markers"] = ps_errors
            if orig.strip() == trans.strip():
                if self.checks.get("untranslated", True) and self._check_untranslated_with_threshold(orig, trans):
                    mismatch_flags["untranslated"] = True
            return len(mismatch_flags) > 0, mismatch_flags

        def count_paragraphs(text):
            parts = [p for p in text.split("\n\n") if p.strip()]
            return len(parts) if len(parts) > 1 else len([p for p in text.split("\n") if p.strip()])

        def first_char_type(text):
            text = text.lstrip()
            if not text: return "none"
            c = text[0]
            if c == '<': return "placeholder"
            elif c in ALL_QUOTES_CHARS: return "quote"
            elif c.isdigit(): return "digit"
            elif c.isalpha(): return "letter"
            else: return "other"

        def last_char_type(text):
            text = text.rstrip()
            if not text: return {"type": "empty"}
            t = text
            for ch in reversed(t):
                if ch not in ALL_QUOTES_CHARS: break
                t = t[:-1]
            t = t.rstrip()
            if not t: return {"type": "quote_only"}
            lc = t[-1]
            if lc == '.' and t.endswith('...'): return {"type": "ellipsis"}
            mapping = {'.': 'period', ',': 'comma', '!': 'exclamation', '?': 'question', '…': 'ellipsis'}
            if lc in mapping: return {"type": mapping[lc]}
            if lc in ';:': return {"type": "semicolon_colon"}
            elif lc.isalpha(): return {"type": "letter"}
            elif lc.isdigit(): return {"type": "digit"}
            else: return {"type": "other"}

        def check_quote_parity(orig_text, trans_text):
            orig_count  = count_all_quotes(orig_text)
            trans_count = count_all_quotes(trans_text)
            if orig_count % 2 != 0 and orig_count == trans_count: return False
            return trans_count % 2 != 0

        orig_para_count  = count_paragraphs(orig)
        trans_para_count = count_paragraphs(trans)

        mismatch_flags = {}

        if self.checks.get("paragraphs", True) and orig_para_count != trans_para_count:
            mismatch_flags["paragraphs"] = {"orig": orig_para_count, "trans": trans_para_count}

        if self.checks.get("first_char", True) and first_char_type(orig) != first_char_type(trans):
            mismatch_flags["first_char"] = True

        if self.checks.get("last_char", True) and last_char_type(orig)["type"] != last_char_type(trans)["type"]:
            mismatch_flags["last_char"] = True

        if self.checks.get("length", True):
            length_flag = self._dynamic_length_mismatch(orig, trans)
            if length_flag:
                orig_words  = len(orig.split())
                trans_words = len(trans.split())
                if 0.5 <= trans_words / max(orig_words, 1) <= 3.0:
                    length_flag = False
            if length_flag:
                mismatch_flags["length"] = True

        if self.checks.get("quote_parity", True) and check_quote_parity(orig, trans):
            mismatch_flags["quote_parity"] = True

        if self.checks.get("untranslated", True) and self._check_untranslated_with_threshold(orig, trans):
            mismatch_flags["untranslated"] = True

        if not reserve_ok and self.checks.get("reserve_elements", True):
            mismatch_flags["reserve_elements"] = reserve_errors

        if not nt_ok and self.checks.get("nt_markers", True):
            mismatch_flags["nt_markers"] = nt_errors

        if not inline_ok and self.checks.get("inline_formatting", True):
            mismatch_flags["inline_formatting"] = inline_errors

        if not ps_ok and self.checks.get("ps_markers", True):
            mismatch_flags["ps_markers"] = ps_errors

        has_mismatch = any(mismatch_flags.values())
        return has_mismatch, mismatch_flags

    def _check_mismatch_legacy(self, para: Dict) -> Tuple[bool, Dict]:
        if para.get("ignore_mismatch", False):
            return False, {}
        if para.get("force_mismatch", False):
            return True, {"manual": True}
        if not para.get("is_translated"):
            return False, {}

        orig_raw = para.get("original_text", "")
        trans_raw = para.get("translated_text", "")

        reserve_ok, reserve_errors = self._check_reserve_elements_legacy(orig_raw, trans_raw)

        orig = re.sub(r'<id_\d{2}>', '', orig_raw)
        trans = re.sub(r'<id_\d{2}>', '', trans_raw)

        if not trans.strip() or orig.strip() == trans.strip():
            mismatch_flags = {}
            if not reserve_ok and self.checks.get("reserve_elements", True):
                mismatch_flags["reserve_elements"] = reserve_errors
            if orig.strip() == trans.strip():
                if self.checks.get("untranslated", True) and self._check_untranslated_with_threshold(orig, trans):
                    mismatch_flags["untranslated"] = True
            return len(mismatch_flags) > 0, mismatch_flags

        def count_paragraphs(text):
            parts = [p for p in text.split("\n\n") if p.strip()]
            return len(parts) if len(parts) > 1 else len([p for p in text.split("\n") if p.strip()])

        def first_char_type(text):
            text = text.lstrip()
            if not text: return "none"
            c = text[0]
            if c in ALL_QUOTES_CHARS: return "quote"
            elif c.isdigit(): return "digit"
            elif c.isalpha(): return "letter"
            else: return "other"

        def last_char_type(text):
            text = text.rstrip()
            if not text: return {"type": "empty"}
            t = text
            for ch in reversed(t):
                if ch not in ALL_QUOTES_CHARS: break
                t = t[:-1]
            t = t.rstrip()
            if not t: return {"type": "quote_only"}
            lc = t[-1]
            if lc == '.' and t.endswith('...'): return {"type": "ellipsis"}
            mapping = {'.': 'period', ',': 'comma', '!': 'exclamation', '?': 'question', '…': 'ellipsis'}
            if lc in mapping: return {"type": mapping[lc]}
            if lc in ';:': return {"type": "semicolon_colon"}
            elif lc.isalpha(): return {"type": "letter"}
            elif lc.isdigit(): return {"type": "digit"}
            else: return {"type": "other"}

        def check_quote_parity(orig_text, trans_text):
            orig_count = count_all_quotes(orig_text)
            trans_count = count_all_quotes(trans_text)
            if orig_count % 2 != 0 and orig_count == trans_count: return False
            return trans_count % 2 != 0

        orig_para_count = count_paragraphs(orig)
        trans_para_count = count_paragraphs(trans)

        mismatch_flags = {}

        if self.checks.get("paragraphs", True) and orig_para_count != trans_para_count:
            mismatch_flags["paragraphs"] = {"orig": orig_para_count, "trans": trans_para_count}

        if self.checks.get("first_char", True) and first_char_type(orig) != first_char_type(trans):
            mismatch_flags["first_char"] = True

        if self.checks.get("last_char", True) and last_char_type(orig)["type"] != last_char_type(trans)["type"]:
            mismatch_flags["last_char"] = True

        if self.checks.get("length", True):
            length_flag = self._dynamic_length_mismatch(orig, trans)
            if length_flag:
                orig_words = len(orig.split())
                trans_words = len(trans.split())
                if 0.5 <= trans_words / max(orig_words, 1) <= 3.0:
                    length_flag = False
            if length_flag:
                mismatch_flags["length"] = True

        if self.checks.get("quote_parity", True) and check_quote_parity(orig, trans):
            mismatch_flags["quote_parity"] = True

        if self.checks.get("untranslated", True) and self._check_untranslated_with_threshold(orig, trans):
            mismatch_flags["untranslated"] = True

        if not reserve_ok and self.checks.get("reserve_elements", True):
            mismatch_flags["reserve_elements"] = reserve_errors

        has_mismatch = any(mismatch_flags.values())
        return has_mismatch, mismatch_flags

    def _check_untranslated(self, orig: str, trans: str) -> bool:
        return self._check_untranslated_with_threshold(orig, trans)

    def _check_reserve_elements_integrity(self, original: str, translated: str) -> Tuple[bool, Optional[Dict]]:
        orig_reserves  = re.findall(r'<id_(\d{2})>', original)
        trans_reserves = re.findall(r'<id_(\d{2})>', translated)

        error_details: Dict = {}

        if sorted(orig_reserves) != sorted(trans_reserves):
            missing = [f"<id_{t}>" for t in orig_reserves
                       if orig_reserves.count(t) > trans_reserves.count(t)]
            extra   = [f"<id_{t}>" for t in trans_reserves
                       if trans_reserves.count(t) > orig_reserves.count(t)]
            if missing:
                error_details['missing'] = list(set(missing))
            if extra:
                error_details['extra'] = list(set(extra))

        spurious_closing = re.findall(r'</id_(\d{2})>', translated)
        if spurious_closing:
            error_details['spurious_closing'] = list(
                set(f"</id_{t}>" for t in spurious_closing)
            )

        threshold = self.thresholds.get("position_shift_threshold", 0.15)

        orig_positions  = self._get_self_closing_tag_positions(original,    r'<id_(\d{2})>')
        trans_positions = self._get_self_closing_tag_positions(translated,  r'<id_(\d{2})>')

        common_ids = set(orig_positions.keys()) & set(trans_positions.keys())
        positioning_errors = []

        for tag_id in sorted(common_ids):
            orig_pos  = orig_positions[tag_id]
            trans_pos = trans_positions[tag_id]

            if abs(orig_pos - trans_pos) > threshold:
                positioning_errors.append({
                    'tag_id':       tag_id,
                    'issue':        'position_shift',
                    'orig_rel_pos':  round(orig_pos,  2),
                    'trans_rel_pos': round(trans_pos, 2),
                    'description':  (
                        f"<id_{tag_id}> is at ~{orig_pos:.0%} in original "
                        f"but at ~{trans_pos:.0%} in translation"
                    ),
                })

        if positioning_errors:
            error_details['positioning'] = positioning_errors

        if error_details:
            return False, error_details
        return True, None

    def _check_reserve_elements_legacy(self, original: str, translated: str) -> Tuple[bool, Optional[Dict]]:
        return self._check_reserve_elements_integrity(original, translated)

    def _check_inline_formatting_integrity(self, original: str, translated: str) -> Tuple[bool, Optional[Dict]]:
        orig_opens  = re.findall(r'<p_(\d{2})>', original)
        orig_closes = re.findall(r'</p_(\d{2})>', original)
        trans_opens  = re.findall(r'<p_(\d{2})>', translated)
        trans_closes = re.findall(r'</p_(\d{2})>', translated)

        error_details = {}

        if sorted(orig_opens) != sorted(trans_opens):
            missing_opens = [f"<p_{t}>" for t in orig_opens if orig_opens.count(t) > trans_opens.count(t)]
            extra_opens   = [f"<p_{t}>" for t in trans_opens if trans_opens.count(t) > orig_opens.count(t)]
            if missing_opens or extra_opens:
                error_details['opening_tags'] = {}
                if missing_opens: error_details['opening_tags']['missing'] = list(set(missing_opens))
                if extra_opens:   error_details['opening_tags']['extra']   = list(set(extra_opens))

        if sorted(orig_closes) != sorted(trans_closes):
            missing_closes = [f"</p_{t}>" for t in orig_closes if orig_closes.count(t) > trans_closes.count(t)]
            extra_closes   = [f"</p_{t}>" for t in trans_closes if trans_closes.count(t) > orig_closes.count(t)]
            if missing_closes or extra_closes:
                error_details['closing_tags'] = {}
                if missing_closes: error_details['closing_tags']['missing'] = list(set(missing_closes))
                if extra_closes:   error_details['closing_tags']['extra']   = list(set(extra_closes))

        unpaired = []
        for tag_id in set(trans_opens + trans_closes):
            open_count  = trans_opens.count(tag_id)
            close_count = trans_closes.count(tag_id)
            if open_count != close_count:
                unpaired.append({'tag_id': tag_id, 'open_count': open_count, 'close_count': close_count})
        if unpaired:
            error_details['unpaired_tags'] = unpaired

        position_errors = self._check_inline_formatting_position(original, translated)
        if position_errors:
            error_details['positioning'] = position_errors

        is_ok = len(error_details) == 0
        return is_ok, error_details if not is_ok else None

    def _check_inline_formatting_position(self, original: str, translated: str) -> List[Dict]:
        errors = []
        orig_structure  = self._parse_inline_structure(original)
        trans_structure = self._parse_inline_structure(translated)
        all_tag_ids = set(orig_structure.keys()) | set(trans_structure.keys())

        inline_threshold = self.thresholds.get("inline_position_shift_threshold", 0.30)

        for tag_id in all_tag_ids:
            orig_info  = orig_structure.get(tag_id)
            trans_info = trans_structure.get(tag_id)
            if not orig_info or not trans_info:
                continue

            orig_coverage  = orig_info.get('coverage', 0)
            trans_coverage = trans_info.get('coverage', 0)
            if abs(orig_coverage - trans_coverage) > 0.3:
                errors.append({
                    'tag_id':        tag_id,
                    'issue':         'coverage_mismatch',
                    'orig_coverage':  round(orig_coverage,  2),
                    'trans_coverage': round(trans_coverage, 2),
                    'orig_content':   orig_info.get('content', ''),
                    'trans_content':  trans_info.get('content', ''),
                    'description': (
                        f"Tag <p_{tag_id}> covers {orig_coverage:.0%} of text in original "
                        f"but {trans_coverage:.0%} in translation"
                    ),
                })

            orig_rel_start  = orig_info.get('relative_start', 0)
            trans_rel_start = trans_info.get('relative_start', 0)
            if abs(orig_rel_start - trans_rel_start) > inline_threshold:
                errors.append({
                    'tag_id':          tag_id,
                    'issue':           'position_shift',
                    'orig_rel_start':   round(orig_rel_start,  2),
                    'trans_rel_start':  round(trans_rel_start, 2),
                    'orig_content':     orig_info.get('content', ''),
                    'trans_content':    trans_info.get('content', ''),
                    'description': (
                        f"Tag <p_{tag_id}> starts at {orig_rel_start:.0%} in original "
                        f"but at {trans_rel_start:.0%} in translation"
                    ),
                })

            orig_parent  = orig_info.get('parent')
            trans_parent = trans_info.get('parent')
            if orig_parent != trans_parent:
                errors.append({
                    'tag_id':       tag_id,
                    'issue':        'nesting_mismatch',
                    'orig_parent':   orig_parent,
                    'trans_parent':  trans_parent,
                    'orig_content':  orig_info.get('content', ''),
                    'trans_content': trans_info.get('content', ''),
                    'description': (
                        f"Tag <p_{tag_id}> has different nesting: "
                        f"parent {orig_parent} in original, {trans_parent} in translation"
                    ),
                })

        return errors

    def _parse_inline_structure(self, text: str) -> Dict:
        clean_text = re.sub(r'</?p_\d{2}>|<id_\d{2}>|<nt_\d{2}/>|<ps>', '', text)
        total_length = len(clean_text)
        if total_length == 0:
            return {}

        structure = {}
        stack = []
        current_pos = 0
        content_accumulator: Dict[str, str] = {}

        pattern = r'(<p_(\d{2})>|</p_(\d{2})>|<id_\d{2}>|<nt_\d{2}/>|<ps>|[^<]+|<[^>]*>)'

        for match in re.finditer(pattern, text):
            token = match.group(0)

            if token.startswith('<p_') and not token.startswith('</p_'):
                tag_id = re.search(r'<p_(\d{2})>', token).group(1)
                stack.append((tag_id, current_pos))
                content_accumulator[tag_id] = ''

            elif token.startswith('</p_'):
                tag_id = re.search(r'</p_(\d{2})>', token).group(1)
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i][0] == tag_id:
                        opening_tag_id, start_pos = stack.pop(i)
                        end_pos = current_pos
                        length = end_pos - start_pos
                        coverage = length / total_length if total_length > 0 else 0
                        relative_start = start_pos / total_length if total_length > 0 else 0
                        parent = stack[-1][0] if stack else None
                        tag_content = content_accumulator.pop(tag_id, '').strip()
                        if tag_id not in structure:
                            structure[tag_id] = {
                                'start': start_pos, 'end': end_pos,
                                'length': length, 'coverage': coverage,
                                'relative_start': relative_start,
                                'content': tag_content,
                                'parent': parent, 'children': []
                            }
                        if parent and parent in structure:
                            if tag_id not in structure[parent]['children']:
                                structure[parent]['children'].append(tag_id)
                        break

            elif (token.startswith('<id_')
                  or token.startswith('<nt_')
                  or token == '<ps>'):
                pass

            else:
                current_pos += len(token)
                for open_tag_id, _ in stack:
                    if open_tag_id in content_accumulator:
                        content_accumulator[open_tag_id] += token

        return structure

    def _get_self_closing_tag_positions(self, text: str, tag_regex: str) -> Dict[str, float]:
        ALL_PLACEHOLDER = re.compile(
            r'</?p_\d{2}>|<id_\d{2}>|<nt_\d{2}/>|<ps>'
        )
        TARGET_TAG = re.compile(tag_regex)

        clean_text = ALL_PLACEHOLDER.sub('', text)
        total_len = len(clean_text)
        if total_len == 0:
            return {}

        positions: Dict[str, float] = {}
        clean_pos = 0

        token_re = re.compile(
            r'</?p_\d{2}>|<id_\d{2}>|<nt_\d{2}/>|<ps>|[^<]+|<[^>]*>'
        )

        for m in token_re.finditer(text):
            token = m.group(0)

            target_match = TARGET_TAG.match(token)
            if target_match:
                tag_id = target_match.group(1)
                if tag_id not in positions:
                    positions[tag_id] = clean_pos / total_len
            elif ALL_PLACEHOLDER.match(token):
                pass
            else:
                clean_pos += len(token)

        return positions

    def _extract_text_without_auto_wrap(self, text: str, auto_wrap_tags: List[Dict]) -> str:
        working_text = text.strip()
        for tag_info in auto_wrap_tags:
            opening = tag_info['opening']
            closing = tag_info['closing']
            if working_text.startswith(opening) and working_text.endswith(closing):
                working_text = working_text[len(opening):-len(closing)].strip()
            else:
                break
        return working_text

    def _remove_all_placeholders(self, text: str) -> str:
        text = re.sub(r'<id_\d{2}>', '', text)
        text = re.sub(r'</id_\d{2}>', '', text)
        text = re.sub(r'</?p_\d{2}>', '', text)
        text = re.sub(r'<nt_\d{2}/>', '', text)
        text = re.sub(r'<ps>', '', text)
        return text

    def _check_nt_markers_integrity(self, original: str, translated: str) -> Tuple[bool, Optional[Dict]]:
        orig_nt  = re.findall(r'<nt_(\d{2})/>', original)
        trans_nt = re.findall(r'<nt_(\d{2})/>', translated)

        error_details: Dict = {}

        if sorted(orig_nt) != sorted(trans_nt):
            missing = [f"<nt_{t}/>" for t in orig_nt
                       if orig_nt.count(t) > trans_nt.count(t)]
            extra   = [f"<nt_{t}/>" for t in trans_nt
                       if trans_nt.count(t) > orig_nt.count(t)]
            if missing:
                error_details['missing'] = list(set(missing))
            if extra:
                error_details['extra'] = list(set(extra))

        threshold = self.thresholds.get("position_shift_threshold", 0.15)

        orig_positions  = self._get_self_closing_tag_positions(original,   r'<nt_(\d{2})/>')
        trans_positions = self._get_self_closing_tag_positions(translated, r'<nt_(\d{2})/>')

        common_ids = set(orig_positions.keys()) & set(trans_positions.keys())
        positioning_errors = []

        for tag_id in sorted(common_ids):
            orig_pos  = orig_positions[tag_id]
            trans_pos = trans_positions[tag_id]

            if abs(orig_pos - trans_pos) > threshold:
                positioning_errors.append({
                    'tag_id':        tag_id,
                    'issue':         'position_shift',
                    'orig_rel_pos':   round(orig_pos,  2),
                    'trans_rel_pos':  round(trans_pos, 2),
                    'description':   (
                        f"<nt_{tag_id}/> is at ~{orig_pos:.0%} in original "
                        f"but at ~{trans_pos:.0%} in translation"
                    ),
                })

        if positioning_errors:
            error_details['positioning'] = positioning_errors

        if error_details:
            return False, error_details
        return True, None

    def _check_ps_markers_integrity(self, original: str, translated: str) -> Tuple[bool, Optional[Dict]]:
        orig_ps = re.findall(r'<ps>', original)
        trans_ps = re.findall(r'<ps>', translated)

        if not orig_ps and not trans_ps:
            return True, None

        if not orig_ps and trans_ps:
            return False, {
                'expected': 0,
                'found': len(trans_ps),
                'missing': 0,
                'extra': len(trans_ps),
            }

        if len(orig_ps) == len(trans_ps):
            return True, None

        error_details = {
            'expected': len(orig_ps),
            'found': len(trans_ps),
            'missing': max(0, len(orig_ps) - len(trans_ps)),
            'extra': max(0, len(trans_ps) - len(orig_ps)),
        }
        return False, error_details

class FormattingSynchronizer:
    def __init__(self, file_type: str):
        self.file_type = file_type

    def sync_formatting(
        self,
        original: str,
        translated: str,
        para: Optional[Dict] = None
    ) -> str:
        if original == translated:
            return translated

        if not translated.strip():
            return translated

        if para and para.get('processing_mode') == 'legacy':
            return self._sync_formatting_legacy(original, translated)
        else:
            return self._sync_formatting_new(original, translated, para)

    def _sync_formatting_legacy(self, original: str, translated: str) -> str:
        if original == translated:
            return translated

        if not translated.strip():
            return translated

        result = translated

        result = self.remove_malformed_xml_tags(result)

        result = self.normalize_quotes(result)

        if self.file_type == "srt":
            result = self.sync_formatting_legacy_srt(original, result)
        else:
            result = self.sync_formatting_legacy_epub(original, result)

        return result

    def _sync_formatting_new(self, original: str, translated: str, para: Optional[Dict] = None) -> str:
        result = self.remove_malformed_xml_tags(translated)

        result = self.normalize_quotes(result)

        if self.file_type == "srt":
            result = self.sync_formatting_srt(original, result)
        else:
            result = self.sync_formatting_epub(original, result)

        return result

    def remove_malformed_xml_tags(self, text: str) -> str:
        text_stripped = text.strip()
        leading_ws  = text[:len(text) - len(text.lstrip())]
        trailing_ws = text[len(text.rstrip()):]

        has_translation_open  = re.match(r'^<translation>', text_stripped, re.IGNORECASE)
        has_translation_close = re.search(r'</translation>$', text_stripped, re.IGNORECASE)

        if has_translation_open and has_translation_close:
            text_without_tags = re.sub(
                r'^<translation>\s*|\s*</translation>$',
                '',
                text_stripped,
                flags=re.IGNORECASE
            ).strip()
            return leading_ws + text_without_tags + trailing_ws

        if '</text_to_translate>' in text_stripped:
            text_stripped = text_stripped.replace('</text_to_translate>', '')
            logger.debug("Removed rogue </text_to_translate> tag")

        if re.search(r'</id_\d{2}>', text_stripped):
            text_stripped = re.sub(r'</id_\d{2}>', '', text_stripped).strip()
            logger.debug("Removed spurious </id_XX> closing tag(s)")

        OPENING_TAGS = [
            '<translated>',
            '<TRANSLATED>',
            '<translation>',
            '<TRANSLATION>',
        ]

        CLOSING_TAGS = [
            '</translated>',
            '</TRANSLATED>',
            '</translation>',
            '</TRANSLATION>',
        ]

        has_opening_tag    = False
        opening_tag_length = 0

        for tag in OPENING_TAGS:
            if text_stripped.startswith(tag):
                has_opening_tag    = True
                opening_tag_length = len(tag)
                break

        has_closing_tag    = False
        closing_tag_length = 0

        for tag in CLOSING_TAGS:
            if text_stripped.endswith(tag):
                has_closing_tag    = True
                closing_tag_length = len(tag)
                break

        if has_opening_tag and has_closing_tag:
            text_without_tags = text_stripped[opening_tag_length:-closing_tag_length].strip()
            return leading_ws + text_without_tags + trailing_ws

        if has_opening_tag and not has_closing_tag:
            text_without_opening = text_stripped[opening_tag_length:].strip()
            return leading_ws + text_without_opening + trailing_ws

        if not has_opening_tag and has_closing_tag:
            text_without_closing = text_stripped[:-closing_tag_length].strip()
            return leading_ws + text_without_closing + trailing_ws

        return leading_ws + text_stripped + trailing_ws

    def normalize_quotes(self, text: str) -> str:
        DOUBLE_QUOTES_VARIANTS = [
            '\u201C',
            '\u201D',
            '\u201E',
            '\u201F',
            '\u00AB',
            '\u00BB',
            '\u301D',
            '\u301E',
            '\u301F',
            '\uFF02',
        ]

        SINGLE_QUOTES_VARIANTS = [
            '\u2018',
            '\u2019',
            '\u201A',
            '\u201B',
            '\u2032',
            '\u2039',
            '\u203A',
        ]

        result = text

        for variant in DOUBLE_QUOTES_VARIANTS:
            result = result.replace(variant, '"')

        for variant in SINGLE_QUOTES_VARIANTS:
            result = result.replace(variant, "'")

        return result

    def _sync_all_caps_legacy(self, original: str, translated: str) -> str:
        orig_letters_only = re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', '', original)

        if not orig_letters_only:
            return translated

        is_all_caps = orig_letters_only.isupper()

        if not is_all_caps:
            return translated

        result = []
        for char in translated:
            if char.isalpha():
                result.append(char.upper())
            else:
                result.append(char)

        return ''.join(result)

    def sync_formatting_srt(self, original: str, translated: str) -> str:
        result = translated

        result = self.sync_quotes(original, result)

        result = self.sync_dialogue_dashes_srt(original, result)

        result = self.sync_leading_whitespace(original, result)

        result = self.sync_all_caps(original, result)

        orig_letters_only = re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', '', original)
        is_all_caps = orig_letters_only.isupper() if orig_letters_only else False

        if not is_all_caps:
            result = self.sync_first_letter_case(original, result)

        result = self.sync_ending_punctuation(original, result)

        return result

    def sync_formatting_epub(self, original: str, translated: str) -> str:
        result = translated

        result = self.sync_quotes(original, result)

        result = self.sync_leading_whitespace_epub(original, result)

        result = self.sync_all_caps(original, result)

        orig_letters_only = re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', '', original)
        is_all_caps = orig_letters_only.isupper() if orig_letters_only else False

        if not is_all_caps:
            result = self.sync_first_letter_case_epub(original, result)

        result = self.sync_ending_punctuation_epub(original, result)

        result = self._remove_temporary_markers(result)

        return result

    def sync_dialogue_dashes_srt(self, original: str, translated: str) -> str:
        if '\n' not in original:
            return translated

        orig_lines = original.split('\n')
        trans_lines = translated.split('\n')

        orig_has_dash = []
        for line in orig_lines:
            line_stripped = line.lstrip()
            has_dash = line_stripped.startswith('- ')
            orig_has_dash.append(has_dash)

        dash_count = sum(orig_has_dash)

        if dash_count < 2:
            return translated

        if len(orig_lines) != len(trans_lines):
            if dash_count == len(orig_lines):
                result_lines = []
                for trans_line in trans_lines:
                    trans_stripped = trans_line.lstrip()
                    leading_ws = trans_line[:len(trans_line) - len(trans_stripped)]

                    if trans_stripped.startswith('- '):
                        result_lines.append(trans_line)
                    else:
                        result_lines.append(leading_ws + '- ' + trans_stripped)

                return '\n'.join(result_lines)
            else:
                return translated

        result_lines = []
        for i, (orig_line, trans_line) in enumerate(zip(orig_lines, trans_lines)):
            trans_stripped = trans_line.lstrip()
            leading_ws = trans_line[:len(trans_line) - len(trans_stripped)]

            should_have_dash = orig_has_dash[i]
            has_dash = trans_stripped.startswith('- ')

            if should_have_dash and not has_dash:
                result_lines.append(leading_ws + '- ' + trans_stripped)
            elif not should_have_dash and has_dash:
                trans_no_dash = trans_stripped[2:]
                result_lines.append(leading_ws + trans_no_dash)
            else:
                result_lines.append(trans_line)

        return '\n'.join(result_lines)

    def sync_leading_whitespace_epub(self, original: str, translated: str) -> str:
        orig_leading = re.match(r'^([ \t]*)', original)
        orig_whitespace = orig_leading.group(1) if orig_leading else ''
        trans_stripped = translated.lstrip(' \t')

        return orig_whitespace + trans_stripped

    def sync_first_letter_case_epub(self, original: str, translated: str) -> str:
        def get_first_letter_info_local(text):
            text_work = text.lstrip()
            text_work = re.sub(f'^{ALL_QUOTES}+', '', text_work).lstrip()

            if not text_work:
                return None, None

            in_tag = False
            for i, char in enumerate(text_work):
                if char == '<':
                    in_tag = True
                elif char == '>':
                    in_tag = False
                elif char.isalpha() and not in_tag:
                    return i, char.isupper()

            return None, None

        orig_pos, orig_is_upper = get_first_letter_info_local(original)

        if orig_pos is None:
            return translated

        trans_stripped = translated.lstrip()
        leading_ws = translated[:len(translated) - len(trans_stripped)]

        trans_no_quotes = re.sub(f'^{ALL_QUOTES}+', '', trans_stripped)
        quote_prefix = trans_stripped[:len(trans_stripped) - len(trans_no_quotes)]

        trans_no_quotes_stripped = trans_no_quotes.lstrip()
        quote_ws = trans_no_quotes[:len(trans_no_quotes) - len(trans_no_quotes_stripped)]

        if not trans_no_quotes_stripped:
            return translated

        result_chars = list(trans_no_quotes_stripped)
        in_tag = False
        for i, char in enumerate(result_chars):
            if char == '<':
                in_tag = True
            elif char == '>':
                in_tag = False
            elif char.isalpha() and not in_tag:
                if orig_is_upper:
                    result_chars[i] = char.upper()
                else:
                    result_chars[i] = char.lower()
                break

        return leading_ws + quote_prefix + quote_ws + ''.join(result_chars)

    def sync_ending_punctuation_epub(self, original: str, translated: str) -> str:
        def get_ending_punctuation_local(text):
            text_stripped = text.rstrip()
            if not text_stripped:
                return None, False, False

            has_double = bool(re.search(f'{DOUBLE_QUOTES}$', text_stripped))

            if has_double:
                text_no_double = re.sub(f'{DOUBLE_QUOTES}+$', '', text_stripped).rstrip()
            else:
                text_no_double = text_stripped

            if not text_no_double:
                return None, has_double, False

            has_apostrophe = bool(re.search(f'{SINGLE_QUOTES}$', text_no_double))

            if has_apostrophe:
                text_no_quotes = re.sub(f'{SINGLE_QUOTES}+$', '', text_no_double).rstrip()
            else:
                text_no_quotes = text_no_double

            if not text_no_quotes:
                return None, has_double, has_apostrophe

            last_char = text_no_quotes[-1]

            punct_type = None
            if last_char == '.':
                punct_type = 'period'
            elif last_char == ',':
                punct_type = 'comma'
            elif last_char == '!':
                punct_type = 'exclamation'
            elif last_char == '?':
                punct_type = 'question'
            elif last_char == '…':
                punct_type = 'ellipsis'

            return punct_type, has_double, has_apostrophe

        def add_punctuation_local(text, punct_type):
            text_stripped = text.rstrip()
            trailing_ws = text[len(text_stripped):]

            has_double = bool(re.search(f'{DOUBLE_QUOTES}$', text_stripped))
            if has_double:
                text_no_double = re.sub(f'{DOUBLE_QUOTES}+$', '', text_stripped)
                trailing_double = text_stripped[len(text_no_double):]
            else:
                text_no_double = text_stripped
                trailing_double = ''

            has_apostrophe = bool(re.search(f'{SINGLE_QUOTES}$', text_no_double))
            if has_apostrophe:
                text_no_quotes = re.sub(f'{SINGLE_QUOTES}+$', '', text_no_double)
                trailing_apostrophe = text_no_double[len(text_no_quotes):]
                text_core = text_no_quotes.rstrip()
                quotes_ws = text_no_quotes[len(text_core):]
            else:
                text_core = text_no_double.rstrip()
                quotes_ws = text_no_double[len(text_core):]
                trailing_apostrophe = ''

            punct = '.' if punct_type == 'period' else ','
            return text_core + punct + quotes_ws + trailing_apostrophe + trailing_double + trailing_ws

        def remove_punctuation_local(text, punct_type):
            text_stripped = text.rstrip()
            trailing_ws = text[len(text_stripped):]

            has_double = bool(re.search(f'{DOUBLE_QUOTES}$', text_stripped))
            if has_double:
                text_no_double = re.sub(f'{DOUBLE_QUOTES}+$', '', text_stripped)
                trailing_double = text_stripped[len(text_no_double):]
            else:
                text_no_double = text_stripped
                trailing_double = ''

            has_apostrophe = bool(re.search(f'{SINGLE_QUOTES}$', text_no_double))
            if has_apostrophe:
                text_no_quotes = re.sub(f'{SINGLE_QUOTES}+$', '', text_no_double)
                trailing_apostrophe = text_no_double[len(text_no_quotes):]
                text_core = text_no_quotes.rstrip()
            else:
                text_core = text_no_double
                trailing_apostrophe = ''

            if not text_core:
                return text

            punct = '.' if punct_type == 'period' else ','
            if text_core.endswith(punct):
                text_core = text_core[:-1]

            return text_core + trailing_apostrophe + trailing_double + trailing_ws

        def replace_punctuation_local(text, from_punct, to_punct):
            text_stripped = text.rstrip()
            trailing_ws = text[len(text_stripped):]

            has_double = bool(re.search(f'{DOUBLE_QUOTES}$', text_stripped))
            if has_double:
                text_no_double = re.sub(f'{DOUBLE_QUOTES}+$', '', text_stripped)
                trailing_double = text_stripped[len(text_no_double):]
            else:
                text_no_double = text_stripped
                trailing_double = ''

            has_apostrophe = bool(re.search(f'{SINGLE_QUOTES}$', text_no_double))
            if has_apostrophe:
                text_no_quotes = re.sub(f'{SINGLE_QUOTES}+$', '', text_no_double)
                trailing_apostrophe = text_no_double[len(text_no_quotes):]
                text_core = text_no_quotes.rstrip()
            else:
                text_core = text_no_double
                trailing_apostrophe = ''

            if not text_core:
                return text

            from_char = '.' if from_punct == 'period' else ','
            to_char = '.' if to_punct == 'period' else ','

            if text_core.endswith(from_char):
                text_core = text_core[:-1] + to_char

            return text_core + trailing_apostrophe + trailing_double + trailing_ws

        orig_punct, orig_has_double, orig_has_apos = get_ending_punctuation_local(original)
        trans_punct, trans_has_double, trans_has_apos = get_ending_punctuation_local(translated)

        if orig_punct == 'period':
            if trans_punct == 'period':
                return translated
            elif trans_punct == 'comma':
                return replace_punctuation_local(translated, 'comma', 'period')
            elif trans_punct in ['exclamation', 'question', 'ellipsis']:
                return translated
            else:
                return add_punctuation_local(translated, 'period')

        elif orig_punct == 'comma':
            if trans_punct == 'comma':
                return translated
            elif trans_punct == 'period':
                return replace_punctuation_local(translated, 'period', 'comma')
            elif trans_punct in ['exclamation', 'question', 'ellipsis']:
                return translated
            else:
                return add_punctuation_local(translated, 'comma')

        elif orig_punct in ['exclamation', 'question', 'ellipsis']:
            return translated

        else:
            if trans_punct == 'period':
                return remove_punctuation_local(translated, 'period')
            elif trans_punct == 'comma':
                return remove_punctuation_local(translated, 'comma')
            elif trans_punct in ['exclamation', 'question', 'ellipsis']:
                return translated
            else:
                return translated

    def sync_all_caps(self, original: str, translated: str) -> str:
        orig_letters_only = re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', '', original)

        if not orig_letters_only:
            return translated

        is_all_caps = orig_letters_only.isupper()

        if not is_all_caps:
            return translated

        result = []
        for char in translated:
            if char.isalpha():
                result.append(char.upper())
            else:
                result.append(char)

        return ''.join(result)

    def sync_leading_whitespace(self, original: str, translated: str) -> str:
        if '\n' in original or '\n' in translated:
            orig_lines = original.split('\n')
            trans_lines = translated.split('\n')

            if len(orig_lines) != len(trans_lines):
                orig_leading = re.match(r'^([ \t]*)', original)
                orig_whitespace = orig_leading.group(1) if orig_leading else ''
                trans_stripped = translated.lstrip(' \t')
                return orig_whitespace + trans_stripped

            result_lines = []
            for orig_line, trans_line in zip(orig_lines, trans_lines):
                orig_leading = re.match(r'^([ \t]*)', orig_line)
                orig_whitespace = orig_leading.group(1) if orig_leading else ''
                trans_stripped = trans_line.lstrip(' \t')
                result_lines.append(orig_whitespace + trans_stripped)

            return '\n'.join(result_lines)

        orig_leading = re.match(r'^([ \t]*)', original)
        orig_whitespace = orig_leading.group(1) if orig_leading else ''
        trans_stripped = translated.lstrip(' \t')

        return orig_whitespace + trans_stripped

    def sync_first_letter_case(self, original: str, translated: str) -> str:
        def sync_single_line(orig_line, trans_line):
            orig_pos, orig_is_upper = get_first_letter_info(orig_line)

            if orig_pos is None:
                return trans_line

            trans_stripped = trans_line.lstrip()
            leading_ws = trans_line[:len(trans_line) - len(trans_stripped)]

            dash_prefix = ''
            if trans_stripped.startswith('- '):
                dash_prefix = '- '
                trans_stripped = trans_stripped[2:].lstrip()

            trans_no_quotes = re.sub(f'^{ALL_QUOTES}+', '', trans_stripped)
            quote_prefix = trans_stripped[:len(trans_stripped) - len(trans_no_quotes)]

            trans_no_quotes_stripped = trans_no_quotes.lstrip()
            quote_ws = trans_no_quotes[:len(trans_no_quotes) - len(trans_no_quotes_stripped)]

            if not trans_no_quotes_stripped:
                return trans_line

            result_chars = list(trans_no_quotes_stripped)
            in_tag = False
            for i, char in enumerate(result_chars):
                if char == '<':
                    in_tag = True
                elif char == '>':
                    in_tag = False
                elif char.isalpha() and not in_tag:
                    if orig_is_upper:
                        result_chars[i] = char.upper()
                    else:
                        result_chars[i] = char.lower()
                    break

            return leading_ws + dash_prefix + quote_prefix + quote_ws + ''.join(result_chars)

        if '\n' in original or '\n' in translated:
            orig_lines = original.split('\n')
            trans_lines = translated.split('\n')

            if len(orig_lines) != len(trans_lines):
                return sync_single_line(original, translated)

            result_lines = []
            for orig_line, trans_line in zip(orig_lines, trans_lines):
                result_lines.append(sync_single_line(orig_line, trans_line))

            return '\n'.join(result_lines)

        return sync_single_line(original, translated)

    def sync_ending_punctuation(self, original: str, translated: str) -> str:
        def sync_single_line_ending(orig_line, trans_line):
            orig_punct, _ = get_ending_punctuation(orig_line)
            trans_punct, _ = get_ending_punctuation(trans_line)

            if orig_punct == 'period':
                if trans_punct == 'period':
                    return trans_line
                elif trans_punct == 'comma':
                    return replace_punctuation(trans_line, 'comma', 'period')
                elif trans_punct in ['exclamation', 'question', 'ellipsis']:
                    return trans_line
                else:
                    return add_punctuation(trans_line, 'period')

            elif orig_punct == 'comma':
                if trans_punct == 'comma':
                    return trans_line
                elif trans_punct == 'period':
                    return replace_punctuation(trans_line, 'period', 'comma')
                elif trans_punct in ['exclamation', 'question', 'ellipsis']:
                    return trans_line
                else:
                    return add_punctuation(trans_line, 'comma')

            elif orig_punct in ['exclamation', 'question', 'ellipsis']:
                return trans_line

            else:
                if trans_punct == 'period':
                    return remove_punctuation(trans_line, 'period')
                elif trans_punct == 'comma':
                    return remove_punctuation(trans_line, 'comma')
                elif trans_punct in ['exclamation', 'question', 'ellipsis']:
                    return trans_line
                else:
                    return trans_line

        if '\n' in original or '\n' in translated:
            orig_lines = original.split('\n')
            trans_lines = translated.split('\n')

            if len(orig_lines) != len(trans_lines):
                return sync_single_line_ending(original, translated)

            result_lines = []
            for i, (orig_line, trans_line) in enumerate(zip(orig_lines, trans_lines)):
                if i == len(orig_lines) - 1:
                    result_lines.append(sync_single_line_ending(orig_line, trans_line))
                else:
                    result_lines.append(trans_line)

            return '\n'.join(result_lines)

        return sync_single_line_ending(original, translated)

    def sync_quotes(self, original: str, translated: str) -> str:
        def process_single_line(orig_line, trans_line):
            orig_has_start = has_quote_at_start(orig_line)
            orig_has_end = has_quote_at_end(orig_line)

            trans_stripped = trans_line.lstrip()
            trans_has_quote_start = bool(trans_stripped and re.match(ALL_QUOTES, trans_stripped))
            trans_has_dash_start = bool(trans_stripped and (
                trans_stripped.startswith('- ') or
                trans_stripped.startswith('– ') or
                trans_stripped.startswith('— ')
            ))
            trans_has_start = trans_has_quote_start or trans_has_dash_start

            trans_has_end = has_quote_at_end(trans_line)

            result = trans_line

            current_quote_count = count_double_quotes(trans_line)
            is_even = (current_quote_count % 2 == 0)

            need_start = orig_has_start and not trans_has_start
            need_end = orig_has_end and not trans_has_end

            if need_start and need_end:
                result = add_quote_to_start(result)
                result = add_quote_to_end(result)
                trans_has_start = True
                trans_has_end = True
            elif need_start:
                if orig_has_start and trans_has_dash_start and not trans_has_quote_start:
                    pass
                elif not is_even:
                    result = add_quote_to_start(result)
                    trans_has_start = True
            elif need_end:
                if not is_even:
                    result = add_quote_to_end(result)
                    trans_has_end = True
            else:
                pass

            if (not orig_has_start and trans_has_quote_start) and (not orig_has_end and trans_has_end):
                result = remove_quote_from_start(result, force=True)
                result = remove_quote_from_end(result, force=True)
            else:
                if not orig_has_start and trans_has_quote_start:
                    result = remove_quote_from_start(result)
                if not orig_has_end and trans_has_end:
                    result = remove_quote_from_end(result)

            if self.file_type in ["epub", "txt"]:
                orig_has_dash = contains_dash(orig_line)
                trans_has_dash = contains_dash(result)

                if not orig_has_dash and trans_has_dash:
                    total_double_quote_count = count_double_quotes(result)
                    external_quote_count = count_external_quotes(result)

                    if total_double_quote_count % 2 != 0 and external_quote_count == 1:
                        result_stripped = result.lstrip()
                        if result_stripped and re.match(DOUBLE_QUOTES, result_stripped):
                            if (total_double_quote_count - 1) % 2 == 0:
                                result = remove_quote_from_start(result, force=True)
                        else:
                            result_stripped_end = result.rstrip()
                            if result_stripped_end and re.match(DOUBLE_QUOTES, result_stripped_end[-1]):
                                if (total_double_quote_count - 1) % 2 == 0:
                                    result = remove_quote_from_end(result, force=True)

            return result

        if self.file_type == "srt":
            if '\n' in original or '\n' in translated:
                orig_lines = original.split('\n')
                trans_lines = translated.split('\n')

                if len(orig_lines) != len(trans_lines):
                    return process_single_line(original, translated)

                result_lines = []
                for orig_line, trans_line in zip(orig_lines, trans_lines):
                    result_lines.append(process_single_line(orig_line, trans_line))

                return '\n'.join(result_lines)
            else:
                return process_single_line(original, translated)

        else:
            return process_single_line(original, translated)

    def sync_formatting_legacy_srt(self, original: str, translated: str) -> str:
        result = translated

        result = self.sync_quotes(original, result)

        result = self.sync_dialogue_dashes_srt(original, result)

        result = self.sync_leading_whitespace(original, result)

        result = self._sync_all_caps_legacy(original, result)

        orig_letters_only = re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', '', original)
        is_all_caps = orig_letters_only.isupper() if orig_letters_only else False

        if not is_all_caps:
            result = self.sync_first_letter_case(original, result)

        result = self.sync_ending_punctuation(original, result)

        return result

    def sync_formatting_legacy_epub(self, original: str, translated: str) -> str:
        result = translated

        result = self.sync_quotes(original, result)

        result = self.sync_leading_whitespace_epub(original, result)

        result = self._sync_all_caps_legacy(original, result)

        orig_letters_only = re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', '', original)
        is_all_caps = orig_letters_only.isupper() if orig_letters_only else False

        if not is_all_caps:
            result = self.sync_first_letter_case_epub(original, result)

        result = self.sync_ending_punctuation_epub(original, result)

        result = self._remove_temporary_markers(result)

        return result

    def _remove_temporary_markers(self, text: str) -> str:
        nt_markers = re.findall(r'<nt_\d{2}/>', text)
        if nt_markers:
            logger.warning(f"Found {len(nt_markers)} unreplaced NT markers - removing them")
            for marker in nt_markers:
                logger.warning(f"  Unreplaced: {marker}")

        cleaned = re.sub(r'<nt_\d{2}/>', '', text)

        ps_markers = re.findall(r'</?ps>', cleaned)
        if ps_markers:
            logger.warning(
                f"Found {len(ps_markers)} unreplaced PS markers - removing them"
            )
        cleaned = re.sub(r'</?ps>', '', cleaned)

        cleaned = re.sub(r'  +', ' ', cleaned)

        return cleaned
