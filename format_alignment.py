# format_alignment.py
from __future__ import annotations
import gc
import logging
import os
import re
import torch
import torch.nn.functional as F
from collections import Counter
from lxml import etree
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

INLINE_TRANSFER_TAGS = {
    'i', 'em', 'b', 'strong', 'u', 'span', 'small', 'mark',
    'cite', 'q', 's', 'del', 'ins'
}
RESERVE_TAGS = {
    'img', 'code', 'br', 'hr', 'sub', 'sup', 'kbd',
    'abbr', 'wbr', 'var', 'canvas', 'svg', 'script',
    'style', 'math', 'a'
}
XHTML_NS = 'http://www.w3.org/1999/xhtml'
MODELS_SUBDIR = 'models'

_RE_APOSTROPHES  = re.compile(r"['\u2019\u2018\u02bc\u0060\u00b4]")
_RE_PUNCT_STRIP  = re.compile(r'^[.,;:!?"\(\)\[\]\-]+|[.,;:!?"\(\)\[\]\-]+$')
_RE_SENTENCE_END = re.compile(r'[.!?]["\'\)\]]*$')
_RE_ABBREV       = re.compile(r'^[A-Za-z]{1,3}\.$|^\d+\.$')
_RE_ELLIPSIS     = re.compile(r'\.\.\.$')
_RE_CONTENT_WORD = re.compile(r'\w', re.UNICODE)
_HYPHEN_JOINERS  = frozenset({'-', '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015'})

def _sanitize_model_name(model_name: str) -> str:
    return model_name.replace('/', '_').replace('\\', '_')

def get_local_model_path(model_name: str, models_dir: str) -> str:
    safe_name = _sanitize_model_name(model_name)
    return os.path.join(models_dir, safe_name)

def is_model_downloaded(model_name: str, models_dir: str) -> bool:
    local_path = get_local_model_path(model_name, models_dir)
    config_path = os.path.join(local_path, 'config.json')
    return os.path.isdir(local_path) and os.path.isfile(config_path)

def download_model(
    model_name: str = 'xlm-roberta-large',
    models_dir: str = '',
) -> str:
    try:
        pass
    except ImportError as exc:
        raise ImportError(
            "Brak pakietu 'transformers'. "
            "Zainstaluj: pip install transformers torch"
        ) from exc

    if not models_dir:
        models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), MODELS_SUBDIR
        )

    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"[FormatAlignment] Folder modeli: {models_dir}")

    local_model_path = get_local_model_path(model_name, models_dir)
    os.makedirs(local_model_path, exist_ok=True)

    logger.info(f"[FormatAlignment] Pobieranie modelu '{model_name}' → {local_model_path}")

    logger.info("[FormatAlignment] Pobieranie tokenizera...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_model_path)
    logger.info("[FormatAlignment] Tokenizer zapisany.")

    logger.info("[FormatAlignment] Pobieranie modelu (może potrwać kilka minut)...")
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        ignore_mismatched_sizes=True,
    )
    model.save_pretrained(local_model_path)
    logger.info(f"[FormatAlignment] Model zapisany w: {local_model_path}")

    del model
    del tokenizer
    gc.collect()

    return local_model_path

def _parse_element_html(html_str: str) -> Optional[etree._Element]:
    try:
        return etree.fromstring(html_str.encode('utf-8'))
    except etree.XMLSyntaxError:
        pass

    try:
        wrapped = f'<__root__ xmlns="{XHTML_NS}">{html_str}</__root__>'
        root = etree.fromstring(wrapped.encode('utf-8'))
        if len(root):
            return root[0]
    except etree.XMLSyntaxError:
        pass

    try:
        fixed = re.sub(
            r'^(<\w[^>]*?)>',
            lambda m: m.group(1) + f' xmlns="{XHTML_NS}">',
            html_str.strip(),
            count=1
        )
        return etree.fromstring(fixed.encode('utf-8'))
    except etree.XMLSyntaxError:
        pass

    return None

def _element_to_html(element: etree._Element) -> str:
    return etree.tostring(element, encoding='unicode', method='xml')

def _get_plain_text(element: etree._Element) -> str:
    text = etree.tostring(element, encoding='unicode', method='text')
    return re.sub(r'\s+', ' ', text).strip()

def _tokenize_words(text: str) -> List[str]:
    return [w for w in text.split() if w]

def _build_attrs_str(attrs: Dict[str, str]) -> str:
    if not attrs:
        return ''
    parts = []
    for k, v in attrs.items():
        if '{' in k:
            continue
        escaped_v = v.replace('"', '&quot;')
        parts.append(f' {k}="{escaped_v}"')
    return ''.join(parts)

def _escape_xml(text: str) -> str:
    return (
        text
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
    )

class FormatAlignmentEngine:
    _ALIGN_LAYER_FALLBACK = 8

    def __init__(
        self,
        model_name: str = 'xlm-roberta-large',
        device: str = 'cpu',
        local_models_dir: str = '',
    ) -> None:
        self.model_name = model_name
        self.device = device

        if local_models_dir:
            self.local_models_dir = local_models_dir
        else:
            self.local_models_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), MODELS_SUBDIR
            )

        self.tokenizer = None
        self.model = None
        self._loaded = False
        self._last_src_len: int = 1
        self._align_layers: List[int] = []
        self._sp_model: bool = False
        self._full_src_embs = None
        self._full_tgt_embs = None

    def load_model(self) -> None:
        try:
            pass
        except ImportError as exc:
            raise ImportError(
                "Brak pakietu 'transformers'. "
                "Zainstaluj: pip install transformers torch"
            ) from exc

        local_path = get_local_model_path(self.model_name, self.local_models_dir)

        if is_model_downloaded(self.model_name, self.local_models_dir):
            load_source = local_path
            logger.info(f"[FormatAlignment] Ładowanie modelu z lokalnego folderu: {local_path}")
        else:
            load_source = self.model_name
            logger.info(
                f"[FormatAlignment] Lokalny model nie znaleziony w: {local_path}\n"
                f"[FormatAlignment] Ładowanie z HuggingFace Hub: '{self.model_name}'"
            )

        logger.info(f"[FormatAlignment] Urządzenie: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(load_source)
            self.model = AutoModel.from_pretrained(
                load_source,
                output_hidden_states=True,
                ignore_mismatched_sizes=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Nie można załadować modelu '{self.model_name}' z '{load_source}': {exc}"
            ) from exc

        self.model.eval()
        self.model.to(torch.device(self.device))
        self._loaded = True

        self._detect_tokenizer_type()
        self._detect_align_layers()

        logger.info(
            f"[FormatAlignment] Model '{self.model_name}' załadowany pomyślnie "
            f"na '{self.device}'. "
            f"Warstwy alignmentu: {self._align_layers}. "
            f"SentencePiece: {self._sp_model}."
        )

    def _detect_tokenizer_type(self) -> None:
        test_tokens = self.tokenizer.tokenize("hello world")
        self._sp_model = any(t.startswith('\u2581') for t in test_tokens)
        logger.debug(
            f"[FormatAlignment] Tokeny testowe: {test_tokens} | "
            f"SentencePiece={self._sp_model}"
        )

    def _detect_align_layers(self) -> None:
        n_layers = None

        try:
            cfg = self.model.config
            if hasattr(cfg, 'encoder_layers'):
                n_layers = cfg.encoder_layers
            elif hasattr(cfg, 'num_hidden_layers'):
                n_layers = cfg.num_hidden_layers
        except Exception as exc:
            logger.warning(f"[FormatAlignment] Nie można odczytać liczby warstw z config: {exc}")

        if n_layers and n_layers > 0:
            last = n_layers
            first = max(1, last - 3)
            self._align_layers = list(range(first, last + 1))
        else:
            logger.warning(
                f"[FormatAlignment] Nieznana liczba warstw – "
                f"używam fallback warstwy {self._ALIGN_LAYER_FALLBACK}"
            )
            self._align_layers = [self._ALIGN_LAYER_FALLBACK]

        logger.info(
            f"[FormatAlignment] Wykryto {n_layers} warstw. "
            f"Warstwy alignmentu: {self._align_layers}"
        )

    def unload_model(self) -> None:
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._align_layers = []

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        gc.collect()
        logger.info("[FormatAlignment] Model zwolniony z pamięci.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _get_layer_output(self, outputs) -> 'torch.Tensor':
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            available = len(outputs.hidden_states)
            layers = [l for l in self._align_layers if l < available]
            if not layers:
                layers = [available - 1]
            stacked = torch.stack(
                [outputs.hidden_states[l][0] for l in layers], dim=0
            )
            return stacked.mean(dim=0)

        if (hasattr(outputs, 'encoder_hidden_states')
                and outputs.encoder_hidden_states is not None):
            available = len(outputs.encoder_hidden_states)
            layers = [l for l in self._align_layers if l < available]
            if not layers:
                layers = [available - 1]
            stacked = torch.stack(
                [outputs.encoder_hidden_states[l][0] for l in layers], dim=0
            )
            return stacked.mean(dim=0)

        if (hasattr(outputs, 'last_hidden_state')
                and outputs.last_hidden_state is not None):
            logger.debug("[FormatAlignment] Używam last_hidden_state jako fallback")
            return outputs.last_hidden_state[0]

        raise RuntimeError(
            "[FormatAlignment] Model nie zwrócił żadnych hidden_states. "
            "Upewnij się, że model jest załadowany z output_hidden_states=True."
        )

    def align_batch(self, paragraphs: List[Dict]) -> int:
        if not self._loaded:
            raise RuntimeError(
                "Model nie jest załadowany. Wywołaj najpierw load_model()."
            )

        processed = 0
        skipped_no_inline = 0
        skipped_reserve = 0
        skipped_other = 0

        for para in paragraphs:
            if para.get('processing_mode') != 'legacy':
                skipped_other += 1
                continue

            if not para.get('is_translated'):
                skipped_other += 1
                continue

            if para.get('is_non_translatable'):
                skipped_other += 1
                continue

            if para.get('reserve_elements'):
                skipped_reserve += 1
                continue

            original_html = para.get('original_html', '').strip()
            translated_text = para.get('translated_text', '').strip()

            if not original_html or not translated_text:
                logger.debug(
                    f"[FormatAlignment] Para {para.get('id', '?')} – "
                    f"pusty original_html lub translated_text – pomijam"
                )
                skipped_other += 1
                continue

            original_plain = re.sub(r'<[^>]+>', '', original_html)
            original_plain = re.sub(r'\s+', ' ', original_plain).strip()
            clean_translated = re.sub(r'</?id_\d{2}>', '', translated_text).strip()
            clean_translated = re.sub(r'\s+', ' ', clean_translated).strip()

            if not clean_translated:
                skipped_other += 1
                continue

            if clean_translated == original_plain:
                logger.debug(
                    f"[FormatAlignment] Para {para.get('id', '?')} – "
                    f"translated_text == original_text – pomijam"
                )
                skipped_other += 1
                continue

            if '<a ' in original_html or '<a>' in original_html:
                logger.debug(
                    f"[FormatAlignment] Para {para.get('id', '?')} "
                    f"zawiera <a> – pomijam alignment"
                )
                skipped_reserve += 1
                continue

            try:
                result_html = self.align_element(original_html, clean_translated)
                if result_html:
                    para['aligned_translated_html'] = result_html
                    processed += 1
                else:
                    skipped_no_inline += 1
            except Exception as exc:
                logger.warning(
                    f"[FormatAlignment] Błąd dla para {para.get('id', '?')}: {exc}",
                    exc_info=True
                )
                skipped_other += 1

        logger.info(
            f"[FormatAlignment] Batch zakończony: "
            f"przetworzone={processed}, "
            f"pominięte(reserve+linki)={skipped_reserve}, "
            f"pominięte(brak tagów inline)={skipped_no_inline}, "
            f"pominięte(inne/nieprzetłumaczone)={skipped_other}"
        )
        return processed

    def _split_to_sentence_spans(
        self,
        words: List[str],
    ) -> List[Tuple[int, int, List[str]]]:
        if not words:
            return []

        sentences: List[Tuple[int, int, List[str]]] = []
        current_start = 0

        for i, word in enumerate(words):
            is_last = (i == len(words) - 1)

            if is_last:
                sub = words[current_start:]
                sentences.append((current_start, i, sub))
                break

            if _RE_SENTENCE_END.search(word):
                if _RE_ABBREV.match(word) or _RE_ELLIPSIS.search(word):
                    continue

                next_word = words[i + 1]
                if not (next_word[0].isupper() or next_word[0].isdigit()):
                    continue

                sub = words[current_start: i + 1]

                if len(sub) < 3 and sentences:
                    prev_start, _, prev_sub = sentences[-1]
                    merged_sub = prev_sub + sub
                    sentences[-1] = (prev_start, i, merged_sub)
                else:
                    sentences.append((current_start, i, sub))

                current_start = i + 1

        if not sentences:
            return [(0, len(words) - 1, words)]

        return sentences

    def _compute_alignments_by_sentences(
        self,
        src_words: List[str],
        tgt_words: List[str],
    ) -> List[Tuple[int, int]]:
        MIN_WORDS_FOR_SENTENCE_SPLIT = 9

        if len(src_words) < MIN_WORDS_FOR_SENTENCE_SPLIT:
            logger.debug(
                f"[FormatAlignment] Krótki tekst ({len(src_words)} słów) "
                f"– alignment bez podziału na zdania"
            )
            result = self._compute_alignments(src_words, tgt_words)
            self._last_src_len = len(src_words)
            return result

        src_sentences = self._split_to_sentence_spans(src_words)
        tgt_sentences = self._split_to_sentence_spans(tgt_words)

        logger.debug(
            f"[FormatAlignment] Podział: {len(src_sentences)} zdań src, "
            f"{len(tgt_sentences)} zdań tgt"
        )

        if len(src_sentences) != len(tgt_sentences):
            logger.debug(
                f"[FormatAlignment] Liczba zdań src≠tgt "
                f"({len(src_sentences)}≠{len(tgt_sentences)}) – fallback pełny"
            )
            result = self._compute_alignments(src_words, tgt_words)
            self._last_src_len = len(src_words)
            return result

        if len(src_sentences) == 1:
            result = self._compute_alignments(src_words, tgt_words)
            self._last_src_len = len(src_words)
            return result

        all_alignments: List[Tuple[int, int]] = []

        for sent_idx, (src_sent, tgt_sent) in enumerate(
            zip(src_sentences, tgt_sentences)
        ):
            s_start, s_end, s_words = src_sent
            t_start, t_end, t_words = tgt_sent

            if not s_words or not t_words:
                continue

            src_counter = Counter(w.lower() for w in s_words if len(w) >= 4)
            has_repetitions = any(v >= 2 for v in src_counter.values())

            if has_repetitions:
                logger.debug(
                    f"[FormatAlignment] Zdanie {sent_idx}: powtórzenia "
                    f"({[k for k, v in src_counter.items() if v >= 2]}) "
                    f"– pomijam (Path B)"
                )
                continue

            try:
                pairs = self._compute_alignments(s_words, t_words)
            except Exception as exc:
                logger.warning(
                    f"[FormatAlignment] Błąd alignment zdania {sent_idx}: {exc}"
                )
                continue

            for (s_local, t_local) in pairs:
                all_alignments.append((
                    s_local + s_start,
                    t_local + t_start,
                ))

            logger.debug(
                f"[FormatAlignment] Zdanie {sent_idx}: {len(pairs)} par "
                f"(src[{s_start}:{s_end}] → tgt[{t_start}:{t_end}])"
            )

        self._last_src_len = len(src_words)
        logger.debug(
            f"[FormatAlignment] _last_src_len przywrócony do {len(src_words)}"
        )

        return all_alignments

    def align_element(self, original_html: str, translated_text: str) -> str:
        if not original_html or not translated_text:
            return ''

        orig_elem = _parse_element_html(original_html)
        if orig_elem is None:
            logger.debug("[FormatAlignment] Nie można sparsować original_html")
            return ''

        has_links = orig_elem.xpath('.//*[local-name()="a"]')
        if has_links:
            logger.debug("[FormatAlignment] Element zawiera <a> – pomijam")
            return ''

        inline_spans = self._extract_inline_spans(orig_elem)
        if not inline_spans:
            return ''

        word_tag_pairs: List = []
        self._walk_collect_words(orig_elem, parent_tags=[], output=word_tag_pairs)
        src_words: List[str] = [word for word, _ in word_tag_pairs]

        tgt_words = _tokenize_words(translated_text)

        if len(src_words) < 1 or len(tgt_words) < 1:
            return ''

        if ' '.join(src_words).lower() == ' '.join(tgt_words).lower():
            logger.debug("[FormatAlignment] src_words == tgt_words – pomijam")
            return ''

        try:
            self._full_src_embs = self._get_word_embeddings(src_words)
            self._full_tgt_embs = self._get_word_embeddings(tgt_words)

            alignments = self._compute_alignments_by_sentences(src_words, tgt_words)
        except Exception as exc:
            logger.warning(f"[FormatAlignment] Błąd wyrównania: {exc}")
            self._full_src_embs = None
            self._full_tgt_embs = None
            return ''

        if not alignments:
            return ''

        src_to_tgt: Dict[int, List[int]] = {}
        for (s, t) in alignments:
            src_to_tgt.setdefault(s, []).append(t)

        tgt_spans = self._transfer_spans(inline_spans, src_to_tgt, len(tgt_words), tgt_words, src_words)
        self._full_src_embs = None
        self._full_tgt_embs = None
        if not tgt_spans:
            return ''

        return self._build_result_html(orig_elem, tgt_words, tgt_spans)

    def _extract_inline_spans(self, element: etree._Element) -> List[Dict]:
        word_tag_pairs: List[Tuple[str, List[Dict]]] = []
        self._walk_collect_words(element, parent_tags=[], output=word_tag_pairs)

        if not word_tag_pairs:
            return []

        spans: List[Dict] = []
        seen_tag_ids: set = set()

        for i, (word, tags) in enumerate(word_tag_pairs):
            for tag_info in tags:
                tid = id(tag_info)
                if tid in seen_tag_ids:
                    continue

                seen_tag_ids.add(tid)

                word_end = i
                for j in range(i, len(word_tag_pairs)):
                    _, jtags = word_tag_pairs[j]
                    if any(t is tag_info for t in jtags):
                        word_end = j

                spans.append({
                    'tag': tag_info['tag'],
                    'attrs': tag_info['attrs'],
                    'word_start': i,
                    'word_end': word_end,
                })

        return spans

    def _walk_collect_words(
        self,
        element: etree._Element,
        parent_tags: List[Dict],
        output: List[Tuple[str, List[Dict]]],
    ) -> None:
        if callable(element.tag):
            return

        tag_name = etree.QName(element).localname

        current_tags = list(parent_tags)
        if tag_name.lower() in INLINE_TRANSFER_TAGS:
            attrs = {
                k: v for k, v in element.attrib.items()
                if not k.startswith('{')
            }
            tag_info = {'tag': tag_name, 'attrs': attrs}
            current_tags = parent_tags + [tag_info]

        if element.text:
            for word in element.text.split():
                if word and _RE_CONTENT_WORD.search(word):
                    output.append((word, current_tags))

        for child in element:
            if callable(child.tag):
                continue
            child_tag = etree.QName(child).localname

            if child_tag.lower() in RESERVE_TAGS:
                if child.tail:
                    for word in child.tail.split():
                        if word and _RE_CONTENT_WORD.search(word):
                            output.append((word, list(parent_tags)))
                continue

            if (child_tag.lower() == 'span'
                    and 'first-letter' in (child.get('class', '') or '').split()):
                child_text = child.text or ''
                child_tail = child.tail or ''
                tail_words = child_tail.split()

                attrs = {k: v for k, v in child.attrib.items() if not k.startswith('{')}
                tag_info = {'tag': child_tag, 'attrs': attrs}
                span_tags = current_tags + [tag_info]

                if child_text and tail_words:
                    merged_word = child_text + tail_words[0]
                    output.append((merged_word, span_tags))
                    for word in tail_words[1:]:
                        if _RE_CONTENT_WORD.search(word):
                            output.append((word, list(parent_tags)))
                elif child_text:
                    output.append((child_text, span_tags))
                else:
                    for word in tail_words:
                        if _RE_CONTENT_WORD.search(word):
                            output.append((word, list(parent_tags)))
                continue

            self._walk_collect_words(child, current_tags, output)

            if child.tail:
                for word in child.tail.split():
                    if word and _RE_CONTENT_WORD.search(word):
                        output.append((word, list(parent_tags)))

    def _transfer_spans(
        self,
        spans: List[Dict],
        src_to_tgt: Dict[int, List[int]],
        tgt_len: int,
        tgt_words: List[str],
        src_words: List[str],
    ) -> List[Dict]:
        src_len = max(1, self._last_src_len)
        result = []

        is_content: List[bool] = [
            bool(_RE_CONTENT_WORD.search(w)) for w in tgt_words
        ]

        def _nearest_content(idx: int) -> int:
            if is_content[idx]:
                return idx
            for offset in range(1, tgt_len):
                for candidate in (idx + offset, idx - offset):
                    if 0 <= candidate < tgt_len and is_content[candidate]:
                        return candidate
            return idx

        def _trim_punct_boundaries(ts: int, te: int) -> Optional[Tuple[int, int]]:
            while ts <= te and not is_content[ts]:
                ts += 1
            while te >= ts and not is_content[te]:
                te -= 1
            if ts > te:
                return None
            return ts, te

        def _norm_w(w: str) -> str:
            w = w.lower()
            w = _RE_APOSTROPHES.sub('', w)
            w = _RE_PUNCT_STRIP.sub('', w)
            return w

        for span in spans:
            if 'first-letter' in span.get('attrs', {}).get('class', '').split():
                logger.debug(
                    f"[FormatAlignment] Span first-letter → wymuszam pozycję 0-0 w tgt"
                )
                result.append({
                    'tag':        span['tag'],
                    'attrs':      span['attrs'],
                    'word_start': 0,
                    'word_end':   0,
                })
                continue

            ws = span['word_start']
            we = span['word_end']

            src_coverage = (we - ws + 1) / src_len
            if ws == 0 and src_coverage >= 0.9:
                logger.debug(
                    f"[FormatAlignment] Span '{span['tag']}' pokrywa cały src "
                    f"({ws}-{we}/{src_len-1}) → rozszerzam na cały tgt (0-{tgt_len-1})"
                )
                result.append({
                    'tag':        span['tag'],
                    'attrs':      span['attrs'],
                    'word_start': 0,
                    'word_end':   tgt_len - 1,
                })
                continue

            tgt_indices = []
            for s in range(ws, we + 1):
                tgt_indices.extend(src_to_tgt.get(s, []))

            span_len = we - ws + 1
            _neural_span_n = (max(tgt_indices) - min(tgt_indices) + 1) if tgt_indices else 0
            _neural_is_contiguous_multi = (
                len(tgt_indices) > 1
                and _neural_span_n == len(tgt_indices)
                and _neural_span_n >= span_len
            )
            if (span_len <= 3
                    and not _neural_is_contiguous_multi
                    and self._full_src_embs is not None
                    and self._full_tgt_embs is not None):
                try:
                    span_src_emb = self._full_src_embs[ws: we + 1].mean(dim=0, keepdim=True)
                    span_src_norm = F.normalize(span_src_emb, p=2, dim=-1)
                    tgt_norm_emb  = F.normalize(self._full_tgt_embs, p=2, dim=-1)
                    direct_sim    = torch.mv(tgt_norm_emb, span_src_norm.squeeze())

                    for t_idx, ok in enumerate(is_content):
                        if not ok:
                            direct_sim[t_idx] = float('-inf')

                    direct_best_t = int(direct_sim.argmax().item())
                    direct_best_v = direct_sim[direct_best_t].item()

                    neural_best_t = tgt_indices[0] if len(tgt_indices) == 1 else (
                        min(tgt_indices) if tgt_indices else None
                    )
                    neural_best_v = (
                        direct_sim[neural_best_t].item()
                        if neural_best_t is not None else -1.0
                    )

                    SIM_ACCEPT = 0.20
                    MARGIN     = 0.02

                    if (direct_best_v >= SIM_ACCEPT
                            and direct_best_v >= neural_best_v + MARGIN):
                        logger.debug(
                            f"[FormatAlignment] Span '{span['tag']}' [{ws}-{we}]: "
                            f"direct sim override → tgt[{direct_best_t}] "
                            f"(sim={direct_best_v:.3f} vs neural={neural_best_v:.3f})"
                        )
                        tgt_indices = [direct_best_t]
                except Exception as exc:
                    logger.debug(
                        f"[FormatAlignment] Direct-sim lookup failed for span "
                        f"'{span['tag']}': {exc}"
                    )

            if not tgt_indices:
                src_total = max(1, src_len - 1)
                ratio_start = ws / src_total
                ratio_end   = we / src_total
                t_start = min(tgt_len - 1, int(ratio_start * (tgt_len - 1)))
                t_end   = min(tgt_len - 1, int(ratio_end   * (tgt_len - 1)))
                tgt_indices = list(range(t_start, t_end + 1))

            if not tgt_indices:
                continue

            t_start = max(0,         min(tgt_indices))
            t_end   = min(tgt_len-1, max(tgt_indices))

            src_total = max(1, src_len - 1)
            prop_t_start = _nearest_content(
                min(tgt_len - 1, int(ws / src_total * (tgt_len - 1)))
            )
            prop_t_end   = _nearest_content(
                min(tgt_len - 1, round(we / src_total * (tgt_len - 1)))
            )

            span_words = we - ws + 1
            drift_pct  = 0.20 if span_words <= 2 else 0.35
            drift_limit = max(1, int(drift_pct * tgt_len))
            if abs(t_start - prop_t_start) >= drift_limit:
                logger.debug(
                    f"[FormatAlignment] Span '{span['tag']}': neural drift too large "
                    f"(neural=[{t_start},{t_end}] vs prop=[{prop_t_start},{prop_t_end}], "
                    f"limit={drift_limit}, pct={drift_pct}) → falling back to proportional"
                )
                t_start = prop_t_start
                t_end   = prop_t_end

            _src_span_norms = {
                _norm_w(src_words[i])
                for i in range(ws, min(we + 1, len(src_words)))
                if len(_norm_w(src_words[i])) >= 2
            }
            _tgt_cur_norms = {
                _norm_w(tgt_words[i])
                for i in range(t_start, t_end + 1)
                if len(_norm_w(tgt_words[i])) >= 2
            }
            _has_exact_anchor = bool(_src_span_norms & _tgt_cur_norms)

            if t_start > prop_t_start and not _has_exact_anchor:
                slide  = t_start - prop_t_start
                new_ts = max(0, t_start - slide)
                new_te = min(tgt_len - 1, t_end - slide)
                if new_te >= new_ts:
                    logger.debug(
                        f"[FormatAlignment] Span '{span['tag']}' [{ws}-{we}]: "
                        f"right-drift slide {slide} → [{new_ts},{new_te}] "
                        f"(was [{t_start},{t_end}], prop_t_start={prop_t_start})"
                    )
                    t_start, t_end = new_ts, new_te

            span_src_words_n = we - ws + 1
            expected_tgt_span_n = max(1, round(span_src_words_n * tgt_len / max(1, src_len)))
            current_tgt_span_n  = t_end - t_start + 1
            if current_tgt_span_n < expected_tgt_span_n:
                extra  = expected_tgt_span_n - current_tgt_span_n
                half_l = (extra + 1) // 2

                new_ts = max(t_start - half_l, prop_t_start)

                new_te = min(tgt_len - 1, new_ts + expected_tgt_span_n - 1)

                if new_ts > prop_t_start:
                    shift  = new_ts - prop_t_start
                    new_ts = prop_t_start
                    new_te = max(new_ts, new_te - shift)
                logger.debug(
                    f"[FormatAlignment] Span '{span['tag']}' [{ws}-{we}]: "
                    f"expansion [{t_start},{t_end}] → [{new_ts},{new_te}] "
                    f"(exp={expected_tgt_span_n}, was={current_tgt_span_n}, "
                    f"prop_t_start={prop_t_start})"
                )
                t_start, t_end = new_ts, new_te

            if span_src_words_n >= 3 and t_end > prop_t_end:
                clipped = t_end - prop_t_end
                t_end   = prop_t_end
                t_start = max(0, t_start - clipped)
                logger.debug(
                    f"[FormatAlignment] Span '{span['tag']}' [{ws}-{we}]: "
                    f"t_end ceiling → [{t_start},{t_end}] (clipped={clipped})"
                )
                if t_start > t_end:
                    t_start = t_end

            tgt_coverage = (t_end - t_start + 1) / tgt_len

            trimmed = _trim_punct_boundaries(t_start, t_end)
            if trimmed is None:
                logger.debug(
                    f"[FormatAlignment] Span '{span['tag']}' [{ws}-{we}]: "
                    f"entire tgt span [{t_start},{t_end}] is pure-punct → skipping"
                )
                continue
            t_start, t_end = trimmed

            result.append({
                'tag':        span['tag'],
                'attrs':      span['attrs'],
                'word_start': t_start,
                'word_end':   t_end,
            })

        return result

    def _build_result_html(
        self,
        orig_elem: etree._Element,
        tgt_words: List[str],
        tgt_spans: List[Dict],
    ) -> str:
        word_spans: List[List[Dict]] = [[] for _ in range(len(tgt_words))]
        for span in tgt_spans:
            for wi in range(span['word_start'], min(span['word_end'] + 1, len(tgt_words))):
                word_spans[wi].append(span)

        inner_parts: List[str] = []
        active: List[Dict] = []

        for wi, word in enumerate(tgt_words):
            wanted = word_spans[wi]
            wanted_ids = {id(s) for s in wanted}
            active_ids = {id(s) for s in active}

            to_close = [s for s in reversed(active) if id(s) not in wanted_ids]
            for s in to_close:
                inner_parts.append(f'</{s["tag"]}>')
                active = [x for x in active if x is not s]

            if wi > 0:
                inner_parts.append(' ')

            to_open = [s for s in wanted if id(s) not in active_ids]
            for s in to_open:
                attrs_str = _build_attrs_str(s['attrs'])
                inner_parts.append(f'<{s["tag"]}{attrs_str}>')
                active.append(s)

            inner_parts.append(_escape_xml(word))

        for s in reversed(active):
            inner_parts.append(f'</{s["tag"]}>')

        inner_html = ''.join(inner_parts)

        outer_tag = etree.QName(orig_elem).localname
        outer_attrs = _build_attrs_str({
            k: v for k, v in orig_elem.attrib.items()
            if not k.startswith('{')
        })

        full_html = (
            f'<{outer_tag} xmlns="{XHTML_NS}"{outer_attrs}>'
            f'{inner_html}'
            f'</{outer_tag}>'
        )

        try:
            etree.fromstring(full_html.encode('utf-8'))
            return full_html
        except etree.XMLSyntaxError as exc:
            logger.warning(
                f"[FormatAlignment] Wygenerowany HTML jest nieprawidłowy: {exc}\n"
                f"Fragment: {full_html[:300]}"
            )
            return ''

    def _itermax(
        self,
        sim,
        S: int,
        T: int,
    ) -> List[Tuple[int, int]]:
        SIM_THRESHOLD = 0.05
        pairs: List[Tuple[int, int]] = []
        sim_work = sim.clone().float()
        max_pairs = min(S, T)

        for _ in range(max_pairs):
            best_val = sim_work.max().item()
            if best_val < SIM_THRESHOLD:
                break

            flat_idx = sim_work.argmax().item()
            s = int(flat_idx) // T
            t = int(flat_idx) % T

            pairs.append((s, t))
            sim_work[s, :] = -2.0
            sim_work[:, t] = -2.0

        return sorted(pairs)

    def _itermax_bidirectional(
        self,
        sim,
        S: int,
        T: int,
    ) -> List[Tuple[int, int]]:
        fwd_pairs = set(self._itermax(sim, S, T))
        bwd_raw   = self._itermax(sim.t().contiguous(), T, S)
        bwd_pairs = {(s, t) for (t, s) in bwd_raw}

        intersection = fwd_pairs & bwd_pairs

        covered_src = {s for (s, _) in intersection}
        for (s, t) in fwd_pairs:
            if s not in covered_src:
                intersection.add((s, t))

        return sorted(intersection)

    def _compute_alignments(
        self,
        src_words: List[str],
        tgt_words: List[str],
    ) -> List[Tuple[int, int]]:
        self._last_src_len = len(src_words)
        S = len(src_words)
        T = len(tgt_words)

        if S == 1:
            return [(0, t) for t in range(T)]
        if T == 1:
            return [(s, 0) for s in range(S)]

        def _normalize(token: str) -> str:
            t = token.lower()
            t = _RE_APOSTROPHES.sub('', t)
            t = _RE_PUNCT_STRIP.sub('', t)
            return t

        _MIN_MATCH_LEN = 2

        src_norm = [_normalize(w) for w in src_words]
        tgt_norm = [_normalize(w) for w in tgt_words]

        forced_s2t: Dict[int, int] = {}
        forced_used_tgt: set = set()

        for s_idx, s_n in enumerate(src_norm):
            if len(s_n) < _MIN_MATCH_LEN:
                continue
            for t_idx, t_n in enumerate(tgt_norm):
                if t_idx in forced_used_tgt:
                    continue
                if s_n == t_n:
                    forced_s2t[s_idx] = t_idx
                    forced_used_tgt.add(t_idx)
                    logger.debug(
                        f"[FormatAlignment] Forced: '{src_words[s_idx]}' "
                        f"(norm='{s_n}') → '{tgt_words[t_idx]}' (idx {s_idx}→{t_idx})"
                    )
                    break

        src_embs = self._get_word_embeddings(src_words)
        tgt_embs = self._get_word_embeddings(tgt_words)

        neural_pairs: List[Tuple[int, int]] = []

        if src_embs is not None and tgt_embs is not None:
            src_norm_emb = F.normalize(src_embs, p=2, dim=-1)
            tgt_norm_emb = F.normalize(tgt_embs, p=2, dim=-1)
            sim = torch.mm(src_norm_emb, tgt_norm_emb.t())

            for t_idx, tw in enumerate(tgt_words):
                if not _RE_CONTENT_WORD.search(tw):
                    sim[:, t_idx] = -2.0
                    logger.debug(
                        f"[FormatAlignment] Masking pure-punct tgt[{t_idx}]='{tw}'"
                    )

            if forced_s2t:
                forced_s_list = list(forced_s2t.keys())
                forced_t_list = list(forced_s2t.values())
                sim[forced_s_list, :] = -2.0
                sim[:, forced_t_list] = -2.0

            neural_pairs = self._itermax_bidirectional(sim, S, T)

        else:
            logger.warning("[FormatAlignment] Brak embeddingów – używam fallback proporcjonalny")
            for s_idx in range(S):
                if s_idx in forced_s2t:
                    continue
                t_idx = min(T - 1, int(s_idx / max(1, S - 1) * (T - 1)))
                if t_idx not in forced_used_tgt:
                    neural_pairs.append((s_idx, t_idx))

        combined: Dict[int, int] = {}

        for s, t in neural_pairs:
            if s in forced_s2t:
                continue
            if t in forced_used_tgt:
                continue
            combined[s] = t

        combined.update(forced_s2t)

        result = sorted(combined.items())

        logger.debug(
            f"[FormatAlignment] Alignmenty: {len(forced_s2t)} forced + "
            f"{len(result) - len(forced_s2t)} neural = {len(result)} par łącznie "
            f"(S={S}, T={T})"
        )

        return result

    def _get_word_embeddings(self, words: List[str]):
        if not words:
            return None

        text = ' '.join(words)
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            add_special_tokens=True,
            truncation=True,
            max_length=512,
        )
        encoding = {k: v.to(self.model.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding, output_hidden_states=True)

        hidden = self._get_layer_output(outputs)

        token_ids = encoding['input_ids'][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        word_ids = self._build_word_ids(words, tokens)

        H = hidden.shape[-1]
        word_embs = torch.zeros(len(words), H, device=self.model.device)
        word_counts = torch.zeros(len(words), device=self.model.device)

        for tok_idx, wi in enumerate(word_ids):
            if wi is not None and 0 <= wi < len(words):
                word_embs[wi] += hidden[tok_idx]
                word_counts[wi] += 1

        if word_counts.sum().item() == 0:
            logger.warning(
                f"[FormatAlignment] _get_word_embeddings: żadne słowo nie dostało "
                f"embeddingu (words={words[:5]}, tokens={tokens[:10]})"
            )
            return None

        word_counts = word_counts.clamp(min=1.0)
        word_embs = word_embs / word_counts.unsqueeze(-1)

        return word_embs

    def _build_word_ids(
        self,
        words: List[str],
        tokens: List[str],
    ) -> List[Optional[int]]:
        word_ids: List[Optional[int]] = [None] * len(tokens)

        try:
            special_tokens: set = set(self.tokenizer.all_special_tokens)
        except Exception:
            special_tokens = {
                '[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]',
                '<s>', '</s>', '<pad>', '<mask>', '<unk>',
            }

        if self._sp_model:
            word_idx = -1
            first_content_seen = False
            prev_bare_tok: Optional[str] = None

            for tok_pos, tok in enumerate(tokens):
                if tok in special_tokens:
                    continue

                bare_tok = tok.lstrip('\u2581')
                starts_new_word = tok.startswith('\u2581') or not first_content_seen

                if starts_new_word and prev_bare_tok in _HYPHEN_JOINERS:
                    starts_new_word = False

                if starts_new_word:
                    word_idx += 1
                    first_content_seen = True

                if 0 <= word_idx < len(words):
                    word_ids[tok_pos] = word_idx

                prev_bare_tok = bare_tok

        else:
            word_idx = -1

            for tok_pos, tok in enumerate(tokens):
                if tok in special_tokens:
                    continue

                if not tok.startswith('##'):
                    word_idx += 1

                if 0 <= word_idx < len(words):
                    word_ids[tok_pos] = word_idx

        return word_ids
