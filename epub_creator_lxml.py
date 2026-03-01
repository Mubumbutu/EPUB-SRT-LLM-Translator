# epub_creator_lxml.py
import copy
import html
import logging
import re
import traceback
import unicodedata
from epub_utils import write_epub
from format_alignment import FormatAlignmentEngine
from lxml import etree
from PyQt6.QtCore import pyqtSignal, QThread

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class EPUBCreatorLxml(QThread):
    finished = pyqtSignal(str, bool)

    def __init__(self, book, paragraphs, output_path):
        super().__init__()
        self.book = book
        self.paragraphs = paragraphs
        self.output_path = output_path

        self.position = 'only'
        self.original_color = None
        self.translation_color = None
        self.target_direction = 'auto'
        self.translation_lang = None
        self.column_gap = None

        self.ns = {
            'x': 'http://www.w3.org/1999/xhtml'
        }

        self.table_tags = {'li', 'th', 'td', 'caption'}

        self.alignment_enabled: bool = False
        self.alignment_model_name: str = 'xlm-roberta-large'
        self.alignment_device: str = 'cpu'
        self.alignment_models_dir: str = ''

    def run(self):
        try:
            logger.debug(f"Starting EPUB save to: {self.output_path}")

            if self.alignment_enabled:
                logger.info("[EPUBCreator] Alignment wlaczony – uruchamiam mBERT batch...")
                self._run_alignment()

            self._insert_translations()

            write_epub(self.output_path, self.book)

            logger.debug("EPUB save completed successfully")
            self.finished.emit(self.output_path, False)

        except Exception as e:
            logger.exception("Error during EPUB creation")
            logger.error(traceback.format_exc())
            self.finished.emit(str(e), True)

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ''

        text = unicodedata.normalize('NFKC', text)

        text = text.replace('\u2014', '\u2013')
        text = text.replace('\u2012', '\u2013')

        for ch in '\u201c\u201d\u201e\u201f\u00ab\u00bb\u301d\u301e\u301f\uff02':
            text = text.replace(ch, '"')
        for ch in '\u2018\u2019\u201a\u201b\u2032\u2039\u203a':
            text = text.replace(ch, "'")

        text = ' '.join(text.split())

        return text.lower()

    def _insert_translations(self):
        for item in self.book.get_items_of_type('DOCUMENT'):
            if item.data is None:
                continue

            for para in self.paragraphs:
                if not para.get('is_translated'):
                    continue

                if para['item_href'] != item.href:
                    continue

                xpath = f'.//*[@id="{para["id"]}"]'
                elements = item.data.xpath(xpath, namespaces=self.ns)

                if not elements:
                    logger.warning(
                        f"Element not found by ID: {para['id']}, "
                        f"searching by normalized content..."
                    )

                    original_text = para.get('original_text', '').strip()

                    if not original_text:
                        logger.error(
                            f"Element not found: {para['id']} (no original_text to search)"
                        )
                        continue

                    tag = para.get('element_type', 'p')
                    candidate_xpath = f'.//x:{tag}'

                    try:
                        candidates = item.data.xpath(candidate_xpath, namespaces=self.ns)
                        logger.debug(f"  Found {len(candidates)} <{tag}> candidates")
                    except Exception as e:
                        logger.error(f"  XPath error: {e}, xpath was: {candidate_xpath}")
                        continue

                    norm_original = self._normalize_text(original_text)

                    element = None
                    for candidate in candidates:
                        candidate_text = self._get_element_text_clean(candidate)
                        norm_candidate = self._normalize_text(candidate_text)

                        if norm_candidate == norm_original:
                            element = candidate
                            logger.info(
                                f"  ✓ Found element by normalized content match! "
                                f"(para id={para['id']})"
                            )

                            existing_id = element.get('id', '')
                            if not existing_id:
                                element.set('id', para['id'])
                                logger.debug(f"  Added missing ID: {para['id']}")
                            elif existing_id != para['id']:
                                logger.debug(
                                    f"  ID mismatch: EPUB has '{existing_id}', "
                                    f"session has '{para['id']}' – using EPUB ID"
                                )
                                para['id'] = existing_id

                            break

                    if element is None and len(norm_original) > 50:
                        prefix = norm_original[:100]
                        logger.warning(
                            f"  Exact match failed – trying prefix match "
                            f"(first 100 chars): '{prefix[:50]}...'"
                        )

                        for candidate in candidates:
                            candidate_text = self._get_element_text_clean(candidate)
                            norm_candidate = self._normalize_text(candidate_text)

                            if norm_candidate.startswith(prefix):
                                element = candidate
                                logger.info(f"  ✓ Found element by prefix match!")

                                existing_id = element.get('id', '')
                                if not existing_id:
                                    element.set('id', para['id'])
                                elif existing_id != para['id']:
                                    para['id'] = existing_id

                                break

                    if element is None:
                        logger.error(
                            f"  ✗ Element not found by any strategy: "
                            f"id={para['id']}, text={original_text[:80]}..."
                        )
                        continue
                else:
                    element = elements[0]

                self._add_translation_to_element(element, para, item.data)

    def _get_element_text_clean(self, element):
        element_copy = copy.deepcopy(element)

        RESERVE_TAGS = ['img', 'code', 'br', 'hr', 'sub', 'sup', 'kbd',
                       'abbr', 'wbr', 'var', 'canvas', 'svg', 'script',
                       'style', 'math']

        for reserve_tag in RESERVE_TAGS:
            reserve_xpath = f'.//x:{reserve_tag}'
            for reserve_elem in element_copy.xpath(reserve_xpath, namespaces=self.ns):
                parent = reserve_elem.getparent()
                if parent is not None:
                    parent.remove(reserve_elem)

        text = etree.tostring(
            element_copy,
            encoding='unicode',
            method='text'
        )

        return ' '.join(text.split())

    def _add_translation_to_element(self, element, para, root):
        translation = para['translated_text']
        element_name = para['element_type']

        parent = element.getparent()
        if parent is None:
            logger.error(f"Cannot translate element {para['id']}: no parent")
            return

        logger.debug(f"\n=== TRANSLATING ELEMENT {para['id']} ===")
        logger.debug(f"Element tag: {element.tag}")
        logger.debug(f"Element.text: {repr(element.text)}")
        logger.debug(f"Element.tail: {repr(element.tail)}")
        logger.debug(f"Translation: {repr(translation[:100])}")

        links_in_original = element.xpath('.//x:a', namespaces=self.ns)
        logger.debug(f"Links in ORIGINAL element: {len(links_in_original)}")
        if links_in_original:
            logger.debug(f"  Link href: {links_in_original[0].get('href')}")

        processing_mode = para.get('processing_mode', 'inline')
        is_legacy_mode = processing_mode == 'legacy'
        logger.debug(f"Mode: {processing_mode}")

        if is_legacy_mode and para.get('aligned_translated_html'):
            logger.debug("Using ALIGNED element from FormatAlignmentEngine")
            new_element = self._build_element_from_aligned_html(element, para)
            if new_element is None:
                logger.warning(
                    f"_build_element_from_aligned_html zwrocil None dla {para['id']} "
                    "– fallback do legacy path"
                )
                new_element = self._legacy_path(element, element_name, translation, para, root)

        elif not is_legacy_mode and 'inline_formatting_map' in para:
            logger.debug(
                "Using DIRECT inline restoration (bypassing _create_clean_translation_element)"
            )
            new_element = copy.deepcopy(element)

            original_id = element.get('id')
            if original_id:
                if original_id.startswith('trans_'):
                    new_element.set('id', original_id)
                    logger.debug(f"ID already has trans_ prefix: {original_id}")
                else:
                    new_element.set('id', f"trans_{original_id}")
                    logger.debug(f"Added trans_ prefix: {original_id} -> trans_{original_id}")

            self._restore_inline_translation(new_element, para)

        else:
            logger.debug("Using legacy path via _create_clean_translation_element")
            new_element = self._legacy_path(element, element_name, translation, para, root)

        links_in_new = new_element.xpath('.//x:a', namespaces=self.ns)
        logger.debug(f"Links in NEW element: {len(links_in_new)}")
        if links_in_new:
            logger.debug(f"  Link href: {links_in_new[0].get('href')}")
        elif links_in_original:
            logger.error("LINK WAS LOST!")
            logger.error(
                f"New element HTML:\n"
                f"{etree.tostring(new_element, encoding='unicode', pretty_print=True)}"
            )

        preserved_tail = element.tail
        new_element.tail = preserved_tail
        logger.debug(f"Preserved tail: {repr(preserved_tail)}")

        parent.replace(element, new_element)
        logger.debug(f"Replaced element {para['id']}")

    def _legacy_path(self, element, element_name, translation, para, root):
        translation = self._restore_reserved_elements(translation, para)
        translation = self._cleanup_translation(translation)

        translation = re.sub(r'(?<=[^\s>])(<a[\s>])', r' \1', translation)

        translation = re.sub(r'(</a>)(?=[^\s<,;:!?.\u2013\u2014\-])', r'\1 ', translation)

        return self._create_clean_translation_element(
            element,
            element_name,
            translation,
            root,
            is_legacy_mode=True,
        )

    def _build_element_from_aligned_html(self, original_element, para):
        aligned_html = para.get('aligned_translated_html', '')
        if not aligned_html:
            return None

        translated_text = para.get('translated_text', '').strip()
        if translated_text:
            aligned_plain = re.sub(r'<[^>]+>', '', aligned_html)
            aligned_plain = re.sub(r'\s+', ' ', aligned_plain).strip()
            original_html = para.get('original_html', '')
            original_plain = re.sub(r'<[^>]+>', '', original_html)
            original_plain = re.sub(r'\s+', ' ', original_plain).strip()

            if aligned_plain == original_plain:
                logger.warning(
                    f"[EPUBCreator] aligned_translated_html zawiera oryginalny tekst "
                    f"dla para {para.get('id', '?')} – fallback do legacy path"
                )
                return None

        prefix_tags = para.get('prefix_reserve_tags', [])
        suffix_tags = para.get('suffix_reserve_tags', [])

        try:
            aligned_elem = etree.fromstring(aligned_html.encode('utf-8'))
        except etree.XMLSyntaxError as exc:
            logger.warning(
                f"[EPUBCreator] Nie można sparsować aligned_translated_html: {exc}\n"
                f"HTML (pierwsze 300 zn.): {aligned_html[:300]}"
            )
            return None

        para_id = para.get('id', '')
        if para_id:
            clean_id = para_id
            if clean_id.startswith('trans_'):
                clean_id = clean_id[len('trans_'):]
            aligned_elem.set('id', f"trans_{clean_id}")
        else:
            aligned_elem.attrib.pop('id', None)

        aligned_elem.set('dir', self.target_direction or 'auto')

        if self.translation_lang:
            aligned_elem.set('lang', self.translation_lang)

        if self.translation_color:
            style = aligned_elem.get('style', '').strip().rstrip(';')
            color_style = f"color:{self.translation_color}"
            if style:
                aligned_elem.set('style', f"{style}; {color_style}")
            else:
                aligned_elem.set('style', color_style)

        if prefix_tags:
            prefix_html = ' '.join(prefix_tags)
            current_text = aligned_elem.text or ''
            aligned_elem.text = prefix_html + ' ' + current_text

        if suffix_tags:
            suffix_html = ' '.join(suffix_tags)
            children = list(aligned_elem)
            if children:
                last = children[-1]
                last.tail = (last.tail or '') + ' ' + suffix_html
            else:
                aligned_elem.text = (aligned_elem.text or '') + ' ' + suffix_html

        self._fix_drop_cap_in_element(aligned_elem)
        self._fix_last_word_in_element(aligned_elem)

        return aligned_elem

    def _fix_drop_cap_in_element(self, element) -> None:
        first_letter_spans = []
        for child in element.iter():
            if child is element:
                continue
            tag = etree.QName(child).localname if not hasattr(child.tag, '__call__') else ''
            if tag == 'span':
                class_attr = child.get('class', '')
                if 'first-letter' in class_attr.split():
                    first_letter_spans.append(child)

        if not first_letter_spans:
            return

        span = first_letter_spans[0]
        span_text = span.text or ''

        if len(span_text) <= 1:
            return

        first_char = span_text[0]
        remainder = span_text[1:]

        logger.debug(
            f"[fix_drop_cap] span.first-letter zawierał '{span_text}' – "
            f"skracam do '{first_char}', reszta: '{remainder}'"
        )

        span.text = first_char

        current_tail = span.tail or ''
        if remainder and current_tail and not current_tail.startswith(' '):
            span.tail = remainder + ' ' + current_tail
        else:
            span.tail = remainder + current_tail

        logger.debug(f"[fix_drop_cap] Naprawiono drop cap: text='{first_char}', tail='{span.tail[:50]}'")

    def _create_drop_cap_span(self, element, nsmap, class_attr) -> None:
        text = element.text or ''
        if not text:
            logger.debug("[create_drop_cap] element.text jest pusty – pomijam")
            return

        first_alpha_idx = -1
        for i, ch in enumerate(text):
            if ch.isalpha():
                first_alpha_idx = i
                break

        if first_alpha_idx == -1:
            logger.debug("[create_drop_cap] Nie znaleziono litery w element.text – pomijam")
            return

        before_text = text[:first_alpha_idx]
        first_char = text[first_alpha_idx]
        after_text = text[first_alpha_idx + 1:]

        ns = (nsmap or {}).get(None, self.ns['x'])
        effective_nsmap = nsmap if nsmap else {None: ns}

        new_span = etree.Element(f"{{{ns}}}span", nsmap=effective_nsmap)
        new_span.set('class', class_attr)
        new_span.text = first_char
        new_span.tail = after_text

        element.text = before_text if before_text else None

        element.insert(0, new_span)

        logger.debug(
            f"[create_drop_cap] Utworzono span.{class_attr}: "
            f"text='{first_char}', tail='{after_text[:40]}'"
        )

    def _fix_last_word_in_element(self, element) -> None:
        last_word_spans = []
        for child in element.iter():
            if child is element:
                continue
            tag = etree.QName(child).localname if not hasattr(child.tag, '__call__') else ''
            if tag == 'span':
                class_attr = child.get('class', '')
                if 'last-word' in class_attr.split():
                    last_word_spans.append(child)

        if not last_word_spans:
            return

        full_text = self._get_element_text_clean(element)
        words = full_text.split()
        if not words:
            return

        last_word = words[-1]

        for span in last_word_spans:
            parent = span.getparent()
            if parent is None:
                continue
            span_text = span.text or ''
            tail = span.tail or ''
            combined = span_text + tail
            parent.remove(span)
            if not list(parent):
                if parent.text is None:
                    parent.text = ''
                parent.text += combined
            else:
                last_child = parent[-1]
                last_child.tail = (last_child.tail or '') + combined

        text_nodes = list(element.xpath('.//text()'))
        for node in reversed(text_nodes):
            node_text = str(node)
            if last_word in node_text:
                parent = node.getparent()
                if parent is None:
                    continue

                if parent.text == node:
                    location = 'text'
                else:
                    location = 'tail'

                pos = node_text.rfind(last_word)
                if pos == -1:
                    continue
                prefix = node_text[:pos]
                suffix = node_text[pos + len(last_word):]

                if location == 'text':
                    parent.text = prefix
                else:
                    parent.tail = prefix

                nsmap = parent.nsmap
                default_ns = nsmap.get(None, self.ns['x'])
                new_span = etree.Element(f"{{{default_ns}}}span", nsmap=nsmap)
                new_span.set('class', 'last-word')
                new_span.text = last_word

                if location == 'text':
                    parent.insert(0, new_span)
                    new_span.tail = suffix
                else:
                    grandparent = parent.getparent()
                    if grandparent is None:
                        parent.append(new_span)
                        new_span.tail = suffix
                    else:
                        idx = list(grandparent).index(parent)
                        grandparent.insert(idx + 1, new_span)
                        new_span.tail = suffix
                break

    def _run_alignment(self):
        try:
            pass
        except ImportError as exc:
            logger.error(
                f"[EPUBCreator] Nie mozna zaimportowac format_alignment: {exc}. "
                "Zapisuje bez alignmentu."
            )
            return

        engine = FormatAlignmentEngine(
            model_name=self.alignment_model_name,
            device=self.alignment_device,
            local_models_dir=self.alignment_models_dir,
        )

        try:
            logger.info(
                f"[EPUBCreator] Ladowanie mBERT '{self.alignment_model_name}' "
                f"na '{self.alignment_device}'..."
            )
            if self.alignment_models_dir:
                logger.info(
                    f"[EPUBCreator] Lokalny folder modeli: {self.alignment_models_dir}"
                )

            engine.load_model()

            legacy_count = sum(
                1 for p in self.paragraphs
                if (
                    p.get('processing_mode') == 'legacy'
                    and p.get('is_translated')
                    and not p.get('is_non_translatable')
                    and not p.get('reserve_elements')
                )
            )
            logger.info(
                f"[EPUBCreator] Paragrafow legacy do alignmentu (kwalifikujacych): "
                f"{legacy_count}"
            )

            processed = engine.align_batch(self.paragraphs)
            logger.info(
                f"[EPUBCreator] Alignment zakonczony: {processed} paragrafow przetworzonych."
            )

        except Exception as exc:
            logger.error(
                f"[EPUBCreator] Blad podczas alignmentu: {exc}. "
                "Kontynuuje zapis bez alignmentu.",
                exc_info=True,
            )
        finally:
            try:
                engine.unload_model()
                logger.info("[EPUBCreator] Model mBERT zwolniony z pamieci.")
            except Exception as exc:
                logger.warning(f"[EPUBCreator] Blad przy zwalnianiu modelu: {exc}")

    def _restore_translation_to_element(self, element, para):
        logger.debug(f"\n=== RESTORE TRANSLATION ===")
        logger.debug(f"Element: {element.get('id')}")

        translated_text = para.get('translated_text', '')

        if not translated_text.strip():
            logger.warning(f"Empty translation for element {para['id']}")
            return

        processing_mode = para.get('processing_mode', 'inline')

        logger.debug(f"Processing mode: {processing_mode}")

        if processing_mode == 'legacy':
            self._restore_legacy_translation(element, para)
        else:
            self._restore_inline_translation(element, para)

    def _restore_legacy_translation(self, element, para):
        logger.debug("Using LEGACY restoration")

        translated_text = para['translated_text']
        reserve_elements = para.get('reserve_elements', [])
        prefix_tags = para.get('prefix_reserve_tags', [])
        suffix_tags = para.get('suffix_reserve_tags', [])

        full_text = ""
        for tag in prefix_tags:
            full_text += tag + " "
        full_text += translated_text.strip()
        for tag in suffix_tags:
            full_text += " " + tag

        for idx, reserve_html in enumerate(reserve_elements):
            placeholder = f'<id_{idx:02d}>'
            full_text = full_text.replace(placeholder, reserve_html)

        try:
            nsmap = element.nsmap
            default_ns = nsmap.get(None, self.ns['x'])

            fragment = f'<temp xmlns="{default_ns}">{full_text}</temp>'
            temp = etree.fromstring(fragment.encode('utf-8'))

            element.text = temp.text

            for child in list(element):
                element.remove(child)

            for child in temp:
                element.append(child)

        except Exception as e:
            logger.error(f"Failed to parse legacy content: {e}")
            element.text = full_text

    def _restore_inline_translation(self, element, para):
        logger.debug(f"\n=== RESTORE TRANSLATION ===")
        logger.debug(f"Element ID: {element.get('id', 'UNKNOWN')}")

        translated_text = para.get('translated_text', '')
        inline_formatting_map = para.get('inline_formatting_map', {})
        reserve_elements = para.get('reserve_elements', [])
        non_translatable = para.get('non_translatable_placeholders', {})
        prefix_tags = para.get('prefix_reserve_tags', [])
        suffix_tags = para.get('suffix_reserve_tags', [])
        auto_wrap_tags = para.get('auto_wrap_tags', [])

        if '\n' in translated_text:
            translated_text = re.sub(r'\n\s*(?=<)', '', translated_text)
            translated_text = re.sub(r'(?<=>)\s*\n', '', translated_text)
            translated_text = translated_text.replace('\n', ' ')
            translated_text = re.sub(r'  +', ' ', translated_text)
            logger.debug(f"Cleaned \\n from translated_text")

        _has_ps_tag = bool(re.search(r'</?ps(?:_\d{2})?>', translated_text))
        if _has_ps_tag:
            ps_count = len(re.findall(r'</?ps(?:_\d{2})?>', translated_text))
            logger.warning(
                f"Found {ps_count} unreplaced <ps>/<ps> or <ps_NN> markers in translated_text "
                f"for element {para.get('id', '?')} – removing them"
            )
            translated_text = re.sub(r'</?ps(?:_\d{2})?>', '', translated_text)
            translated_text = re.sub(r'  +', ' ', translated_text)

        translated_text = re.sub(
            r'(</p_\d{2}>)[ \t\u00a0]+([,;:!?.])',
            r'\1\2',
            translated_text
        )
        translated_text = re.sub(
            r'[ \t\u00a0]+(</p_\d{2}>)',
            r'\1',
            translated_text
        )
        translated_text = re.sub(
            r'(?<![>])[ \t\u00a0]+([,;:!?.])',
            r'\1',
            translated_text
        )
        logger.debug(f"Applied FIX 10c (punctuation spacing cleanup)")

        _dup_pattern = r'(\b(\S+))(</(?:strong|em|b|i|u|small)>)\s*\2\b'
        if re.search(_dup_pattern, translated_text, flags=re.IGNORECASE):
            translated_text = re.sub(
                _dup_pattern,
                r'\1\3',
                translated_text,
                flags=re.IGNORECASE
            )
            logger.debug("Applied FIX 10d (word deduplication after closing tag)")

        logger.debug(f"non_translatable map: {non_translatable}")
        logger.debug(f"auto_wrap_tags: {auto_wrap_tags}")

        if not translated_text.strip():
            logger.warning(f"Empty translation for element {para.get('id', 'UNKNOWN')}")
            return

        had_drop_cap = False
        drop_cap_nsmap = None
        drop_cap_class = 'first-letter'
        for _child in element.iter():
            if _child is element:
                continue
            _tag = etree.QName(_child).localname if not hasattr(_child.tag, '__call__') else ''
            if _tag == 'span':
                _class = _child.get('class', '')
                if 'first-letter' in _class.split():
                    had_drop_cap = True
                    drop_cap_nsmap = _child.nsmap
                    drop_cap_class = _class
                    logger.debug(f"[drop_cap_track] Detected first-letter span in original element")
                    break

        auto_wrap_tag_count = len(auto_wrap_tags) if auto_wrap_tags else 0
        expected_inline_count = len(inline_formatting_map) - len(non_translatable) - auto_wrap_tag_count
        expected_inline_count = max(0, expected_inline_count)
        expected_reserve_count = len(reserve_elements)

        has_inline_placeholders = bool(re.search(r'</?p_\d{2}>', translated_text))
        has_reserve_placeholders = bool(re.search(r'<id_\d{2}>', translated_text))

        actual_inline_count = len(re.findall(r'<p_\d{2}>', translated_text))
        actual_reserve_count = len(re.findall(r'<id_\d{2}>', translated_text))

        logger.debug("Placeholder check:")
        logger.debug(f"  Expected inline (excl. auto_wrap={auto_wrap_tag_count}): {expected_inline_count}, found: {actual_inline_count}")
        logger.debug(f"  Expected reserve: {expected_reserve_count}, found: {actual_reserve_count}")

        placeholders_missing = False

        if expected_inline_count > 0 and not has_inline_placeholders:
            logger.warning(f"⚠️ Missing all inline placeholders: expected {expected_inline_count}")
            placeholders_missing = True

        if expected_reserve_count > 0 and not has_reserve_placeholders:
            logger.warning(f"⚠️ Missing all reserve placeholders: expected {expected_reserve_count}")
            placeholders_missing = True

        if actual_inline_count < expected_inline_count:
            logger.warning(f"⚠️ Some inline placeholders missing: {expected_inline_count} → {actual_inline_count}")
            placeholders_missing = True

        if actual_reserve_count < expected_reserve_count:
            logger.warning(f"⚠️ Some reserve placeholders missing: {expected_reserve_count} → {actual_reserve_count}")
            placeholders_missing = True

        if placeholders_missing:
            logger.warning(f"Translation text preview: {repr(translated_text[:150])}")
            logger.warning("🔧 FALLBACK: Attempting text-only replacement...")
            self._fallback_text_only_replacement(element, translated_text)
            if had_drop_cap:
                self._fix_drop_cap_in_element(element)
            logger.debug("Fallback replacement complete")
            return

        logger.debug("All placeholders present - proceeding with full rebuild")
        full_text = translated_text

        for tag_info in reversed(auto_wrap_tags):
            opening = tag_info.get('opening', '')
            closing = tag_info.get('closing', '')
            full_text = opening + full_text + closing
            logger.debug(f"Restored auto-wrap: {opening} ... {closing}")

        if prefix_tags:
            full_text = " ".join(prefix_tags) + " " + full_text
            logger.debug(f"Added prefix: {' '.join(prefix_tags)}")

        if suffix_tags:
            full_text = full_text + " " + " ".join(suffix_tags)
            logger.debug(f"Added suffix: {' '.join(suffix_tags)}")

        for tag_id in sorted(non_translatable.keys()):
            info = non_translatable[tag_id]
            marker = f'<nt_{int(tag_id):02d}/>'
            original = info.get('full_match', '')
            if marker in full_text:
                full_text = full_text.replace(marker, original)
                logger.debug(f"Restored non-translatable: {marker} → {original}")
            else:
                full_text = full_text.rstrip() + original
                logger.warning(f"NT marker dropped by LLM, appended to end: {repr(original)}")

        element.text = ""
        for child in list(element):
            element.remove(child)
        logger.debug(f"Cleared existing children")

        self._rebuild_element_from_placeholders(
            element,
            full_text,
            inline_formatting_map,
            reserve_elements
        )

        if had_drop_cap:
            self._fix_drop_cap_in_element(element)
            has_drop_cap_now = any(
                'first-letter' in (c.get('class', '') or '').split()
                for c in element.iter()
                if c is not element and not hasattr(c.tag, '__call__')
            )
            if not has_drop_cap_now:
                logger.debug("[drop_cap_restore] Span first-letter zgubiony przy rebuild – odtwarzam")
                self._create_drop_cap_span(element, drop_cap_nsmap, drop_cap_class)

        logger.debug("✓ Inline restoration complete")

    def _fallback_text_only_replacement(self, element, translated_text):
        logger.debug(f"\n=== FALLBACK: Text-only replacement ===")
        logger.debug(f"Original text: {repr(translated_text[:100])}")

        clean_translated = re.sub(r'</?p_\d{2}>', '', translated_text)
        clean_translated = re.sub(r'<id_\d{2}>', '', clean_translated)
        clean_translated = re.sub(r'</id_\d{2}>', '', clean_translated)
        clean_translated = re.sub(r'<nt_\d{2}/>', '', clean_translated)
        clean_translated = re.sub(r'  +', ' ', clean_translated).strip()

        logger.debug(f"Cleaned text:  {repr(clean_translated[:100])}")

        original_text_nodes = element.xpath('.//text()')

        if not original_text_nodes:
            element.text = clean_translated
            logger.debug("No text nodes - set element.text directly")
            return

        SKIP_IN_FALLBACK = {'sub', 'sup', 'abbr', 'code', 'kbd', 'var'}

        content_text_nodes = [
            n for n in original_text_nodes
            if str(n).strip()
            and etree.QName(n.getparent()).localname not in SKIP_IN_FALLBACK
        ]
        whitespace_text_nodes = [n for n in original_text_nodes if not str(n).strip()]

        logger.debug(f"Content nodes: {len(content_text_nodes)}, whitespace-only: {len(whitespace_text_nodes)}")

        if not content_text_nodes:
            element.text = clean_translated
            logger.debug("No content nodes - set element.text directly")
            return

        for text_node in whitespace_text_nodes:
            parent = text_node.getparent()
            if parent is None:
                continue
            if parent.text == text_node:
                parent.text = str(text_node)
            else:
                parent.tail = str(text_node)

        total_original_len = sum(len(str(node).strip()) for node in content_text_nodes)

        if total_original_len == 0:
            first = content_text_nodes[0]
            parent = first.getparent()
            if parent is not None:
                if parent.text == first:
                    parent.text = clean_translated
                else:
                    parent.tail = clean_translated
            return

        logger.debug(
            f"Distributing {len(clean_translated)} chars "
            f"across {len(content_text_nodes)} content nodes"
        )

        remaining_text = clean_translated

        for idx, text_node in enumerate(content_text_nodes):
            parent = text_node.getparent()

            if parent is None:
                continue

            if parent.text == text_node:
                location = 'text'
            else:
                location = 'tail'

            if idx == len(content_text_nodes) - 1:
                chunk = remaining_text
            else:
                original_stripped_len = len(str(text_node).strip())
                ratio = original_stripped_len / total_original_len
                target_len = int(len(clean_translated) * ratio)

                if target_len < len(remaining_text):
                    space_pos = remaining_text.rfind(
                        ' ',
                        max(0, target_len - 20),
                        min(len(remaining_text), target_len + 20)
                    )

                    if space_pos != -1 and space_pos > 0:
                        chunk = remaining_text[:space_pos + 1]
                        remaining_text = remaining_text[space_pos + 1:]
                    else:
                        chunk = remaining_text[:target_len]
                        remaining_text = remaining_text[target_len:]
                else:
                    chunk = remaining_text
                    remaining_text = ''

            if location == 'text':
                parent.text = chunk
            else:
                parent.tail = chunk

            logger.debug(f"  Content node {idx} ({location}): {repr(chunk[:50])}")

        logger.debug("✓ Fallback replacement complete - structure preserved")

    def _create_clean_translation_element(self, original_element, element_name, translation, root, is_legacy_mode=False):
        logger.debug(f"\n{'='*60}")
        logger.debug(f"CREATING TRANSLATION ELEMENT")
        logger.debug(f"Element: <{element_name}> id={original_element.get('id')}")
        logger.debug(f"Translation: {repr(translation[:100])}")
        logger.debug(f"Mode: {'LEGACY' if is_legacy_mode else 'INLINE'}")

        nsmap = original_element.nsmap
        default_ns = nsmap.get(None, self.ns['x'])

        logger.debug(f"\n>>> _create_clean_translation_element called")
        logger.debug(f"    translation: {repr(translation[:100])}")

        link_info_original = self._analyze_link_structure_on_element(original_element)

        if link_info_original:
            logger.debug(f"    Link found in ORIGINAL - will preserve structure")
            logger.debug(f"    Link ratio: {link_info_original['link_text_ratio']:.1%}")

        new_element = copy.deepcopy(original_element)

        links_after_copy = new_element.xpath('.//x:a', namespaces=self.ns)
        logger.debug(f"    Links after deepcopy: {len(links_after_copy)}")

        link_info = None
        if link_info_original and links_after_copy:
            link_info = {
                'should_preserve': link_info_original['should_preserve'],
                'link_element': links_after_copy[0],
                'link_text_ratio': link_info_original['link_text_ratio'],
                'original_text': link_info_original['original_text'],
                'link_text': link_info_original['link_text']
            }
            logger.debug(f"    Link info mapped to COPIED element")

        original_id = original_element.get('id', '')
        if original_id:
            clean_id = original_id
            if clean_id.startswith('trans_'):
                clean_id = clean_id[len('trans_'):]
            new_element.set('id', f"trans_{clean_id}")

        new_element.set('dir', self.target_direction or 'auto')

        if self.translation_lang:
            new_element.set('lang', self.translation_lang)

        if self.translation_color:
            style = new_element.get('style', '')
            new_element.set('style', f"{style}; color:{self.translation_color}".strip('; '))

        has_html = bool(re.search(r'<[^>]+>', translation))
        logger.debug(f"    has_html: {has_html}")

        if has_html:
            logger.debug(f"    → Calling _insert_html_translation")
            self._insert_html_translation(new_element, translation, default_ns)
        else:
            logger.debug(f"    link_info: {link_info is not None}")
            if link_info:
                logger.debug(f"    should_preserve: {link_info.get('should_preserve')}")
                logger.debug(f"    link_ratio: {link_info.get('link_text_ratio'):.1%}")

            if link_info and link_info['should_preserve']:
                logger.debug(f"    → Calling _map_with_link_semantics")
                self._map_with_link_semantics(new_element, translation, link_info)
            else:
                logger.debug(f"    → Calling _map_text_to_structure_simple")
                self._map_text_to_structure_simple(new_element, translation, is_legacy_mode=is_legacy_mode)

        links_final = new_element.xpath('.//x:a', namespaces=self.ns)
        logger.debug(f"    Links in final element: {len(links_final)}")

        return new_element

    def _insert_html_translation(self, element, translation, default_ns):
        logger.debug(f"\n=== INSERT HTML TRANSLATION ===")
        logger.debug(f"Translation: {repr(translation[:100])}")

        has_placeholders = bool(re.search(r'</?p_\d{2}>', translation))
        logger.debug(f"Has placeholders: {has_placeholders}")

        if has_placeholders:
            logger.debug("Using placeholder-based reconstruction")

            element.text = None
            for child in list(element):
                element.remove(child)

            try:
                temp_html = f'<div xmlns="{default_ns}">{translation}</div>'
                temp_tree = etree.fromstring(temp_html.encode('utf-8'))

                if temp_tree.text:
                    element.text = temp_tree.text

                for child in temp_tree:
                    element.append(child)

                logger.debug("HTML parsed and inserted successfully")

            except etree.XMLSyntaxError as e:
                logger.error(f"Failed to parse HTML with placeholders: {e}")
                element.text = re.sub(r'<[^>]+>', ' ', translation).strip()

            return

        has_anchor_tags = bool(re.search(r'<a\s[^>]*href=', translation))
        if has_anchor_tags:
            logger.debug("Translation contains <a href=...> tags — performing full HTML replacement")
            element.text = None
            for child in list(element):
                element.remove(child)

            def _try_parse_full(html_str, ns):
                try:
                    wrapped = f'<div xmlns="{ns}">{html_str}</div>'
                    return etree.fromstring(wrapped.encode('utf-8'))
                except etree.XMLSyntaxError:
                    return None

            temp_tree = _try_parse_full(translation, default_ns)
            if temp_tree is None:
                clean_trans = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', '', translation)
                temp_tree = _try_parse_full(clean_trans, default_ns)
                if temp_tree is not None:
                    logger.debug("Parsed successfully after xmlns strip")

            if temp_tree is not None:
                if temp_tree.text:
                    element.text = temp_tree.text
                for child in temp_tree:
                    element.append(child)
                logger.debug("Full HTML replacement complete")
            else:
                logger.warning("Failed to parse full HTML — falling back to plain text")
                element.text = re.sub(r'<[^>]+>', ' ', translation).strip()
            return

        links = element.xpath('.//x:a', namespaces=self.ns)

        if links:
            logger.debug(f"Element has {len(links)} link(s) - preserving structure")

            link_elem = links[0]

            def _insert_into_link(target_link, trans_text, ns):
                try:
                    tmp = f'<div xmlns="{ns}">{trans_text}</div>'
                    tree = etree.fromstring(tmp.encode('utf-8'))
                    target_link.text = None
                    for c in list(target_link):
                        target_link.remove(c)
                    if tree.text:
                        target_link.text = tree.text
                    for c in tree:
                        target_link.append(c)
                    return True
                except etree.XMLSyntaxError:
                    return False

            if not _insert_into_link(link_elem, translation, default_ns):
                clean_trans = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', '', translation)
                if not _insert_into_link(link_elem, clean_trans, default_ns):
                    link_elem.text = re.sub(r'<[^>]+>', ' ', translation).strip()
                    logger.warning(f"Inserted translation as plain text into link")
                else:
                    logger.debug(f"Inserted HTML translation into link (after xmlns strip)")
            else:
                logger.debug(f"Inserted HTML translation into link")

        else:
            logger.debug("No links found - replacing content entirely")
            element.text = None
            element.tail = element.tail
            for child in list(element):
                element.remove(child)

            def _try_parse(html_str, ns):
                try:
                    wrapped = f'<div xmlns="{ns}">{html_str}</div>'
                    return etree.fromstring(wrapped.encode('utf-8'))
                except etree.XMLSyntaxError:
                    return None

            temp_tree = _try_parse(translation, default_ns)

            if temp_tree is None:
                clean_trans = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', '', translation)
                temp_tree = _try_parse(clean_trans, default_ns)
                if temp_tree is not None:
                    logger.debug("Parsed successfully after xmlns strip")

            if temp_tree is not None:
                if temp_tree.text:
                    element.text = temp_tree.text
                for child in temp_tree:
                    element.append(child)
            else:
                logger.warning(
                    f"Failed to parse HTML translation even after xmlns strip – "
                    f"falling back to plain text"
                )
                plain = re.sub(r'<[^>]+>', ' ', translation)
                plain = re.sub(r'\s+', ' ', plain).strip()
                element.text = plain

    def _map_text_to_structure_simple(self, element, translation, is_legacy_mode=False):
        if is_legacy_mode:
            logger.debug("=== ROUTING TO LEGACY MODE ===")
            self._map_text_to_structure_legacy(element, translation)
        else:
            logger.debug("=== ROUTING TO INLINE MODE ===")
            self._map_text_to_structure_inline(element, translation)

    def _map_text_to_structure_inline(self, element, translation):
        logger.debug(f"\n=== INLINE TEXT MAPPING ===")
        logger.debug(f"Translation: {repr(translation)}")

        first_letter_span = None

        for child in element.iter():
            if child is element:
                continue
            tag = etree.QName(child).localname if not hasattr(child.tag, '__call__') else 'unknown'
            if tag == 'span':
                class_attr = child.get('class', '')
                if 'first-letter' in class_attr.split():
                    first_letter_span = child
                    logger.debug("Found drop cap with class='first-letter'")
                    break

        if first_letter_span is None:
            children = list(element)
            if len(children) > 0:
                first_child = children[0]
                tag = etree.QName(first_child).localname
                if tag == 'span':
                    child_text = (first_child.text or '').strip()
                    child_class = first_child.get('class')
                    if len(child_text) == 1 and child_text.isalpha() and not child_class:
                        first_letter_span = first_child
                        logger.debug(f"✓ Detected drop cap without class: '{child_text}'")

        original_texts = self._collect_text_nodes(element)

        if not original_texts:
            element.text = translation
            return

        logger.debug(f"\nOriginal text structure ({len(original_texts)} nodes):")
        for idx, (el, loc, text, stripped, tag) in enumerate(original_texts):
            logger.debug(f"  [{idx}] <{tag}>.{loc} = {repr(text[:50])}")

        content_nodes = []
        for idx, (el, loc, text, stripped, tag) in enumerate(original_texts):
            if not stripped:
                continue

            if loc == 'text' and tag == 'span':
                epub_type = el.get('{http://www.idpf.org/2007/ops}type')
                if epub_type == 'pagebreak':
                    logger.debug(f"  [{idx}] SKIP - pagebreak span .text")
                    continue

            content_nodes.append((idx, el, loc, text, stripped, tag))

        logger.debug(f"\nContent nodes (after filtering): {len(content_nodes)}")

        if len(content_nodes) == 1:
            logger.debug("CASE 1: Single content node - direct mapping")

            idx, el, loc, text, stripped, tag = content_nodes[0]

            if first_letter_span is not None and el is first_letter_span and loc == 'text':
                translation_stripped = translation.lstrip()
                if translation_stripped:
                    first_char = translation_stripped[0]
                    remaining = translation_stripped[1:].lstrip()

                    for node, location, _, _, _ in original_texts:
                        if node is first_letter_span and location == 'text':
                            continue
                        self._set_text_at_location(node, location, '')

                    first_letter_span.text = first_char
                    first_letter_span.tail = remaining
                    logger.debug(f"Drop cap: '{first_char}' → tail: {repr(remaining[:50])}")
                else:
                    element.text = translation
                return

            for node, location, _, _, _ in original_texts:
                self._set_text_at_location(node, location, '')

            self._set_text_at_location(el, loc, translation)
            logger.debug(f"Mapped to node [{idx}] <{tag}>.{loc}")
            return

        formatting_elements = self._find_formatting_elements(element)
        logger.debug(f"\nFormatting elements found: {len(formatting_elements)}")

        if len(formatting_elements) == 1:
            fmt_elem = formatting_elements[0]
            fmt_tag = etree.QName(fmt_elem).localname

            logger.debug(f"Checking if all content is in <{fmt_tag}>")

            nodes_in_fmt = []
            nodes_outside_fmt = []

            for idx, el, loc, text, stripped, tag in content_nodes:
                if first_letter_span is not None and el is first_letter_span and loc == 'text':
                    continue

                if el == fmt_elem and loc == 'text':
                    nodes_in_fmt.append((idx, el, loc, text, stripped, tag))
                elif loc == 'tail' and el == fmt_elem:
                    nodes_outside_fmt.append((idx, el, loc, text, stripped, tag))
                else:
                    is_descendant = False
                    parent = el
                    while parent is not None:
                        if parent == fmt_elem:
                            is_descendant = True
                            break
                        parent = parent.getparent()

                    if is_descendant:
                        nodes_in_fmt.append((idx, el, loc, text, stripped, tag))
                    else:
                        nodes_outside_fmt.append((idx, el, loc, text, stripped, tag))

            if len(nodes_in_fmt) > 0:
                total_inside = sum(len(s) for _, _, _, _, s, _ in nodes_in_fmt)
                total_outside = sum(len(s) for _, _, _, _, s, _ in nodes_outside_fmt)

                logger.debug(f"  Chars inside: {total_inside}, outside: {total_outside}")

                if total_inside >= total_outside * 4:
                    logger.debug("CASE 2: Content mostly inside formatting")

                    working_translation = translation
                    if first_letter_span is not None:
                        translation_stripped = translation.lstrip()
                        if translation_stripped:
                            first_char = translation_stripped[0]
                            first_letter_span.text = first_char
                            working_translation = translation_stripped[1:].lstrip()
                            logger.debug(f"Drop cap: '{first_char}' → working: {repr(working_translation[:50])}")

                    for node, location, _, _, _ in original_texts:
                        if first_letter_span is not None and node is first_letter_span and location == 'text':
                            continue
                        self._set_text_at_location(node, location, '')

                    idx, el, loc, text, stripped, tag = nodes_in_fmt[0]
                    self._set_text_at_location(el, loc, working_translation)

                    if nodes_outside_fmt:
                        for idx_out, el_out, loc_out, text_out, stripped_out, tag_out in nodes_outside_fmt:
                            if el_out == fmt_elem and loc_out == 'tail':
                                if len(stripped_out) <= 3 and stripped_out in '.!?,;:':
                                    if not working_translation.rstrip().endswith(stripped_out):
                                        fmt_elem.tail = stripped_out
                                    else:
                                        fmt_elem.tail = ''
                                else:
                                    fmt_elem.tail = ''
                                break
                    else:
                        fmt_elem.tail = ''

                    logger.debug("✓ CASE 2 complete")
                    return

        smallcaps_spans = [el for el in formatting_elements if el.get('class') == 'smallcaps']
        italic_elements = [el for el in formatting_elements if etree.QName(el).localname == 'i']
        last_word_spans = [el for el in formatting_elements if el.get('class') == 'last-word']
        underline_spans = [el for el in formatting_elements if el.get('class') == 'underline']

        has_nested_structure = False
        if underline_spans and last_word_spans:
            for underline in underline_spans:
                for last_word in last_word_spans:
                    parent = last_word.getparent()
                    while parent is not None:
                        if parent == underline:
                            has_nested_structure = True
                            break
                        parent = parent.getparent()
                    if has_nested_structure:
                        break
                if has_nested_structure:
                    break

        if smallcaps_spans or italic_elements or last_word_spans:
            logger.debug(f"CASE 2b: Found {len(last_word_spans)} last-word, {len(smallcaps_spans)} smallcaps, {len(italic_elements)} italic")

            last_content_elem = None
            for idx, el, loc, text, stripped, tag in content_nodes:
                if stripped:
                    if first_letter_span is not None and el is first_letter_span and loc == 'text':
                        continue
                    last_content_elem = el

            target_element = None
            original_word_count = 0

            if last_content_elem in last_word_spans:
                target_element = last_content_elem
                for idx, el, loc, text, stripped, tag in content_nodes:
                    if el == last_content_elem and loc == 'text':
                        original_word_count = len(stripped.split())
                        break
                logger.debug(f"  → Target: <span.last-word> ({original_word_count} words)")
            elif last_content_elem in italic_elements:
                target_element = last_content_elem
                for idx, el, loc, text, stripped, tag in content_nodes:
                    if el == last_content_elem and loc == 'text':
                        original_word_count = len(stripped.split())
                        break
                logger.debug(f"  → Target: <i> ({original_word_count} words)")
            elif last_content_elem in smallcaps_spans:
                target_element = last_content_elem
                for idx, el, loc, text, stripped, tag in content_nodes:
                    if el == last_content_elem and loc == 'text':
                        original_word_count = len(stripped.split())
                        break
                logger.debug(f"  → Target: <span.smallcaps> ({original_word_count} words)")
            elif last_word_spans:
                target_element = last_word_spans[-1]
                original_word_count = 1
            elif italic_elements:
                target_element = italic_elements[-1]
                original_word_count = 1
            elif smallcaps_spans:
                target_element = smallcaps_spans[-1]
                original_word_count = 1

            if target_element is not None:
                working_translation = translation

                if first_letter_span is not None:
                    translation_stripped = translation.lstrip()
                    if translation_stripped:
                        first_char = translation_stripped[0]
                        first_letter_span.text = first_char
                        working_translation = translation_stripped[1:].lstrip()
                        logger.debug(f"Drop cap updated: '{first_char}' → working: {repr(working_translation[:50])}")

                for node, location, _, _, _ in original_texts:
                    if first_letter_span is not None and node is first_letter_span and location == 'text':
                        continue
                    self._set_text_at_location(node, location, '')

                words = working_translation.split()

                if len(words) >= 2:
                    target_word_count = min(original_word_count if original_word_count > 0 else 1, len(words))
                    target_text = ' '.join(words[-target_word_count:])
                    before_text = ' '.join(words[:-target_word_count])

                    logger.debug(f"  Before: '{before_text}'")
                    logger.debug(f"  Target: '{target_text}'")

                    before_elem = None
                    before_loc = None

                    if has_nested_structure and underline_spans:
                        before_elem = underline_spans[0]
                        before_loc = 'text'
                        logger.debug(f"  Using underline span for 'before'")
                    else:
                        parent = target_element.getparent()
                        while parent is not None and parent != element:
                            for idx, el, loc, text, stripped, tag in content_nodes:
                                if el == parent and loc == 'text':
                                    before_elem = parent
                                    before_loc = 'text'
                                    logger.debug(f"  Found parent: <{etree.QName(parent).localname}>.text")
                                    break
                            if before_elem is not None:
                                break
                            parent = parent.getparent()

                    if before_elem is None:
                        for idx, el, loc, text, stripped, tag in content_nodes:
                            if first_letter_span is not None and el is first_letter_span and loc == 'text':
                                continue
                            if loc == 'text' and tag == 'span':
                                epub_type = el.get('{http://www.idpf.org/2007/ops}type')
                                if epub_type == 'pagebreak':
                                    continue
                            before_elem = el
                            before_loc = loc
                            logger.debug(f"  Using first non-drop-cap content node: <{tag}>.{loc}")
                            break

                    if before_elem is not None and before_loc:
                        self._set_text_at_location(before_elem, before_loc, before_text + ' ' if before_text else '')

                    target_element.text = target_text

                    for fmt_el in formatting_elements:
                        if fmt_el == target_element:
                            continue

                        is_parent = False
                        p = target_element.getparent()
                        while p is not None:
                            if p == fmt_el:
                                is_parent = True
                                break
                            p = p.getparent()

                        if is_parent:
                            continue

                        if not fmt_el.text and not fmt_el.tail and len(fmt_el) == 0:
                            parent = fmt_el.getparent()
                            if parent is not None:
                                parent.remove(fmt_el)

                    return
                else:
                    target_element.text = working_translation
                    return

        logger.debug("CASE 3: Multiple content nodes - proportional distribution (INLINE)")
        self._distribute_translation(element, original_texts, translation, is_legacy_mode=False)

    def _map_text_to_structure_legacy(self, element, translation):
        logger.debug(f"\n=== LEGACY TEXT MAPPING ===")
        logger.debug(f"Translation: {repr(translation)}")

        first_letter_span = None

        for child in element.iter():
            if child is element:
                continue
            tag = etree.QName(child).localname if not hasattr(child.tag, '__call__') else 'unknown'
            if tag == 'span':
                class_attr = child.get('class', '')
                if 'first-letter' in class_attr.split():
                    first_letter_span = child
                    logger.debug("Found drop cap with class='first-letter'")
                    break

        if first_letter_span is None:
            children = list(element)
            if len(children) > 0:
                first_child = children[0]
                tag = etree.QName(first_child).localname
                if tag == 'span':
                    child_text = (first_child.text or '').strip()
                    child_class = first_child.get('class')
                    if len(child_text) == 1 and child_text.isalpha() and not child_class:
                        first_letter_span = first_child
                        logger.debug(f"✓ Detected drop cap without class: '{child_text}'")

        original_texts = self._collect_text_nodes(element)

        if not original_texts:
            element.text = translation
            return

        logger.debug(f"\nOriginal text structure ({len(original_texts)} nodes):")
        for idx, (el, loc, text, stripped, tag) in enumerate(original_texts):
            logger.debug(f"  [{idx}] <{tag}>.{loc} = {repr(text[:50])}")

        content_nodes = []
        for idx, (el, loc, text, stripped, tag) in enumerate(original_texts):
            if not stripped:
                continue

            if loc == 'text' and tag == 'span':
                epub_type = el.get('{http://www.idpf.org/2007/ops}type')
                if epub_type == 'pagebreak':
                    logger.debug(f"  [{idx}] SKIP - pagebreak span .text")
                    continue

            content_nodes.append((idx, el, loc, text, stripped, tag))

        logger.debug(f"\nContent nodes (after filtering): {len(content_nodes)}")

        if len(content_nodes) == 1:
            logger.debug("CASE 1: Single content node - direct mapping")

            idx, el, loc, text, stripped, tag = content_nodes[0]

            if first_letter_span is not None and el is first_letter_span and loc == 'text':
                translation_stripped = translation.lstrip()
                if translation_stripped:
                    first_char = translation_stripped[0]
                    remaining = translation_stripped[1:].lstrip()

                    for node, location, _, _, _ in original_texts:
                        if node is first_letter_span and location == 'text':
                            continue
                        self._set_text_at_location(node, location, '')

                    first_letter_span.text = first_char
                    first_letter_span.tail = remaining
                    logger.debug(f"Drop cap: '{first_char}' → tail: {repr(remaining[:50])}")
                else:
                    element.text = translation
                return

            if el is element and loc == 'text':
                element.text = translation
                for node, location, _, _, _ in original_texts:
                    if node is element and location == 'text':
                        continue
                    self._set_text_at_location(node, location, '')
                logger.debug(f"Mapped to element.text: {repr(translation[:50])}")
            else:
                element.text = ''
                self._set_text_at_location(el, loc, translation)
                for node, location, _, _, _ in original_texts:
                    if node is el and location == loc:
                        continue
                    self._set_text_at_location(node, location, '')
                logger.debug(f"Mapped to <{tag}>.{loc}: {repr(translation[:50])}")
            return

        formatting_elements = self._find_formatting_elements(element)
        logger.debug(f"\nFormatting elements found: {len(formatting_elements)}")

        if len(formatting_elements) >= 1:
            logger.debug(f"Checking if formatting elements cover 80%+ of content")

            nodes_in_fmt = []
            nodes_outside_fmt = []

            for idx, el, loc, text, stripped, tag in content_nodes:
                if first_letter_span is not None and el is first_letter_span and loc == 'text':
                    continue

                is_inside_fmt = False
                parent_fmt_elem = None

                for fmt_elem in formatting_elements:
                    if el == fmt_elem and loc == 'text':
                        is_inside_fmt = True
                        parent_fmt_elem = fmt_elem
                        break
                    elif loc == 'tail' and el == fmt_elem:
                        break
                    else:
                        parent = el
                        while parent is not None:
                            if parent == fmt_elem:
                                is_inside_fmt = True
                                parent_fmt_elem = fmt_elem
                                break
                            parent = parent.getparent()
                        if is_inside_fmt:
                            break

                if is_inside_fmt:
                    nodes_in_fmt.append((idx, el, loc, text, stripped, tag, parent_fmt_elem))
                else:
                    nodes_outside_fmt.append((idx, el, loc, text, stripped, tag))

            if len(nodes_in_fmt) > 0:
                total_inside = sum(len(s) for _, _, _, _, s, _, _ in nodes_in_fmt)
                total_outside = sum(len(s) for _, _, _, _, s, _ in nodes_outside_fmt)

                logger.debug(f"  Chars inside formatting: {total_inside}, outside: {total_outside}")

                if total_inside >= total_outside * 4:
                    logger.debug(f"CASE 2: Content mostly inside {len(formatting_elements)} formatting element(s)")

                    all_identical = self._are_formatting_elements_identical(formatting_elements)
                    logger.debug(f"  All formatting elements identical: {all_identical}")

                    if not all_identical:
                        logger.debug("  → Elements NOT identical (mixed styles) - falling through to FALLBACK")

                    else:
                        logger.debug("  → Elements identical - preserving structure")

                        working_translation = translation
                        if first_letter_span is not None:
                            translation_stripped = translation.lstrip()
                            if translation_stripped:
                                first_char = translation_stripped[0]
                                first_letter_span.text = first_char
                                working_translation = translation_stripped[1:].lstrip()
                                logger.debug(f"Drop cap: '{first_char}' → working: {repr(working_translation[:50])}")

                        for node, location, _, _, _ in original_texts:
                            if first_letter_span is not None and node is first_letter_span and location == 'text':
                                continue
                            self._set_text_at_location(node, location, '')

                        if len(formatting_elements) == 1:
                            idx, el, loc, text, stripped, tag, fmt_elem = nodes_in_fmt[0]
                            self._set_text_at_location(el, loc, working_translation)
                        else:
                            idx, el, loc, text, stripped, tag, fmt_elem = nodes_in_fmt[0]
                            self._set_text_at_location(el, loc, working_translation)

                            logger.debug(f"  Merged {len(formatting_elements)} identical elements into first")

                            for i in range(1, len(formatting_elements)):
                                duplicate_elem = formatting_elements[i]
                                parent = duplicate_elem.getparent()
                                if parent is not None:
                                    parent.remove(duplicate_elem)
                                    logger.debug(f"  Removed duplicate element #{i}")

                        if nodes_outside_fmt:
                            for idx_out, el_out, loc_out, text_out, stripped_out, tag_out in nodes_outside_fmt:
                                if loc_out == 'tail' and len(stripped_out) <= 3 and stripped_out in '.!?,;:':
                                    if not working_translation.rstrip().endswith(stripped_out):
                                        el_out.tail = stripped_out
                                    else:
                                        el_out.tail = ''
                                else:
                                    self._set_text_at_location(el_out, loc_out, '')

                        logger.debug("✓ CASE 2 complete")
                        return

        logger.debug("FALLBACK: Flattening structure - removing all inline formatting")

        logger.debug("Clearing all element content (text + children)")

        working_translation = translation
        drop_cap_char = None

        if first_letter_span is not None:
            translation_stripped = translation.lstrip()
            if translation_stripped:
                drop_cap_char = translation_stripped[0]
                working_translation = translation_stripped[1:].lstrip()
                logger.debug(f"Drop cap: '{drop_cap_char}' → remaining: {repr(working_translation[:50])}")

        element.text = ""

        children_to_remove = []
        for child in list(element):
            if first_letter_span is not None and child is first_letter_span:
                logger.debug(f"  Preserving drop cap span")

                for subchild in list(child):
                    child.remove(subchild)
                child.tail = ''
                continue

            children_to_remove.append(child)

        removed_count = 0
        for child in children_to_remove:
            tag = etree.QName(child).localname if not hasattr(child.tag, '__call__') else 'unknown'
            logger.debug(f"  Removing <{tag}> element")
            element.remove(child)
            removed_count += 1

        logger.debug(f"  Removed {removed_count} child element(s)")

        if first_letter_span is not None and drop_cap_char is not None:
            first_letter_span.text = drop_cap_char
            first_letter_span.tail = working_translation
            logger.debug(f"Inserted translation: drop_cap.text='{drop_cap_char}', drop_cap.tail={repr(working_translation[:50])}")
        else:
            element.text = working_translation
            logger.debug(f"Inserted translation in element.text")

        logger.debug("✓ Flattening complete")

    def _are_formatting_elements_identical(self, elements):
        if len(elements) <= 1:
            return True

        first_elem = elements[0]
        first_tag = etree.QName(first_elem).localname
        first_attribs = dict(first_elem.attrib)

        for elem in elements[1:]:
            tag = etree.QName(elem).localname
            attribs = dict(elem.attrib)

            if tag != first_tag:
                logger.debug(f"  Different tags: {first_tag} vs {tag}")
                return False

            if attribs != first_attribs:
                logger.debug(f"  Different attributes: {first_attribs} vs {attribs}")
                return False

        return True

    def _find_formatting_elements(self, element):
        formatting_tags = {'i', 'b', 'em', 'strong', 'u', 'sup', 'sub', 'small'}
        formatting_elements = []

        drop_cap_span = None
        children = list(element)

        if len(children) > 0:
            first_child = children[0]

            tag = etree.QName(first_child).localname

            if tag == 'span':
                child_text = (first_child.text or '').strip()
                child_class = first_child.get('class')

                if len(child_text) == 1 and child_text.isalpha() and not child_class:
                    drop_cap_span = first_child
                    logger.debug(f"✓ Detected drop cap span: '{child_text}' (will be excluded)")

        for child in element.iter():
            if child is element:
                continue

            if drop_cap_span is not None and child is drop_cap_span:
                logger.debug(f"✓ Skipping drop cap span from formatting_elements")
                continue

            tag = etree.QName(child).localname if not hasattr(child.tag, '__call__') else 'unknown'

            if tag in formatting_tags:
                formatting_elements.append(child)

            elif tag == 'span':
                epub_type = child.get('{http://www.idpf.org/2007/ops}type')
                if epub_type == 'pagebreak':
                    continue

                class_attr = child.get('class')
                if class_attr:
                    formatting_elements.append(child)

        return formatting_elements

    def _analyze_link_structure_on_element(self, element):
        links = element.xpath('.//x:a', namespaces=self.ns)

        if not links:
            return None

        total_text_nodes = element.xpath('.//text()')
        total_text = ''.join(str(t) for t in total_text_nodes).strip()
        total_len = len(total_text)

        if total_len == 0:
            return None

        link = links[0]
        link_text_nodes = link.xpath('.//text()')
        link_text = ''.join(str(t) for t in link_text_nodes).strip()
        link_len = len(link_text)

        ratio = link_len / total_len if total_len > 0 else 0

        should_preserve = True

        logger.debug(f"Link analysis: total={total_len}, link={link_len}, ratio={ratio:.1%}, preserve={should_preserve}")
        logger.debug(f"Original text: {repr(total_text)}")
        logger.debug(f"Link text: {repr(link_text)}")

        return {
            'should_preserve': should_preserve,
            'link_element': link,
            'link_text_ratio': ratio,
            'original_text': total_text,
            'link_text': link_text
        }

    def _map_with_link_semantics(self, element, translation, link_info):
        all_spans = element.xpath('.//x:span', namespaces=self.ns)
        has_complex_formatting = False

        for span in all_spans:
            class_attr = span.get('class')
            if class_attr in ['last-word', 'underline', 'smallcaps']:
                has_complex_formatting = True
                break

        if has_complex_formatting:
            logger.debug("=== ROUTING TO INLINE LINK MODE ===")
            self._map_with_link_semantics_inline(element, translation, link_info)
        else:
            logger.debug("=== ROUTING TO LEGACY LINK MODE ===")
            self._map_with_link_semantics_legacy(element, translation, link_info)

    def _map_with_link_semantics_inline(self, element, translation, link_info):
        links = element.xpath('.//x:a', namespaces=self.ns)

        if not links:
            logger.warning("No link found - fallback to simple mapping")
            self._map_text_to_structure_inline(element, translation)
            return

        link_elem = links[0]
        link_ratio = link_info['link_text_ratio']

        logger.debug(f"\n=== INLINE LINK SEMANTICS MAPPING ===")
        logger.debug(f"Translation: {repr(translation)}")
        logger.debug(f"Link ratio: {link_ratio:.1%}")

        item_number_span = None
        element_number_span = None

        all_spans = element.xpath('.//x:span', namespaces=self.ns)

        for span in all_spans:
            class_attr = span.get('class')
            if class_attr == 'item-number':
                item_number_span = span
            elif class_attr == 'element-number':
                element_number_span = span

        prefix_match = re.match(
            r'^((?:.+?\s)?(?:\d+|[IVXLCDM]+)[\.:–—-]\s+)',
            translation,
            re.IGNORECASE
        )

        if prefix_match:
            prefix = prefix_match.group(1)
            remaining_translation = translation[len(prefix):]

            all_nodes = self._collect_text_nodes(element)
            for node, location, _, _, _ in all_nodes:
                self._set_text_at_location(node, location, '')

            if item_number_span is not None:
                item_number_span.text = prefix
                self._set_text_in_deepest_node(link_elem, remaining_translation)
                return

            if element_number_span is not None:
                element_number_span.text = prefix
                self._set_text_in_deepest_node(link_elem, remaining_translation)
                return

            link_parent = link_elem.getparent()
            if link_parent is None:
                element.text = translation
                return

            link_index = list(link_parent).index(link_elem)

            if link_index > 0:
                prev = link_parent[link_index - 1]
                prev.tail = (prev.tail or '') + prefix + ' '
            else:
                link_parent.text = (link_parent.text or '') + prefix + ' '

            self._set_text_in_deepest_node(link_elem, remaining_translation)
            return

        elif link_ratio > 0.8:
            all_nodes = self._collect_text_nodes(element)
            for node, location, _, _, _ in all_nodes:
                self._set_text_at_location(node, location, '')

            self._set_text_in_deepest_node(link_elem, translation)
            return

        else:
            trans_words = translation.split()
            total_words = len(trans_words)

            link_word_count = max(1, round(total_words * link_ratio))

            before_link_words = trans_words[:-link_word_count] if link_word_count < total_words else []
            link_words = trans_words[-link_word_count:]

            before_link_text = ' '.join(before_link_words)
            link_text = ' '.join(link_words)

            all_nodes = self._collect_text_nodes(element)
            for node, location, _, _, _ in all_nodes:
                self._set_text_at_location(node, location, '')

            link_parent = link_elem.getparent()
            if link_parent is None:
                element.text = translation
                return

            link_index = list(link_parent).index(link_elem)

            if before_link_text:
                if link_index > 0:
                    prev = link_parent[link_index - 1]
                    prev.tail = (prev.tail or '') + before_link_text + ' '
                else:
                    link_parent.text = (link_parent.text or '') + before_link_text + ' '

            self._set_text_in_deepest_node(link_elem, link_text)

    def _map_with_link_semantics_legacy(self, element, translation, link_info):
        links = element.xpath('.//x:a', namespaces=self.ns)

        if not links:
            logger.warning("No link found - fallback to simple mapping")
            self._map_text_to_structure_legacy(element, translation)
            return

        link_elem = links[0]
        link_ratio = link_info['link_text_ratio']

        logger.debug(f"\n=== LEGACY LINK SEMANTICS MAPPING ===")
        logger.debug(f"Translation: {repr(translation)}")
        logger.debug(f"Link ratio: {link_ratio:.1%}")

        item_number_span = None
        element_number_span = None

        all_spans = element.xpath('.//x:span', namespaces=self.ns)

        for span in all_spans:
            class_attr = span.get('class')
            if class_attr:
                if 'item-number' in class_attr:
                    item_number_span = span
                elif 'element-number' in class_attr:
                    element_number_span = span

        prefix_match = re.match(
            r'^((?:.+?\s)?(?:\d+|[IVXLCDM]+)[\.:–—-]\s+)',
            translation,
            re.IGNORECASE
        )

        original_text = link_info['original_text'].strip()
        link_text = link_info['link_text'].strip()
        number_is_outside_link = False

        orig_prefix_match = re.match(
            r'^((?:.+?\s)?(?:\d+|[IVXLCDM]+)[\.:–—-]\s+)',
            original_text,
            re.IGNORECASE
        )
        if orig_prefix_match:
            orig_prefix = orig_prefix_match.group(1).strip()

            if not link_text.startswith(orig_prefix):
                number_is_outside_link = True

        if prefix_match and (item_number_span is not None or element_number_span is not None or number_is_outside_link):
            prefix = prefix_match.group(1)
            remaining_translation = translation[len(prefix):]

            all_nodes = self._collect_text_nodes(element)
            for node, location, _, _, _ in all_nodes:
                self._set_text_at_location(node, location, '')

            if item_number_span is not None:
                item_number_span.text = prefix
                self._set_text_in_deepest_node(link_elem, remaining_translation)
                return

            if element_number_span is not None:
                element_number_span.text = prefix
                self._set_text_in_deepest_node(link_elem, remaining_translation)
                return

            link_parent = link_elem.getparent()
            if link_parent is None:
                element.text = translation
                return

            link_index = list(link_parent).index(link_elem)

            if link_index > 0:
                prev = link_parent[link_index - 1]
                prev.tail = (prev.tail or '') + prefix + ' '
            else:
                link_parent.text = (link_parent.text or '') + prefix + ' '

            self._set_text_in_deepest_node(link_elem, remaining_translation)
            return

        elif link_ratio > 0.8:
            all_nodes = self._collect_text_nodes(element)
            for node, location, _, _, _ in all_nodes:
                self._set_text_at_location(node, location, '')

            self._set_text_in_deepest_node(link_elem, translation)
            return

        else:
            trans_words = translation.split()
            total_words = len(trans_words)

            link_word_count = max(1, round(total_words * link_ratio))

            before_link_words = trans_words[:-link_word_count] if link_word_count < total_words else []
            link_words = trans_words[-link_word_count:]

            before_link_text = ' '.join(before_link_words)
            link_text = ' '.join(link_words)

            all_nodes = self._collect_text_nodes(element)
            for node, location, _, _, _ in all_nodes:
                self._set_text_at_location(node, location, '')

            link_parent = link_elem.getparent()
            if link_parent is None:
                element.text = translation
                return

            link_index = list(link_parent).index(link_elem)

            if before_link_text:
                span_after_link = None
                if link_index + 1 < len(link_parent):
                    next_elem = link_parent[link_index + 1]
                    next_tag = etree.QName(next_elem).localname if not hasattr(next_elem.tag, '__call__') else 'unknown'
                    if next_tag == 'span':
                        span_after_link = next_elem

                if span_after_link is not None:
                    span_class = span_after_link.get('class')
                    nsmap = link_parent.nsmap
                    default_ns = nsmap.get(None, self.ns['x'])

                    new_span = etree.Element(f"{{{default_ns}}}span", nsmap=nsmap)
                    if span_class:
                        new_span.set('class', span_class)
                    new_span.text = before_link_text + ' '
                    new_span.tail = ''

                    link_parent.insert(link_index, new_span)
                    logger.debug(f"Created <span class='{span_class}'> before link")
                else:
                    if link_index > 0:
                        prev = link_parent[link_index - 1]
                        prev.tail = (prev.tail or '') + before_link_text + ' '
                    else:
                        link_parent.text = (link_parent.text or '') + before_link_text + ' '

            self._set_text_in_deepest_node(link_elem, link_text)

    def _set_text_in_deepest_node(self, element, text):
        logger.debug(f"\n_set_text_in_deepest_node:")
        logger.debug(f"  Element: <{etree.QName(element).localname}>")
        logger.debug(f"  Text: {repr(text[:50])}")

        all_text_nodes = element.xpath('.//text()')

        if not all_text_nodes:
            logger.debug(f"  No text nodes found - setting element.text")
            element.text = text
            return

        first_text = all_text_nodes[0]
        parent = first_text.getparent()

        if parent is None:
            logger.debug(f"  First text node has no parent - setting element.text")
            element.text = text
            return

        if parent.text == first_text:
            location = 'text'
        else:
            location = 'tail'

        tag = etree.QName(parent).localname if not hasattr(parent.tag, '__call__') else 'unknown'

        logger.debug(f"  Setting <{tag}>.{location}")

        self._set_text_at_location(parent, location, text)

    def _collect_text_nodes(self, element):
        nodes = []

        for idx, text_node in enumerate(element.xpath('.//text()')):
            parent = text_node.getparent()

            if parent is None:
                continue

            tag = etree.QName(parent).localname if hasattr(parent.tag, '__call__') == False else 'unknown'
            text = str(text_node)
            stripped = text.strip()

            location = 'tail' if text_node.is_tail else 'text'

            nodes.append((parent, location, text, stripped, tag))
            logger.debug(f"  Node {idx}: <{tag}>.{location} = {repr(text[:50] if len(text) > 50 else text)}")

        return nodes

    def _set_text_at_location(self, element, location, text):
        if location == 'text':
            element.text = text
        else:
            element.tail = text

    def _distribute_translation(self, root_element, original_texts, translation, is_legacy_mode=False):
        logger.debug(f"\n=== DISTRIBUTING TRANSLATION ===")
        logger.debug(f"Translation: {repr(translation)}")
        logger.debug(f"Total nodes: {len(original_texts)}")
        logger.debug(f"Mode: {'LEGACY (anchor-based)' if is_legacy_mode else 'INLINE (proportional)'}")

        if is_legacy_mode:
            anchors = self._find_anchor_points(original_texts, translation)

            if len(anchors) >= 2:
                logger.debug(f"✓ Found {len(anchors)} anchors - using anchor-based distribution")

                self._distribute_translation_with_anchors(root_element, original_texts, translation, anchors)
                return
            else:
                logger.debug(f"⚠ Only {len(anchors)} anchor(s) found - falling back to proportional distribution")

        valid_nodes = [(el, loc, text, stripped, tag) for el, loc, text, stripped, tag in original_texts if text]

        if not valid_nodes:
            logger.debug("No valid nodes - putting in root")
            root_element.text = translation
            return

        content_nodes = [(el, loc, text, stripped, tag) for el, loc, text, stripped, tag in valid_nodes if stripped]

        logger.debug(f"Content nodes: {len(content_nodes)}")

        if not content_nodes:
            el, loc, _, _, _ = valid_nodes[0]
            self._set_text_at_location(el, loc, translation)
            for el, loc, _, _, _ in valid_nodes[1:]:
                self._set_text_at_location(el, loc, '')
            return

        boundary_info = self._analyze_element_boundaries(valid_nodes, root_element)

        logger.debug(f"\nElement boundaries detected: {len(boundary_info['boundaries'])}")
        for bound in boundary_info['boundaries']:
            logger.debug(f"  Boundary at node {bound['node_idx']}: {bound['type']}")

        whitespace_map = self._build_whitespace_map(valid_nodes)

        logger.debug("\nWhitespace map:")
        for idx, ws in whitespace_map.items():
            logger.debug(f"  Node {idx}: leading={ws['leading']}, trailing={ws['trailing']}")

        total_len = sum(len(stripped) for _, _, _, stripped, _ in content_nodes)

        if total_len == 0:
            el, loc, _, _, _ = valid_nodes[0]
            self._set_text_at_location(el, loc, translation)
            return

        remaining = translation.strip()
        processed_nodes = 0

        logger.debug("\nDistributing chunks:")

        for idx, (el, loc, orig_text, orig_stripped, tag) in enumerate(valid_nodes):
            if not orig_stripped:
                logger.debug(f"  Node {idx} <{tag}>.{loc}: SKIP (whitespace only)")
                self._set_text_at_location(el, loc, '')
                continue

            processed_nodes += 1

            is_boundary = any(b['node_idx'] == idx for b in boundary_info['boundaries'])

            ws_info = whitespace_map.get(idx, {'leading': False, 'trailing': False})
            should_have_leading = ws_info['leading']
            should_have_trailing = ws_info['trailing']

            if processed_nodes == len(content_nodes):
                chunk = remaining
                logger.debug(f"  Node {idx} <{tag}>.{loc}: LAST NODE")
                logger.debug(f"    Raw chunk: {repr(chunk)}")

                if should_have_leading and chunk and not chunk[0].isspace():
                    chunk = ' ' + chunk
                    logger.debug(f"    Added leading space")

                if should_have_trailing and chunk and not chunk[-1].isspace():
                    chunk = chunk + ' '
                    logger.debug(f"    Added trailing space")

                logger.debug(f"    Final: {repr(chunk)}")
                self._set_text_at_location(el, loc, chunk)
                break

            ratio = len(orig_stripped) / total_len
            target_len = int(len(remaining) * ratio)

            logger.debug(f"  Node {idx} <{tag}>.{loc}:")
            logger.debug(f"    Original: {repr(orig_text)}")
            logger.debug(f"    Ratio: {ratio:.2%}, Target len: {target_len}")
            logger.debug(f"    Is boundary: {is_boundary}")

            if is_boundary:
                chunk, new_remaining, space_was_removed = self._split_at_sentence_boundary(remaining, target_len)
                logger.debug(f"    BOUNDARY SPLIT: chunk={repr(chunk)}, remaining={repr(new_remaining)}")
            else:
                chunk, new_remaining, space_was_removed = self._split_at_natural_break(remaining, target_len)
                logger.debug(f"    Normal split: chunk={repr(chunk)}, remaining={repr(new_remaining)}, space_removed={space_was_removed}")

            if should_have_leading and chunk and not chunk[0].isspace():
                chunk = ' ' + chunk
                logger.debug(f"    Added leading space")

            needs_trailing = False

            if should_have_trailing:
                needs_trailing = True
                logger.debug(f"    Original had trailing space")
            elif space_was_removed:
                needs_trailing = True
                logger.debug(f"    Space was removed during split - preserving")
            else:
                next_content_idx = self._find_next_content_node(valid_nodes, idx)
                if next_content_idx is not None:
                    next_ws = whitespace_map.get(next_content_idx, {})
                    if next_ws.get('leading', False):
                        needs_trailing = True
                        logger.debug(f"    Next node ({next_content_idx}) needs leading space")

            if needs_trailing and chunk and not chunk[-1].isspace():
                chunk = chunk + ' '
                logger.debug(f"    Added trailing space")

            logger.debug(f"    Final: {repr(chunk)}")

            self._set_text_at_location(el, loc, chunk)
            remaining = new_remaining

    def _analyze_element_boundaries(self, nodes, root_element):
        boundaries = []

        for idx, (el, loc, orig_text, orig_stripped, tag) in enumerate(nodes):
            if loc == 'tail' and tag in ['i', 'b', 'em', 'strong', 'u', 'span']:
                boundaries.append({
                    'node_idx': idx,
                    'type': 'end',
                    'element_tag': tag
                })

        return {'boundaries': boundaries}

    def _split_at_sentence_boundary(self, text, target_len):
        if target_len >= len(text):
            return text, '', False

        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        best_pos = -1

        for ending in sentence_endings:
            pos = text.rfind(ending, 0, target_len + 20)
            if pos > best_pos and pos > 0:
                best_pos = pos + len(ending) - 1

        if best_pos > 0:
            chunk = text[:best_pos + 1]
            remaining = text[best_pos + 1:].lstrip()
            return chunk, remaining, True

        return self._split_at_natural_break(text, target_len)

    def _build_whitespace_map(self, nodes):
        whitespace_map = {}

        for idx, (el, loc, orig_text, orig_stripped, tag) in enumerate(nodes):
            if not orig_stripped:
                continue

            has_leading = orig_text != orig_text.lstrip()
            has_trailing = orig_text != orig_text.rstrip()

            whitespace_map[idx] = {
                'leading': has_leading,
                'trailing': has_trailing
            }

        return whitespace_map

    def _find_next_content_node(self, nodes, current_idx):
        for idx in range(current_idx + 1, len(nodes)):
            el, loc, text, stripped, tag = nodes[idx]
            if stripped:
                return idx
        return None

    def _split_at_natural_break(self, text, target_len):
        if target_len >= len(text):
            return text, '', False

        search_start = max(0, target_len - 20)
        search_end = min(len(text), target_len + 20)

        space_pos = text.rfind(' ', search_start, search_end)

        if space_pos != -1 and space_pos > 0:
            chunk = text[:space_pos]
            remaining = text[space_pos + 1:]
            return chunk, remaining, True

        for punct in [',', '.', ';', ':', '!', '?']:
            punct_pos = text.rfind(punct, search_start, search_end)
            if punct_pos != -1 and punct_pos > 0:
                chunk = text[:punct_pos + 1]
                remaining = text[punct_pos + 1:].lstrip()

                space_removed = text[punct_pos + 1:punct_pos + 2] == ' '
                return chunk, remaining, space_removed

        chunk = text[:target_len]
        remaining = text[target_len:]
        return chunk, remaining, False

    def _find_anchor_points(self, original_texts, translation):
        anchors = []
        trans_lower = translation.lower()

        logger.debug(f"\n=== SEARCHING FOR ANCHOR POINTS ===")
        logger.debug(f"Translation: {repr(translation)}")

        content_nodes = []
        for idx, (el, loc, text, stripped, tag) in enumerate(original_texts):
            if stripped:
                content_nodes.append((idx, el, loc, text, stripped, tag))

        logger.debug(f"Content nodes: {len(content_nodes)}")

        for idx, el, loc, text, stripped, tag in content_nodes:
            number_matches = re.finditer(r'\d+', stripped)

            for match in number_matches:
                number = match.group()

                if number in translation:
                    trans_pos = translation.find(number)

                    if not any(a[0] == idx and a[2] == number for a in anchors):
                        anchors.append((idx, trans_pos, number))
                        logger.debug(f"  ✓ Number anchor: node {idx}, pos {trans_pos}, '{number}'")

        for idx, el, loc, text, stripped, tag in content_nodes:
            words = stripped.split()

            for word in words:
                if len(word) >= 3 and word[0].isupper():
                    clean_word = word.rstrip('.,!?;:')

                    if clean_word.lower() in trans_lower:
                        trans_pos = trans_lower.find(clean_word.lower())

                        if not any(a[0] == idx and a[2].lower() == clean_word.lower() for a in anchors):
                            anchors.append((idx, trans_pos, clean_word))
                            logger.debug(f"  ✓ Proper noun anchor: node {idx}, pos {trans_pos}, '{clean_word}'")

        for idx, el, loc, text, stripped, tag in content_nodes:
            if stripped in ['.', '!', '?']:
                if stripped in translation:
                    trans_pos = translation.find(stripped)

                    if not any(a[0] == idx and a[2] == stripped for a in anchors):
                        anchors.append((idx, trans_pos, stripped))
                        logger.debug(f"  ✓ Punctuation anchor: node {idx}, pos {trans_pos}, '{stripped}'")

        anchors.sort(key=lambda x: x[1])

        logger.debug(f"\n=== FOUND {len(anchors)} ANCHORS ===")
        for idx, trans_pos, text in anchors:
            logger.debug(f"  [{idx}] pos={trans_pos}: '{text}'")

        return anchors

    def _distribute_translation_with_anchors(self, root_element, original_texts, translation, anchors):
        logger.debug(f"\n=== ANCHOR-BASED DISTRIBUTION ===")
        logger.debug(f"Anchors: {len(anchors)}")

        whitespace_map = self._build_whitespace_map(original_texts)

        content_nodes = []
        for idx, (el, loc, text, stripped, tag) in enumerate(original_texts):
            if stripped:
                content_nodes.append((idx, el, loc, text, stripped, tag))

        for segment_idx in range(len(anchors) + 1):
            logger.debug(f"\n--- Segment {segment_idx} ---")

            if segment_idx == 0:
                start_node_idx = 0
                start_trans_pos = 0

                if len(anchors) > 0:
                    end_node_idx = anchors[0][0]
                    end_trans_pos = anchors[0][1]
                else:
                    end_node_idx = len(content_nodes) - 1
                    end_trans_pos = len(translation)

            elif segment_idx == len(anchors):
                start_node_idx = anchors[-1][0] + 1
                start_trans_pos = anchors[-1][1] + len(anchors[-1][2])

                end_node_idx = len(content_nodes) - 1
                end_trans_pos = len(translation)

            else:
                start_node_idx = anchors[segment_idx - 1][0] + 1
                start_trans_pos = anchors[segment_idx - 1][1] + len(anchors[segment_idx - 1][2])

                end_node_idx = anchors[segment_idx][0]
                end_trans_pos = anchors[segment_idx][1]

            segment_nodes = []
            for idx, el, loc, text, stripped, tag in content_nodes:
                if start_node_idx <= idx <= end_node_idx:
                    segment_nodes.append((idx, el, loc, text, stripped, tag))

            segment_translation = translation[start_trans_pos:end_trans_pos].strip()

            logger.debug(f"  Nodes: {start_node_idx} to {end_node_idx}")
            logger.debug(f"  Trans: pos {start_trans_pos} to {end_trans_pos}")
            logger.debug(f"  Text: {repr(segment_translation[:50])}")

            if not segment_nodes:
                logger.debug(f"  (no nodes in segment - skip)")
                continue

            if not segment_translation:
                logger.debug(f"  (no translation in segment - clearing nodes)")
                for idx, el, loc, text, stripped, tag in segment_nodes:
                    self._set_text_at_location(el, loc, '')
                continue

            if segment_idx < len(anchors):
                anchor_node_idx, anchor_trans_pos, anchor_text = anchors[segment_idx]

                for idx, el, loc, text, stripped, tag in segment_nodes:
                    if idx == anchor_node_idx:
                        self._set_text_at_location(el, loc, anchor_text)
                        logger.debug(f"  Set anchor node [{idx}] to: '{anchor_text}'")

                        segment_nodes = [n for n in segment_nodes if n[0] != anchor_node_idx]
                        break

            if len(segment_nodes) == 1:
                idx, el, loc, text, stripped, tag = segment_nodes[0]

                ws_info = whitespace_map.get(idx, {'leading': False, 'trailing': False})
                chunk = segment_translation

                if ws_info['leading'] and chunk and not chunk[0].isspace():
                    chunk = ' ' + chunk
                if ws_info['trailing'] and chunk and not chunk[-1].isspace():
                    chunk = chunk + ' '

                self._set_text_at_location(el, loc, chunk)
                logger.debug(f"  Single node [{idx}]: '{chunk}'")

            elif len(segment_nodes) > 1:
                total_len = sum(len(s) for _, _, _, _, s, _ in segment_nodes)

                if total_len == 0:
                    continue

                remaining = segment_translation

                for node_idx, (idx, el, loc, text, stripped, tag) in enumerate(segment_nodes):
                    if node_idx == len(segment_nodes) - 1:
                        chunk = remaining
                    else:
                        ratio = len(stripped) / total_len
                        target_len = int(len(segment_translation) * ratio)

                        chunk, remaining, space_removed = self._split_at_natural_break(remaining, target_len)

                    ws_info = whitespace_map.get(idx, {'leading': False, 'trailing': False})

                    if ws_info['leading'] and chunk and not chunk[0].isspace():
                        chunk = ' ' + chunk
                    if ws_info['trailing'] and chunk and not chunk[-1].isspace():
                        chunk = chunk + ' '

                    self._set_text_at_location(el, loc, chunk)
                    logger.debug(f"  Node [{idx}]: '{chunk[:50]}'")

        logger.debug("\n=== ANCHOR-BASED DISTRIBUTION COMPLETE ===")

    def _rebuild_element_from_placeholders(
        self,
        parent,
        text,
        formatting_map,
        reserve_elements
    ):
        logger.debug(f"\n=== REBUILDING FROM PLACEHOLDERS ===")
        logger.debug(f"Text length: {len(text)}")
        logger.debug(f"Text preview: {repr(text[:200])}")
        logger.debug(f"Formatting map: {len(formatting_map)} entries")
        logger.debug(f"Reserve elements: {len(reserve_elements)} entries")

        pattern = r'<p_(\d{2})>|</p_(\d{2})>|<id_(\d{2})>'

        stack = [(parent, None)]

        buffer = ""
        last_pos = 0

        match_count = 0

        for match in re.finditer(pattern, text):
            match_count += 1

            buffer += text[last_pos:match.start()]

            placeholder = match.group(0)

            if placeholder.startswith('<p_') and not placeholder.startswith('</p_'):
                tag_id = int(match.group(1))

                logger.debug(f"  [{match_count}] Opening <p_{tag_id:02d}>")

                if buffer:
                    if tag_id in formatting_map:
                        info = formatting_map[tag_id]
                        if info.get('has_leading_space', False) and not buffer[-1].isspace():
                            buffer += ' '
                            logger.debug(f"      Restored leading space before <p_{tag_id:02d}>")

                    self._append_text_to_element(stack[-1][0], buffer)
                    logger.debug(f"      Flushed buffer ({len(buffer)} chars)")
                    buffer = ""

                else:
                    if tag_id in formatting_map:
                        info = formatting_map[tag_id]
                        if info.get('has_leading_space', False):
                            current_parent = stack[-1][0]
                            if len(current_parent) > 0:
                                last_child = current_parent[-1]
                                last_tail = last_child.tail or ''
                                if not last_tail or not last_tail[-1].isspace():
                                    last_child.tail = last_tail + ' '
                                    logger.debug(f"      Restored leading space (via tail) before <p_{tag_id:02d}>")
                            else:
                                parent_txt = current_parent.text or ''
                                if not parent_txt or not parent_txt[-1].isspace():
                                    current_parent.text = parent_txt + ' '
                                    logger.debug(f"      Restored leading space (via parent.text) before <p_{tag_id:02d}>")

                if tag_id in formatting_map:
                    info = formatting_map[tag_id]
                    new_elem = self._create_formatting_element(info)
                    stack[-1][0].append(new_elem)
                    stack.append((new_elem, tag_id))
                    logger.debug(f"      Created <{info['tag']}> element")
                else:
                    logger.warning(f"      ⚠️ Tag ID {tag_id} not in formatting_map - skipping")

            elif placeholder.startswith('</p_'):
                tag_id = int(match.group(2))

                logger.debug(f"  [{match_count}] Closing </p_{tag_id:02d}>")

                if buffer:
                    self._append_text_to_element(stack[-1][0], buffer)
                    logger.debug(f"      Flushed buffer ({len(buffer)} chars)")
                    buffer = ""

                if len(stack) > 1 and stack[-1][1] == tag_id:
                    stack.pop()
                    logger.debug(f"      Popped stack")

                    if tag_id in formatting_map:
                        info = formatting_map[tag_id]
                        if info.get('has_trailing_space', False):
                            next_char = text[match.end():match.end() + 1]

                            if next_char and not next_char.isspace() and next_char not in '.,;:!?':
                                buffer += ' '
                                logger.debug(f"      Restored trailing space after </p_{tag_id:02d}>")
                else:
                    logger.warning(f"      ⚠️ Stack mismatch: expected {stack[-1][1]}, got {tag_id}")

            elif placeholder.startswith('<id_'):
                reserve_id = int(match.group(3))

                logger.debug(f"  [{match_count}] Reserve <id_{reserve_id:02d}>")

                if buffer:
                    self._append_text_to_element(stack[-1][0], buffer)
                    logger.debug(f"      Flushed buffer ({len(buffer)} chars)")
                    buffer = ""

                if reserve_id < len(reserve_elements):
                    self._insert_reserve_element_html(
                        stack[-1][0],
                        reserve_elements[reserve_id]
                    )
                    logger.debug(f"      Inserted reserve element")
                else:
                    logger.error(f"      ❌ Invalid reserve_id {reserve_id} (max: {len(reserve_elements) - 1})")

            last_pos = match.end()

        buffer += text[last_pos:]

        if buffer:
            self._append_text_to_element(stack[-1][0], buffer)
            logger.debug(f"  Final buffer flush ({len(buffer)} chars)")

        logger.debug(f"=== REBUILD COMPLETE ===")
        logger.debug(f"  Processed {match_count} placeholders")
        logger.debug(f"  Final stack depth: {len(stack)}")

        if len(stack) > 1:
            logger.warning(f"  ⚠️ Stack not fully unwound - {len(stack) - 1} unclosed tag(s)")

    def _create_formatting_element(self, info):
        tag_name = info['tag']
        attributes = info['attributes']

        nsmap = {None: self.ns['x']}

        elem = etree.Element(f"{{{self.ns['x']}}}{tag_name}", nsmap=nsmap)

        for name, value in attributes.items():
            elem.set(name, value)

        return elem

    def _append_text_to_element(self, element, text):
        if len(element) == 0:
            element.text = (element.text or "") + text
        else:
            last_child = element[-1]
            last_child.tail = (last_child.tail or "") + text

    def _insert_reserve_element_html(self, parent, reserve_html):
        try:
            logger.debug(f"  Inserting reserve element: {repr(reserve_html[:100])}")

            if 'xmlns' in reserve_html:
                try:
                    reserve_elem = etree.fromstring(reserve_html.encode('utf-8'))
                    parent.append(reserve_elem)
                    logger.debug(f"  ✓ Inserted reserve element (with xmlns)")
                    return
                except etree.XMLSyntaxError as e:
                    logger.warning(f"  Failed to parse with xmlns: {e}")

            try:
                wrapped_html = f'<_temp xmlns="{self.ns["x"]}">{reserve_html}</_temp>'
                temp_root = etree.fromstring(wrapped_html.encode('utf-8'))

                if len(temp_root) > 0:
                    reserve_elem = temp_root[0]
                    parent.append(reserve_elem)
                    logger.debug(f"  ✓ Inserted reserve element (wrapped)")
                    return
                else:
                    logger.warning(f"  Wrapped element has no children")

            except etree.XMLSyntaxError as e:
                logger.warning(f"  Failed to parse wrapped: {e}")

            tag_match = re.match(r'<(\w+)([^>]*)/?>', reserve_html)

            if tag_match:
                tag_name = tag_match.group(1)
                attrs_str = tag_match.group(2)

                logger.debug(f"  Creating element manually: <{tag_name}>")

                nsmap = {None: self.ns['x']}
                reserve_elem = etree.Element(f"{{{self.ns['x']}}}{tag_name}", nsmap=nsmap)

                attr_pattern = r'(\w+)="([^"]*)"'
                for attr_match in re.finditer(attr_pattern, attrs_str):
                    attr_name = attr_match.group(1)
                    attr_value = attr_match.group(2)
                    reserve_elem.set(attr_name, attr_value)

                parent.append(reserve_elem)
                logger.debug(f"  ✓ Created element manually: <{tag_name}>")
                return

            raise ValueError("All parsing methods failed")

        except Exception as e:
            logger.error(f"❌ Failed to insert reserve element as HTML: {e}")
            logger.error(f"   HTML was: {repr(reserve_html)}")
            logger.error(f"   Falling back to TEXT insertion (structure will be lost)")

            self._append_text_to_element(parent, reserve_html)
            logger.warning(f"   ⚠️ Reserve element inserted as TEXT (will appear escaped in output)")

    def _color_element_tree(self, element, color):
        style = element.get('style', '')
        if style:
            element.set('style', f"{style}; color:{color}")
        else:
            element.set('style', f"color:{color}")

        for child in element.iter():
            if child is element:
                continue

            style = child.get('style', '')
            if style:
                child.set('style', f"{style}; color:{color}")
            else:
                child.set('style', f"color:{color}")

    def _restore_reserved_elements(self, translation, para):
        result = translation

        prefix_tags = para.get('prefix_reserve_tags', [])
        suffix_tags = para.get('suffix_reserve_tags', [])

        if prefix_tags:
            prefix_str = ' '.join(prefix_tags)
            result = prefix_str + ' ' + result
            logger.debug(f"Added prefix tags: {prefix_str}")

        if suffix_tags:
            suffix_str = ' '.join(suffix_tags)
            result = result + ' ' + suffix_str
            logger.debug(f"Added suffix tags: {suffix_str}")

        if 'reserve_elements' in para and para['reserve_elements']:
            logger.debug(f"Restoring {len(para['reserve_elements'])} reserve elements")

            placeholder_pattern = para.get('placeholder_pattern', '<id_{:02d}>')

            for idx, element_html in enumerate(para['reserve_elements']):
                placeholder = placeholder_pattern.format(idx)

                clean_html = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', '', element_html)

                result = result.replace(placeholder, clean_html)
                logger.debug(f"  {placeholder} → HTML")

            result = re.sub(r'(\w)(<a[\s>])', r'\1 \2', result)
            result = re.sub(r'(</a>)([\w\u00C0-\u024F])', r'\1 \2', result)
            logger.debug("Applied spacing fix around <a> tags")

        if 'inline_formatting_map' in para and para['inline_formatting_map']:
            logger.debug(f"Restoring {len(para['inline_formatting_map'])} inline formatting elements")

            formatting_map = para['inline_formatting_map']

            for elem_id in sorted(formatting_map.keys()):
                info = formatting_map[elem_id]

                opening_ph = info['opening_placeholder']
                closing_ph = info['closing_placeholder']
                tag = info['tag']
                attributes = info['attributes']

                if attributes:
                    attrs_str = ' '.join(f'{k}="{html.escape(v)}"' for k, v in attributes.items())
                    opening_tag = f'<{tag} {attrs_str}>'
                else:
                    opening_tag = f'<{tag}>'

                closing_tag = f'</{tag}>'

                result = result.replace(opening_ph, opening_tag)
                result = result.replace(closing_ph, closing_tag)

        logger.debug("Restored all placeholders to HTML (lxml will handle text escaping)")

        return result

    def _cleanup_translation(self, translation):
        result = translation

        result = re.sub(r'\n\s*(?=<)', '', result)

        result = re.sub(r'(?<=>)\s*\n', '', result)

        result = result.replace('\n', ' ')

        result = re.sub(r'  +', ' ', result)

        result = re.sub(r'((\w)\2{3})\2*', r'\1', result)

        result = re.sub(
            r'((?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})\1+',
            r'\1',
            result
        )

        return result

    def _should_preserve_link_structure(self, element):
        return False

    def _create_translation_preserving_links(self, original_element, translation_text, root):
        pass

    def _insert_by_position(self, original, translation, element_name, root):
        pass

    def _insert_inline_translation(self, original, translation, parent, root):
        pass

    def _create_side_by_side_table(self, original, translation, root):
        pass

