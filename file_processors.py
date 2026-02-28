# file_processors.py
import copy
import ebooklib
import logging
import os
import re
import uuid
from abc import ABC, abstractmethod
from ebooklib import epub
from epub_utils import NAMESPACES, read_epub
from lxml import etree
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class FileProcessor(ABC):
    @abstractmethod
    def load(self, path: str) -> Tuple[List[Dict], Optional[any]]:
        pass

    @abstractmethod
    def get_file_type(self) -> str:
        pass

class EPUBProcessor(FileProcessor):
    def __init__(self, app_settings: dict):
        self.app_settings = app_settings
        self.book = None
        self.paragraphs = []

        self.skip_inline_tags = app_settings.get('skip_inline_tags', {})

        self.ns = NAMESPACES

        self.PRIORITY_TAGS = ['p', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']

        self.NON_INLINE_ELEMENTS = {
            'address', 'blockquote', 'dialog', 'div', 'figure', 'figcaption',
            'footer', 'header', 'legend', 'main', 'p', 'pre', 'search', 'article',
            'aside', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hgroup', 'nav',
            'section', 'dd', 'dl', 'dt', 'menu', 'ol', 'ul', 'table', 'caption',
            'colgroup', 'col', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th', 'li'
        }

        self.NOISE_TAGS = ['rt', 'rp']

        self.RESERVE_TAGS = [
            'img', 'code', 'br', 'hr', 'sub', 'sup', 'kbd',
            'abbr', 'wbr', 'var', 'canvas', 'svg', 'script',
            'style', 'math'
        ]

        logger.info(f"EPUBProcessor initialized with {len(self.RESERVE_TAGS)} structural reserve tags, "
                    f"skip_inline_tags: {self.skip_inline_tags}")

    def get_file_type(self) -> str:
        return "epub"

    def load(self, path: str) -> Tuple[List[Dict], any]:
        try:
            self.book = read_epub(path)
            self.paragraphs = []

            seen = set()

            spine_order = {}
            try:
                opf_path = os.path.join(self.book.temp_dir, self.book.opf_path)

                with open(opf_path, 'rb') as f:
                    opf_tree = etree.parse(f)

                OPF_NS = {'opf': 'http://www.idpf.org/2007/opf'}
                spine_elem = opf_tree.find('.//opf:spine', namespaces=OPF_NS)

                if spine_elem is not None:
                    itemrefs = spine_elem.findall('.//opf:itemref', namespaces=OPF_NS)

                    for idx, itemref in enumerate(itemrefs):
                        idref = itemref.get('idref')
                        if idref:
                            spine_order[idref] = idx

                    logger.info(f"📚 Spine order built from OPF: {len(spine_order)} items")
                else:
                    logger.warning("⚠️ Warning: No <spine> found in OPF. Using alphabetical order.")

            except Exception as e:
                logger.warning(f"⚠️ Warning: Could not build spine order ({e}). Using alphabetical order.")
                spine_order = {}

            doc_items = list(self.book.get_items_of_type('DOCUMENT'))

            def get_sort_key(item):
                try:
                    item_id = None

                    for manifest_item in self.book.manifest_items:
                        if isinstance(manifest_item, dict):
                            if manifest_item.get('href') == item.href:
                                item_id = manifest_item.get('id')
                                break
                        elif hasattr(manifest_item, 'href') and hasattr(manifest_item, 'id'):
                            if manifest_item.href == item.href:
                                item_id = manifest_item.id
                                break

                    if not item_id:
                        item_id = os.path.splitext(os.path.basename(item.href))[0]

                    if item_id in spine_order:
                        return (0, spine_order[item_id])
                    else:
                        return (1, item.href)

                except Exception as e:
                    logger.error(f"⚠️ Sort key error for {item.href}: {e}")
                    return (2, item.href)

            doc_items.sort(key=get_sort_key)

            logger.info(f"📖 Processing {len(doc_items)} documents in spine order...\n")

            total_items = 0
            items_with_body = 0
            total_priority_found = 0
            total_processed = 0

            for item in doc_items:
                total_items += 1

                if item.data is None:
                    logger.debug(f"❌ Item {item.href}: No data")
                    continue

                body = item.data.find('.//x:body', namespaces=NAMESPACES)
                if body is None:
                    logger.debug(f"❌ Item {item.href}: No body")
                    continue

                items_with_body += 1

                priority_xpath = ' | '.join([f'.//x:{tag}' for tag in self.PRIORITY_TAGS])
                priority_elements = body.xpath(priority_xpath, namespaces=NAMESPACES)

                li_elements = body.xpath('.//x:li', namespaces=NAMESPACES)

                logger.debug(f"📄 {item.href}")
                logger.debug(f"   Priority elements: {len(priority_elements)}")
                logger.debug(f"   <li> elements: {len(li_elements)}")

                if priority_elements:
                    total_priority_found += len(priority_elements)
                    for idx, elem in enumerate(priority_elements[:3]):
                        tag = etree.QName(elem).localname
                        text = etree.tostring(elem, encoding='unicode', method='text').strip()[:60]
                        logger.debug(f"   Priority[{idx+1}] <{tag}>: {text}...")

                if li_elements:
                    for idx, elem in enumerate(li_elements[:3]):
                        text = etree.tostring(elem, encoding='unicode', method='text').strip()[:60]
                        logger.debug(f"   <li>[{idx+1}]: {text}...")

                before_count = len(self.paragraphs)

                logger.debug(f"   🔍 Starting extraction...")
                self._extract_elements_lxml(
                    body,
                    item.href,
                    seen
                )

                after_count = len(self.paragraphs)
                added = after_count - before_count
                total_processed += added

                logger.debug(f"   ✓ Added to paragraphs: {added}\n")

            logger.info("=" * 60)
            logger.info("SUMMARY:")
            logger.info(f"  Total items: {total_items}")
            logger.info(f"  Items with body: {items_with_body}")
            logger.info(f"  Priority elements found: {total_priority_found}")
            logger.info(f"  Fragments added: {total_processed}")
            logger.info("=" * 60 + "\n")

            return self.paragraphs, self.book

        except Exception as e:
            logger.error(f"EPUB load error: {e}", exc_info=True)
            raise

    def _is_non_translatable_content(self, text: str) -> bool:
        text_no_placeholders = re.sub(r'<id_\d{2}>', '', text)
        text_no_placeholders = re.sub(r'</?p_\d{2}>', '', text_no_placeholders)

        text_stripped = text_no_placeholders.strip()

        if not text_stripped:
            logger.debug("Non-translatable: empty after removing placeholders")
            return True

        non_alpha_pattern = r'^[\s\d\.,!?:;…\-–—\'\"\u201e\u201d\u201a\u2019]+$'

        if re.match(non_alpha_pattern, text_stripped):
            logger.debug(f"Non-translatable: only special chars: {repr(text_stripped)}")
            return True

        separator_pattern = r'^[\s\*•–—]+$'

        if re.match(separator_pattern, text_stripped):
            logger.debug(f"Non-translatable: chapter separator: {repr(text_stripped)}")
            return True

        repeated_special = r'^([\*•–—])\s*(\1\s*)+$'

        if re.match(repeated_special, text_stripped):
            logger.debug(f"Non-translatable: repeated special chars: {repr(text_stripped)}")
            return True

        return False

    def _extract_elements_lxml(self, root, item_href, seen):
        CONTAINER_TAGS = {
            'ul', 'ol', 'dl',
            'table', 'tbody', 'thead', 'tfoot', 'tr',
            'div', 'section', 'article', 'aside', 'nav', 'main',
            'header', 'footer', 'figure', 'body',
            'blockquote'
        }

        for child in root:
            if not isinstance(child, etree._Element):
                logger.debug(f"   ⏭ Skipping non-element node")
                continue

            tag_name = etree.QName(child).localname

            if tag_name in self.PRIORITY_TAGS:
                logger.debug(f"   ✓ Priority tag <{tag_name}> - processing directly")
                self._process_element_lxml(child, item_href, seen)
                continue

            if tag_name in CONTAINER_TAGS:
                logger.debug(f"   📦 Container <{tag_name}> - recursing into children")
                self._extract_elements_lxml(child, item_href, seen)
                continue

            element_has_content = False

            if self._is_inline_only_lxml(child):
                element_has_content = True
                logger.debug(f"   📝 <{tag_name}> is inline-only - will process")
            elif self._has_any_text(child):
                element_has_content = True
                text_preview = self._get_element_text(child)[:40]
                logger.debug(f"   📝 <{tag_name}> has text content - will process: {text_preview}...")

            if element_has_content:
                logger.debug(f"   ✓ Processing element: <{tag_name}>")
                self._process_element_lxml(child, item_href, seen)
            else:
                logger.debug(f"   🔄 <{tag_name}> has no direct content - recursing to find children")
                self._extract_elements_lxml(child, item_href, seen)

    def _is_inline_only_lxml(self, element):
        non_inline_xpath = ' | '.join([f'.//x:{tag}' for tag in self.NON_INLINE_ELEMENTS])
        found = element.xpath(non_inline_xpath, namespaces=NAMESPACES)
        return len(found) == 0

    def _has_any_text(self, element):
        if element.text and element.text.strip():
            return True

        for descendant in element.iter():
            if descendant is element:
                continue
            if descendant.text and descendant.text.strip():
                return True
            if descendant.tail and descendant.tail.strip():
                return True

        return False

    def _get_element_text(self, element):
        return etree.tostring(element, encoding='unicode', method='text').strip()

    def _process_element_lxml(self, element, item_href, seen):
        tag_name = etree.QName(element).localname

        CONTAINER_TAGS = {
            'ul', 'ol', 'dl', 'table', 'tbody', 'thead', 'tfoot', 'tr',
            'div', 'section', 'article', 'aside', 'nav', 'main',
            'header', 'footer', 'figure', 'body'
        }

        if tag_name in CONTAINER_TAGS:
            logger.warning(f"⚠️ Container <{tag_name}> reached _process_element_lxml - SKIPPING")
            return

        use_inline = self.app_settings.get('use_inline_formatting', True)

        if use_inline:
            self._process_element_new(element, item_href, seen)
        else:
            self._process_element_legacy(element, item_href, seen)

    def _process_element_legacy(self, element, item_href, seen):
        tag_name = etree.QName(element).localname

        if element.get('id') is None:
            element.set('id', f"trans_{uuid.uuid4()}")

        element_id = element.get('id')

        element_copy = copy.deepcopy(element)

        for noise_tag in self.NOISE_TAGS:
            noise_xpath = f'.//x:{noise_tag}'
            for noise_elem in element_copy.xpath(noise_xpath, namespaces=self.ns):
                parent = noise_elem.getparent()
                if parent is not None:
                    parent.remove(noise_elem)

        reserve_elements = []
        placeholder_pattern = '<id_{:02d}>'
        reserve_counter = 0

        legacy_reserve_tags = self.RESERVE_TAGS

        for reserve_tag in legacy_reserve_tags:
            reserve_xpath = f'.//x:{reserve_tag}'
            for reserve_elem in element_copy.xpath(reserve_xpath, namespaces=self.ns):
                reserve_html = etree.tostring(
                    reserve_elem,
                    encoding='unicode',
                    method='xml',
                    with_tail=False
                )

                reserve_html = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', '', reserve_html)

                reserve_elements.append(reserve_html)

                placeholder = placeholder_pattern.format(reserve_counter)
                reserve_counter += 1

                parent = reserve_elem.getparent()
                if parent is not None:
                    tail = reserve_elem.tail or ''
                    prev = reserve_elem.getprevious()
                    if prev is not None:
                        existing = prev.tail or ''

                        space_before = '' if (not existing or existing[-1].isspace()) else ' '
                        prev.tail = existing + space_before + placeholder + tail
                    else:
                        existing = parent.text or ''
                        space_before = '' if (not existing or existing[-1].isspace()) else ' '
                        parent.text = existing + space_before + placeholder + tail
                    parent.remove(reserve_elem)

        clean_text = self._get_element_text(element_copy)

        if not clean_text or not clean_text.strip():
            return

        prefix_tags, suffix_tags, clean_text = self._extract_boundary_reserve_tags(clean_text)

        is_non_translatable = self._is_non_translatable_content(clean_text)

        if is_non_translatable:
            logger.debug(f"Marking element {element_id} as non-translatable (will be preserved in EPUB)")

        key = (item_href, element_id)
        if key in seen:
            return
        seen.add(key)

        original_html = etree.tostring(
            element,
            encoding='unicode',
            method='xml',
            pretty_print=False
        )

        para = {
            "id": element_id,
            "original_text": clean_text,
            "translated_text": "",
            "is_translated": False,
            "item_href": item_href,
            "element_type": tag_name,
            "original_html": original_html,
            "has_mismatch": False,
            "reserve_elements": reserve_elements,
            "placeholder_pattern": placeholder_pattern,
            "processing_mode": "legacy",
            "prefix_reserve_tags": prefix_tags,
            "suffix_reserve_tags": suffix_tags,
            "is_non_translatable": is_non_translatable
        }

        self.paragraphs.append(para)

        logger.debug(f"[LEGACY] Processed element {element_id}: {len(reserve_elements)} reserve, "
                     f"{len(prefix_tags)} prefix tags, {len(suffix_tags)} suffix tags, "
                     f"non-translatable={is_non_translatable}")

    def _process_element_new(self, element, item_href, seen):
        tag_name = etree.QName(element).localname

        if element.get('id') is None:
            element.set('id', f"trans_{uuid.uuid4()}")

        element_id = element.get('id')
        element_copy = copy.deepcopy(element)

        for noise_tag in self.NOISE_TAGS:
            noise_xpath = f'.//x:{noise_tag}'
            for noise_elem in element_copy.xpath(noise_xpath, namespaces=NAMESPACES):
                parent = noise_elem.getparent()
                if parent is not None:
                    parent.remove(noise_elem)

        reserve_elements = []
        placeholder_pattern = '<id_{:02d}>'
        reserve_counter = 0

        for reserve_tag in self.RESERVE_TAGS:
            reserve_xpath = f'.//x:{reserve_tag}'
            for reserve_elem in element_copy.xpath(reserve_xpath, namespaces=NAMESPACES):
                reserve_elem_copy = copy.deepcopy(reserve_elem)
                reserve_elem_copy.tail = None
                reserve_html = etree.tostring(reserve_elem_copy, encoding='unicode', method='xml')
                reserve_elements.append(reserve_html)

                placeholder = placeholder_pattern.format(reserve_counter)
                reserve_counter += 1

                parent = reserve_elem.getparent()
                if parent is not None:
                    tail = reserve_elem.tail or ''
                    prev = reserve_elem.getprevious()
                    if prev is not None:
                        prev.tail = (prev.tail or '') + placeholder + tail
                    else:
                        parent.text = (parent.text or '') + placeholder + tail
                    parent.remove(reserve_elem)

        self._remove_useless_spans(element_copy)

        inline_formatting_map = {}
        inline_counter = reserve_counter

        INLINE_FORMATTING_TAGS = [
            'a', 'i', 'b', 'em', 'strong',
            'u', 'sup', 'sub', 'small', 'span'
        ]

        clean_text, inline_formatting_map, final_counter = \
            self._replace_inline_formatting_with_placeholders(
                element_copy,
                INLINE_FORMATTING_TAGS,
                inline_counter,
                placeholder_pattern
            )

        clean_text = self._cleanup_empty_placeholders(clean_text)

        clean_text, inline_formatting_map = \
            self._flatten_placeholder_nesting(clean_text, inline_formatting_map)

        clean_text, non_translatable_map = \
            self._extract_non_translatable_placeholders(
                clean_text,
                inline_formatting_map
            )

        clean_text = self._cleanup_empty_placeholders(clean_text)

        if not clean_text or not clean_text.strip():
            return

        prefix_tags, suffix_tags, clean_text = \
            self._extract_boundary_reserve_tags(clean_text)

        is_non_translatable = self._is_non_translatable_content(clean_text)

        if is_non_translatable:
            logger.debug(f"Marking element {element_id} as non-translatable")

        auto_wrap_tags = self._detect_auto_wrap_tags(
            clean_text,
            inline_formatting_map
        )

        if auto_wrap_tags:
            clean_text = self._strip_outer_placeholders(
                clean_text,
                auto_wrap_tags
            )
            logger.debug(f"Auto-wrap detected: {len(auto_wrap_tags)} tag(s)")

        used_ids = set(int(m) for m in re.findall(r'<p_(\d{2})>', clean_text))
        non_translatable_ids = set(non_translatable_map.keys())

        auto_wrap_ids = set(tag_info['elem_id'] for tag_info in auto_wrap_tags) if auto_wrap_tags else set()
        keep_ids = used_ids | non_translatable_ids | auto_wrap_ids

        inline_formatting_map = {
            k: v for k, v in inline_formatting_map.items()
            if k in keep_ids
        }

        logger.debug(
            f"Pruned inline_formatting_map: {len(inline_formatting_map)} entries "
            f"(used: {sorted(used_ids)}, nt: {sorted(non_translatable_ids)}, "
            f"auto_wrap: {sorted(auto_wrap_ids)})"
        )

        key = (item_href, element_id)
        if key in seen:
            return
        seen.add(key)

        original_html = etree.tostring(
            element,
            encoding='unicode',
            method='xml',
            pretty_print=False
        )

        para = {
            "id": element_id,
            "original_text": clean_text,
            "translated_text": "",
            "is_translated": False,
            "item_href": item_href,
            "element_type": tag_name,
            "original_html": original_html,
            "has_mismatch": False,
            "reserve_elements": reserve_elements,
            "inline_formatting_map": inline_formatting_map,
            "non_translatable_placeholders": non_translatable_map,
            "placeholder_pattern": placeholder_pattern,
            "processing_mode": "inline",
            "prefix_reserve_tags": prefix_tags,
            "suffix_reserve_tags": suffix_tags,
            "is_non_translatable": is_non_translatable
        }

        if auto_wrap_tags:
            para['auto_wrap_tags'] = auto_wrap_tags

        self.paragraphs.append(para)

    def _remove_useless_spans(self, element):
        spans_to_remove = []

        for span in element.xpath('.//x:span', namespaces=NAMESPACES):
            has_class = span.get('class') is not None
            has_style = span.get('style') is not None
            has_id = span.get('id') is not None
            has_lang = span.get('lang') is not None
            has_dir = span.get('dir') is not None
            has_epub_type = span.get('{http://www.idpf.org/2007/ops}type') is not None

            if has_class or has_style or has_id or has_lang or has_dir or has_epub_type:
                continue

            text = (span.text or '')
            children = list(span)

            should_remove = False

            if not text.strip() and len(children) <= 1:
                should_remove = True

            if should_remove:
                spans_to_remove.append(span)

        removed_count = 0
        for span in reversed(spans_to_remove):
            parent = span.getparent()
            if parent is None:
                continue

            span_index = list(parent).index(span)

            if span.text:
                prev = span.getprevious()
                if prev is not None:
                    prev.tail = (prev.tail or '') + span.text
                else:
                    parent.text = (parent.text or '') + span.text

            for child in reversed(list(span)):
                parent.insert(span_index, child)

            if span.tail:
                if len(span) > 0:
                    last_child_of_span = span[-1]
                    last_child_of_span.tail = (last_child_of_span.tail or '') + span.tail
                else:
                    new_prev = span.getprevious()
                    if new_prev is not None:
                        new_prev.tail = (new_prev.tail or '') + span.tail
                    else:
                        parent.text = (parent.text or '') + span.tail

            parent.remove(span)
            removed_count += 1

        if removed_count > 0:
            logger.info(f"✓ Removed {removed_count} useless <span> wrapper(s)")

    def _replace_inline_formatting_with_placeholders(self, element, inline_tags, start_counter, placeholder_pattern):
        formatting_map = {}
        counter = start_counter

        STRUCTURAL_SPAN_CLASSES = {
            'first-letter',
            'last-word',
            'item-number',
            'element-number',
        }

        replacements = []

        tags_to_process = []
        for tag in inline_tags:
            if self.skip_inline_tags.get(tag, False):
                logger.debug(f"Skipping <{tag}> tags (user preference)")
            else:
                tags_to_process.append(tag)

        if not tags_to_process:
            logger.warning("All inline formatting tags are skipped - no placeholders will be created")

        for tag in tags_to_process:
            xpath = f'.//x:{tag}'
            for elem in element.xpath(xpath, namespaces=NAMESPACES):
                if tag == 'span':
                    epub_type = elem.get('{http://www.idpf.org/2007/ops}type')
                    if epub_type == 'pagebreak':
                        logger.debug(f"Skipping pagebreak span")
                        continue

                    class_attr = elem.get('class')
                    if class_attr in STRUCTURAL_SPAN_CLASSES:
                        logger.debug(f"Skipping structural span: class='{class_attr}'")
                        continue

                    if not elem.attrib:
                        text = (elem.text or '').strip()
                        if len(text) == 1 and text.isalpha():
                            parent = elem.getparent()
                            if parent is not None:
                                children = [c for c in parent if isinstance(c, etree._Element)]
                                if len(children) > 0 and children[0] == elem:
                                    logger.debug(f"Skipping drop cap span: '{text}'")
                                    continue

                elem_id = counter
                opening = "<p_{:02d}>".format(elem_id)
                closing = "</p_{:02d}>".format(elem_id)

                tag_name = etree.QName(elem).localname
                attributes = dict(elem.attrib)

                prev_sibling = elem.getprevious()
                if prev_sibling is not None:
                    preceding_text = prev_sibling.tail or ''
                else:
                    parent_elem = elem.getparent()
                    preceding_text = (parent_elem.text or '') if parent_elem is not None else ''
                has_leading_space = bool(preceding_text) and preceding_text[-1] in ' \t\n\r\u00a0'

                elem_tail = elem.tail or ''
                has_trailing_space = bool(elem_tail) and elem_tail[0] in ' \t\n\r\u00a0'

                logger.debug(
                    f"  elem_id={elem_id} <{tag_name}> "
                    f"has_leading_space={has_leading_space}, has_trailing_space={has_trailing_space}"
                )

                formatting_map[elem_id] = {
                    'tag': tag_name,
                    'attributes': attributes,
                    'opening_placeholder': opening,
                    'closing_placeholder': closing,
                    'has_leading_space': has_leading_space,
                    'has_trailing_space': has_trailing_space,
                }

                replacements.append((elem, opening, closing, elem_id))
                counter += 1

        result_text = self._serialize_element_with_placeholders(element, replacements)

        return result_text, formatting_map, counter

    def _serialize_element_with_placeholders(self, element, replacements):
        replace_map = {id(elem): (opening, closing) for elem, opening, closing, _ in replacements}

        def process_node(node, depth=0):
            parts = []

            node_id = id(node)
            if node_id in replace_map:
                opening, closing = replace_map[node_id]

                parts.append(opening)

                if node.text:
                    parts.append(node.text)

                for child in node:
                    parts.append(process_node(child, depth + 1))
                    if child.tail:
                        parts.append(child.tail)

                parts.append(closing)

                return ''.join(parts)

            else:
                if node.text:
                    parts.append(node.text)

                for child in node:
                    parts.append(process_node(child, depth + 1))
                    if child.tail:
                        parts.append(child.tail)

                return ''.join(parts)

        result = process_node(element)

        return result

    def _cleanup_empty_placeholders(self, text):
        original_text = text
        max_iterations = 10

        for iteration in range(max_iterations):
            pattern = r'<p_(\d{2})></p_\1>'
            new_text = re.sub(pattern, '', text)

            if new_text == text:
                break

            text = new_text

        if text != original_text:
            original_count = original_text.count('<p_')
            new_count = text.count('<p_')
            removed = (original_count - new_count) // 2
            logger.info(f"✓ Removed {removed} empty placeholder pair(s)")

        return text

    def _flatten_placeholder_nesting(self, text, formatting_map):
        pattern = r'<p_(\d{2})>\s*(<p_\d{2}>.*?</p_\d{2}>)\s*</p_\1>'

        max_iterations = 5
        removed_ids = set()

        for iteration in range(max_iterations):
            matches = list(re.finditer(pattern, text, re.DOTALL))

            if not matches:
                break

            for match in reversed(matches):
                outer_id_str = match.group(1)
                inner_content = match.group(2)

                outer_id = int(outer_id_str)

                if outer_id not in formatting_map:
                    continue

                outer_info = formatting_map[outer_id]
                outer_tag = outer_info['tag']
                outer_attrs = outer_info['attributes']

                should_remove = False

                if outer_tag == 'span' and not outer_attrs:
                    should_remove = True
                    logger.debug(f"Flattening: <p_{outer_id_str}> is empty <span> → REMOVE")

                inner_match = re.match(r'<p_(\d{2})>', inner_content)
                if inner_match and not should_remove:
                    inner_id = int(inner_match.group(1))

                    if inner_id in formatting_map:
                        inner_info = formatting_map[inner_id]

                        if (outer_tag == inner_info['tag'] and
                            outer_attrs == inner_info['attributes']):
                            should_remove = True
                            logger.debug(f"Flattening: <p_{outer_id_str}> duplicates <p_{inner_id}> → REMOVE")

                if should_remove:
                    text = text[:match.start()] + inner_content + text[match.end():]
                    removed_ids.add(outer_id)

        for tag_id in removed_ids:
            if tag_id in formatting_map:
                del formatting_map[tag_id]
                logger.debug(f"✓ Removed <p_{tag_id:02d}> from formatting_map")

        if removed_ids:
            logger.info(f"✓ Flattened {len(removed_ids)} nested placeholder(s)")

        return text, formatting_map

    def _extract_non_translatable_placeholders(self, text, formatting_map):
        NON_TRANSLATABLE_PATTERN = r'^[\s\.,!?:;…]*$'
        placeholder_pattern = r'<p_(\d{2})>(.*?)</p_\1>'

        non_translatable_map = {}

        def replace_with_marker(match):
            tag_id = int(match.group(1))
            content = match.group(2)

            if re.match(NON_TRANSLATABLE_PATTERN, content):
                non_translatable_map[tag_id] = {
                    'full_match': match.group(0),
                    'content': content
                }
                logger.debug(f"Marked non-translatable: {repr(match.group(0))}")
                return f'<nt_{tag_id:02d}/>'
            else:
                return match.group(0)

        clean_text = re.sub(placeholder_pattern, replace_with_marker, text, flags=re.DOTALL)

        if non_translatable_map:
            logger.info(f"✓ Marked {len(non_translatable_map)} non-translatable placeholder(s) with markers")
            for tag_id, info in non_translatable_map.items():
                logger.debug(f"  p_{tag_id:02d} → <nt_{tag_id:02d}/>: {repr(info['content'])}")

        return clean_text, non_translatable_map

    def _detect_auto_wrap_tags(self, text, formatting_map):
        if not formatting_map:
            return None

        wrap_tags = []
        working_text = text.strip()

        while True:
            match = re.match(r'^<p_(\d{2})>(.*)</p_\1>$', working_text, re.DOTALL)

            if not match:
                break

            elem_id_str = match.group(1)
            elem_id = int(elem_id_str)
            inner_text = match.group(2)

            if elem_id not in formatting_map:
                break

            info = formatting_map[elem_id]
            wrap_tags.append({
                'elem_id': elem_id,
                'opening': info['opening_placeholder'],
                'closing': info['closing_placeholder'],
                'tag': info['tag'],
                'attributes': info['attributes']
            })

            working_text = inner_text.strip()

        if wrap_tags and not re.search(r'</?p_\d{2}>', working_text):
            logger.debug(f"✓ Auto-wrap detected: {len(wrap_tags)} tag(s)")
            for idx, tag_info in enumerate(wrap_tags):
                logger.debug(f"  [{idx}] <{tag_info['tag']}> (id={tag_info['elem_id']})")
            return wrap_tags

        return None

    def _strip_outer_placeholders(self, text, auto_wrap_tags):
        working_text = text.strip()

        for tag_info in auto_wrap_tags:
            opening = tag_info['opening']
            closing = tag_info['closing']

            if working_text.startswith(opening) and working_text.endswith(closing):
                working_text = working_text[len(opening):-len(closing)].strip()
                logger.debug(f"  Stripped {opening}...{closing}")

        return working_text

    def _extract_boundary_reserve_tags(self, text):
        prefix_tags = []
        suffix_tags = []
        clean_text = text

        tag_pattern = r'<id_\d{2}>'

        while True:
            clean_text = clean_text.lstrip()

            match = re.match(tag_pattern, clean_text)
            if match:
                tag = match.group(0)
                prefix_tags.append(tag)
                clean_text = clean_text[len(tag):]
            else:
                break

        while True:
            clean_text = clean_text.rstrip()

            match = re.search(tag_pattern + r'$', clean_text)
            if match:
                tag = match.group(0)
                suffix_tags.insert(0, tag)
                clean_text = clean_text[:-len(tag)]
            else:
                break

        clean_text = clean_text.strip()

        if prefix_tags or suffix_tags:
            logger.debug(f"Extracted boundary tags: prefix={prefix_tags}, suffix={suffix_tags}")

        return prefix_tags, suffix_tags, clean_text

class SRTProcessor(FileProcessor):
    def get_file_type(self) -> str:
        return "srt"

    def load(self, path: str) -> Tuple[List[Dict], None]:
        encodings_to_try = ['utf-8', 'utf-8-sig', 'windows-1250', 'iso-8859-2', 'cp1252', 'latin1']

        content = None
        used_encoding = None

        for encoding in encodings_to_try:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
                used_encoding = encoding
                logger.info(f"Successfully loaded SRT file with encoding: {encoding}")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"Error loading SRT with {encoding}: {e}")
                continue

        if content is None:
            raise ValueError(f"Failed to load SRT file. Tried encodings: {', '.join(encodings_to_try)}")

        try:
            blocks = [block.strip() for block in content.split('\n\n') if block.strip()]
            paragraphs = []

            for block in blocks:
                lines = block.split('\n')
                if len(lines) < 3:
                    continue

                block_number = lines[0].strip()
                timestamp = lines[1].strip()
                text_lines = lines[2:]

                original_lines_with_tags = text_lines

                srt_tags_by_line = []
                clean_lines = []

                for line_idx, line_text in enumerate(text_lines):
                    line_srt_tags = {}
                    clean_line = line_text

                    for match in re.finditer(r'<(i|b|u|font[^>]*)>', line_text):
                        tag = match.group(0)
                        pos = len(re.sub(r'<[^>]+>', '', line_text[:match.start()]))
                        if pos not in line_srt_tags:
                            line_srt_tags[pos] = []
                        line_srt_tags[pos].append(('open', tag))

                    for match in re.finditer(r'</(i|b|u|font)>', line_text):
                        tag = match.group(0)
                        pos = len(re.sub(r'<[^>]+>', '', line_text[:match.start()]))
                        if pos not in line_srt_tags:
                            line_srt_tags[pos] = []
                        line_srt_tags[pos].append(('close', tag))

                    clean_line = re.sub(r'<[^>]+>', '', line_text)
                    clean_lines.append(clean_line)
                    srt_tags_by_line.append(line_srt_tags)

                combined_text = '\n'.join(clean_lines)

                split_positions = []
                current_pos = 0
                for i, line in enumerate(clean_lines):
                    if i < len(clean_lines) - 1:
                        current_pos += len(line)
                        split_positions.append(current_pos)
                        current_pos += 1

                paragraphs.append({
                    'id': block_number,
                    'original_text': combined_text,
                    'translated_text': '',
                    'is_translated': False,
                    'item_href': path,
                    'element_type': 'subtitle_block',
                    'timestamp': timestamp,
                    'subtitle_block': block_number,
                    'has_mismatch': False,
                    'srt_tags_by_line': srt_tags_by_line,
                    'original_clean_lines': clean_lines,
                    'original_line_count': len(clean_lines),
                    'original_split_positions': split_positions,
                    'original_lines_with_tags': original_lines_with_tags
                })

            if not paragraphs:
                raise ValueError("No valid subtitle blocks found in SRT file")

            logger.info(f"Loaded {len(paragraphs)} subtitle blocks from SRT (Encoding: {used_encoding})")

            return paragraphs, None

        except Exception as e:
            logger.error(f"Failed to parse SRT file: {e}", exc_info=True)
            raise

    def _extract_srt_tags(self, line: str) -> Tuple[str, Dict[int, List[Tuple[str, str]]]]:
        tags_dict = {}
        clean_line = line

        tag_pattern = r'<(/?)([biu]|font[^>]*)>'

        matches = list(re.finditer(tag_pattern, line))

        if not matches:
            return line, {}

        offset = 0

        for match in matches:
            is_closing = match.group(1) == '/'
            tag_content = match.group(2)

            if tag_content == 'b':
                tag_type = 'bold'
                tag_value = '</b>' if is_closing else '<b>'
            elif tag_content == 'i':
                tag_type = 'italic'
                tag_value = '</i>' if is_closing else '<i>'
            elif tag_content == 'u':
                tag_type = 'underline'
                tag_value = '</u>' if is_closing else '<u>'
            elif tag_content.startswith('font'):
                tag_type = 'font'
                tag_value = f'</{tag_content}>' if is_closing else f'<{tag_content}>'
            else:
                continue

            pos = match.start() - offset

            if pos not in tags_dict:
                tags_dict[pos] = []
            tags_dict[pos].append((tag_type, tag_value))

            clean_line = clean_line[:match.start() - offset] + clean_line[match.end() - offset:]
            offset += len(match.group(0))

        return clean_line, tags_dict

class TXTProcessor(FileProcessor):
    MAX_CHARS_PER_FRAGMENT = 6000

    def get_file_type(self) -> str:
        return "txt"

    def load(self, path: str) -> Tuple[List[Dict], None]:
        encodings_to_try = ['utf-8', 'utf-8-sig', 'windows-1250', 'iso-8859-2', 'cp1252', 'latin1']

        content = None
        used_encoding = None

        for encoding in encodings_to_try:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
                used_encoding = encoding
                logger.info(f"Successfully loaded TXT file with encoding: {encoding}")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"Error loading TXT with {encoding}: {e}")
                continue

        if content is None:
            raise ValueError(f"Failed to load TXT file. Tried encodings: {', '.join(encodings_to_try)}")

        try:
            paragraphs = []
            fragment_id = 1

            paragraphs_raw = content.split('\n\n')
            paragraphs_raw = [p.strip() for p in paragraphs_raw if p.strip()]

            if len(paragraphs_raw) >= 2:
                logger.info(f"TXT mode: PARAGRAPHS ({len(paragraphs_raw)} blocks)")

                for para_index, text in enumerate(paragraphs_raw, start=1):
                    if len(text) <= self.MAX_CHARS_PER_FRAGMENT:
                        fragments = [text]
                    else:
                        fragments = self._split_long_text(text)

                    for part_index, part in enumerate(fragments, start=1):
                        paragraphs.append({
                            'id': str(fragment_id),
                            'original_text': part,
                            'translated_text': '',
                            'is_translated': False,
                            'item_href': path,
                            'element_type': 'paragraph_part',
                            'paragraph_number': para_index,
                            'has_mismatch': False,
                            'part_index': part_index,
                            'parts_total': len(fragments)
                        })
                        fragment_id += 1

            else:
                lines = content.split('\n')
                lines = [l.strip() for l in lines if l.strip()]

                if len(lines) >= 2:
                    logger.info(f"TXT mode: LINES ({len(lines)} lines)")

                    for line_index, text in enumerate(lines, start=1):
                        if len(text) <= self.MAX_CHARS_PER_FRAGMENT:
                            fragments = [text]
                        else:
                            fragments = self._split_long_text(text)

                        for part_index, part in enumerate(fragments, start=1):
                            paragraphs.append({
                                'id': str(fragment_id),
                                'original_text': part,
                                'translated_text': '',
                                'is_translated': False,
                                'item_href': path,
                                'element_type': 'line_part',
                                'paragraph_number': line_index,
                                'has_mismatch': False,
                                'part_index': part_index,
                                'parts_total': len(fragments)
                            })
                            fragment_id += 1

                else:
                    logger.info("TXT mode: SENTENCES")
                    sentences = self._split_into_sentences(content)

                    for sent_index, text in enumerate(sentences, start=1):
                        text = text.strip()
                        if not text:
                            continue

                        if len(text) <= self.MAX_CHARS_PER_FRAGMENT:
                            fragments = [text]
                        else:
                            fragments = self._split_long_text(text)

                        for part_index, part in enumerate(fragments, start=1):
                            paragraphs.append({
                                'id': str(fragment_id),
                                'original_text': part,
                                'translated_text': '',
                                'is_translated': False,
                                'item_href': path,
                                'element_type': 'sentence_part',
                                'paragraph_number': sent_index,
                                'has_mismatch': False,
                                'part_index': part_index,
                                'parts_total': len(fragments)
                            })
                            fragment_id += 1

            if not paragraphs:
                logger.warning("TXT mode: FALLBACK (single block)")
                paragraphs.append({
                    'id': '1',
                    'original_text': content.strip(),
                    'translated_text': '',
                    'is_translated': False,
                    'item_href': path,
                    'element_type': 'full_text',
                    'paragraph_number': 1,
                    'has_mismatch': False,
                    'part_index': 1,
                    'parts_total': 1
                })

            logger.info(f"Loaded {len(paragraphs)} fragments (Encoding: {used_encoding})")
            return paragraphs, None

        except Exception as e:
            logger.error(f"Failed to parse TXT file: {e}", exc_info=True)
            raise

    def _split_into_sentences(self, text: str) -> List[str]:
        ABBREVIATIONS = {

            'np', 'itd', 'itp', 'tzw', 'ok', 'ul', 'al', 'pl', 'wg', 'vs',
            'nr', 'str', 'mgr', 'dr', 'prof', 'inż', 'lic', 'hab',
            'mln', 'mld', 'tys', 'godz', 'min', 'sek', 'zł', 'gr',
            'gen', 'ppłk', 'płk', 'mjr', 'kpt', 'por', 'szer',

            'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'rev', 'gen',
            'sgt', 'cpl', 'pvt', 'capt', 'col', 'lt', 'gov', 'pres',
            'dept', 'est', 'approx', 'avg', 'max', 'min', 'vol', 'no',
            'fig', 'eq', 'vs', 'etc', 'inc', 'corp', 'co', 'ltd',
            'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep',
            'oct', 'nov', 'dec',

            'bzw', 'ca', 'dh', 'evtl', 'ggf', 'usw', 'vgl', 'zb', 'zbsp',
            'str', 'nr', 'hr', 'fr', 'dr', 'prof', 'dipl',

            'mme', 'mlle', 'mm', 'dr', 'pr', 'me', 'ste', 'st',
            'bd', 'av', 'pl', 'sq', 'env', 'nb', 'cf', 'ex',

            'sr', 'sra', 'srta', 'dr', 'dra', 'prof', 'lic',
            'num', 'pag', 'cap', 'art', 'fig', 'ed',

            'sig', 'dott', 'prof', 'ing', 'avv', 'geom', 'arch',

            'al', 'et', 'ibid', 'op', 'loc', 'cit', 'viz',
        }

        raw_parts = re.split(r'([.!?…]+)', text)

        sentences = []
        current = ''

        i = 0
        while i < len(raw_parts):
            part = raw_parts[i]

            if i % 2 == 0:
                current += part
                i += 1
            else:
                separator = part

                next_part = raw_parts[i + 1] if i + 1 < len(raw_parts) else ''

                if separator == '.' and next_part:
                    prev_word_match = re.search(r'(\w+)$', current, re.UNICODE)

                    if prev_word_match:
                        prev_word = prev_word_match.group(1).lower()

                        if prev_word in ABBREVIATIONS:
                            current += separator
                            i += 1
                            continue

                        if len(prev_word) == 1:
                            current += separator
                            i += 1
                            continue

                        if prev_word.isdigit():
                            next_stripped = next_part.lstrip()
                            first_char = next_stripped[0] if next_stripped else ''
                            if first_char and not first_char.isupper():
                                current += separator
                                i += 1
                                continue

                if next_part and not next_part.startswith((' ', '\n', '\t')):
                    current += separator
                    i += 1
                    continue

                current += separator
                sentence = current.strip()
                if sentence:
                    sentences.append(sentence)
                current = ''
                i += 1

        if current.strip():
            sentences.append(current.strip())

        if len(sentences) < 2:
            logger.warning("Sentence splitter: fallback to simple split")
            sentences = [s.strip() for s in re.split(r'(?<=[.!?…])\s+', text) if s.strip()]

        logger.debug(f"Split into {len(sentences)} sentences")
        return sentences

    def _split_long_text(self, text: str) -> List[str]:
        chunks = []
        current_chunk = ''

        sentences = self._split_into_sentences(text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) <= self.MAX_CHARS_PER_FRAGMENT:
                separator = ' ' if current_chunk else ''
                if len(current_chunk) + len(separator) + len(sentence) <= self.MAX_CHARS_PER_FRAGMENT:
                    current_chunk += separator + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
                continue

            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ''

            words = sentence.split()

            for word in words:
                if len(word) > self.MAX_CHARS_PER_FRAGMENT:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ''
                    for i in range(0, len(word), self.MAX_CHARS_PER_FRAGMENT):
                        chunks.append(word[i:i + self.MAX_CHARS_PER_FRAGMENT])
                    continue

                separator = ' ' if current_chunk else ''
                if len(current_chunk) + len(separator) + len(word) <= self.MAX_CHARS_PER_FRAGMENT:
                    current_chunk += separator + word
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

class FileProcessorFactory:
    @staticmethod
    def create_processor(file_type: str, app_settings: dict) -> FileProcessor:
        if file_type == "epub":
            return EPUBProcessor(app_settings)
        elif file_type == "srt":
            return SRTProcessor()
        elif file_type == "txt":
            return TXTProcessor()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
