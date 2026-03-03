# app_utils.py
import copy
import json
import logging
import os
from lxml import etree
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class SessionManager:
    @staticmethod
    def save_session(
        path: str,
        paragraphs: List[Dict],
        original_file_path: str,
        file_type: str,
        app_settings: Dict,
        context_before: int,
        context_after: int,
        temperature: float,
        custom_prompts: Dict[str, str],
        single_prompt_mode: bool,
        processing_mode: str,
        prompt_variant: str,
        json_payload_mode: bool = False,
        json_payload_template: str = "",
        json_response_field: str = ""
    ) -> None:
        paragraphs_to_save = []
        for para in paragraphs:
            para_copy = para.copy()

            if 'auto_fix_history' in para_copy:
                del para_copy['auto_fix_history']

            para_copy['processing_mode'] = processing_mode
            paragraphs_to_save.append(para_copy)

        skip_inline_tags = app_settings.get('skip_inline_tags', {})

        session_data = {
            'original_file_path': original_file_path,
            'file_type': file_type,
            'paragraphs': paragraphs_to_save,
            'context_before': context_before,
            'context_after': context_after,
            'temperature': temperature,
            'custom_ollama_prompt': custom_prompts.get('ollama'),
            'custom_system_prompt': custom_prompts.get('system'),
            'custom_assistant_prompt': custom_prompts.get('assistant'),
            'custom_user_prompt': custom_prompts.get('user'),
            'single_prompt_mode': single_prompt_mode,
            'json_payload_mode': json_payload_mode,
            'json_payload_template': json_payload_template,
            'json_response_field': json_response_field,

            'metadata': {
                'processing_mode': processing_mode,
                'prompt_variant': prompt_variant,
                'skip_inline_tags': skip_inline_tags
            }
        }

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Session saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            raise

    @staticmethod
    def load_session(session_path: str) -> Dict:
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            required_fields = ['original_file_path', 'file_type', 'paragraphs']
            for field in required_fields:
                if field not in session_data:
                    raise KeyError(f"Missing required field: {field}")

            metadata = session_data.get('metadata', {})
            processing_mode = metadata.get('processing_mode', 'inline')
            prompt_variant = metadata.get('prompt_variant', None)

            skip_inline_tags = metadata.get('skip_inline_tags')

            if skip_inline_tags is None and 'skip_spans' in metadata:
                old_skip_spans = metadata.get('skip_spans', False)

                skip_inline_tags = {'span': old_skip_spans} if old_skip_spans else {}
                logger.info(f"Converted old skip_spans={old_skip_spans} to skip_inline_tags={skip_inline_tags}")
            elif skip_inline_tags is None:
                skip_inline_tags = {}

            custom_prompts = {
                'ollama': session_data.get('custom_ollama_prompt'),
                'system': session_data.get('custom_system_prompt'),
                'assistant': session_data.get('custom_assistant_prompt'),
                'user': session_data.get('custom_user_prompt')
            }

            if 'context_size' in session_data and 'context_before' not in session_data:
                context_before = session_data['context_size']
                context_after = 0
            else:
                context_before = session_data.get('context_before', 3)
                context_after = session_data.get('context_after', 2)

            result = {
                'original_file_path': session_data['original_file_path'],
                'file_type': session_data['file_type'],
                'paragraphs': session_data['paragraphs'],
                'context_before': context_before,
                'context_after': context_after,
                'temperature': session_data.get('temperature', 0.8),
                'custom_prompts': custom_prompts,
                'single_prompt_mode': session_data.get('single_prompt_mode', False),
                'json_payload_mode': session_data.get('json_payload_mode', False),
                'json_payload_template': session_data.get('json_payload_template', ''),
                'json_response_field': session_data.get('json_response_field', ''),
                'metadata': {
                    'processing_mode': processing_mode,
                    'prompt_variant': prompt_variant,
                    'skip_inline_tags': skip_inline_tags
                }
            }

            logger.info(f"Session loaded from: {session_path}")
            logger.info(f"Mode: {processing_mode}, Variant: {prompt_variant}, skip_inline_tags: {skip_inline_tags}")

            return result

        except FileNotFoundError:
            logger.error(f"Session file not found: {session_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in session file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            raise

    @staticmethod
    def restore_epub_ids(
        elements_tree: Any,
        item_href: str,
        session_map: Dict[Tuple[str, str], Dict],
        namespaces: Dict[str, str]
    ) -> None:
        for element in elements_tree:
            if not isinstance(element, etree._Element):
                continue

            text = SessionManager._get_element_text_lxml(element)
            if not text or not text.strip():
                continue

            key = (item_href, text)
            if key in session_map:
                para = session_map[key]

                element.set('id', para['id'])
                logger.debug(f"Restored ID: {para['id']} for text: {text[:50]}...")

            SessionManager.restore_epub_ids(element, item_href, session_map, namespaces)

    @staticmethod
    def _get_element_text_lxml(element: Any) -> str:
        text_content = etree.tostring(
            element,
            encoding='unicode',
            method='text'
        )

        return ' '.join(text_content.split())

class AppSettingsManager:
    SETTINGS_FILE = "app_settings.json"

    DEFAULT_SETTINGS = {
        "llm_choice": "LM Studio",
        "server_url": "",
        "json_response_field": "",
        "ollama_model_name": "",
        "openrouter_api_key": "",
        "openrouter_model_name": "",
        "ollama_endpoint": "http://localhost:11434",
        "deepl_free_api_key": "",
        "deepl_pro_api_key": "",
        "use_inline_formatting": True,
        "restore_paragraph_structure": True,
        "show_ps_in_ui": False,
        "skip_inline_tags": {},
        "mismatch_checks": {
            "paragraphs": True,
            "first_char": True,
            "last_char": True,
            "length": True,
            "quote_parity": True,
            "untranslated": True,
            "reserve_elements": True,
            "nt_markers": True,
            "inline_formatting": True,
        },
        "mismatch_thresholds": {
            "length_ratio_short": 1.6,
            "length_ratio_medium": 1.4,
            "length_ratio_long": 1.3,
            "untranslated_ratio": 0.3,
            "position_shift_threshold": 0.15,
            "inline_position_shift_threshold": 0.30,
        },
        "alignment_settings": {
            "device":     "cpu",
            "model_name": "xlm-roberta-base",
        },
    }

    @staticmethod
    def load_settings() -> Dict:
        try:
            with open(AppSettingsManager.SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
            logger.info("App settings loaded successfully")
        except FileNotFoundError:
            logger.info("App settings file not found, using defaults")
            settings = {}
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load app settings: {e}, using defaults")
            settings = {}

        for key, val in AppSettingsManager.DEFAULT_SETTINGS.items():
            settings.setdefault(key, val)

        if 'use_ps_markers' in settings:
            if 'restore_paragraph_structure' not in settings:
                settings['restore_paragraph_structure'] = settings['use_ps_markers']
            del settings['use_ps_markers']
            logger.info("Migrated use_ps_markers -> restore_paragraph_structure")

        for obsolete in ('restore_paragraph_epub', 'restore_paragraph_txt'):
            if obsolete in settings:
                del settings[obsolete]
                logger.info(f"Removed obsolete key: {obsolete}")

        for nested_key in (
            "mismatch_checks",
            "mismatch_thresholds",
            "skip_inline_tags",
            "alignment_settings",
        ):
            default_nested = AppSettingsManager.DEFAULT_SETTINGS.get(nested_key, {})
            if not isinstance(settings.get(nested_key), dict):
                settings[nested_key] = default_nested.copy()
            else:
                for k, v in default_nested.items():
                    settings[nested_key].setdefault(k, v)

                settings[nested_key].pop('ps_markers', None)

        try:
            with open(AppSettingsManager.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save merged settings: {e}")

        return settings

    @staticmethod
    def save_settings(settings: Dict) -> None:
        validated_settings = {}

        for key, default_val in AppSettingsManager.DEFAULT_SETTINGS.items():
            if isinstance(default_val, dict):
                src = settings.get(key, {})
                merged = default_val.copy()
                merged.update(src)

                for obsolete_key in list(merged.keys()):
                    if obsolete_key not in default_val:
                        merged.pop(obsolete_key, None)
                validated_settings[key] = merged
            else:
                validated_settings[key] = settings.get(key, default_val)

        for obsolete in ('use_ps_markers', 'restore_paragraph_epub', 'restore_paragraph_txt'):
            validated_settings.pop(obsolete, None)

        try:
            with open(AppSettingsManager.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(validated_settings, f, ensure_ascii=False, indent=2)
            logger.info("App settings saved successfully")
        except Exception as e:
            logger.error(f"Failed to save app settings: {e}")
            raise

    @staticmethod
    def get_llm_client_config(settings: Dict) -> Dict:
        llm_choice = settings.get('llm_choice', 'LM Studio')

        if llm_choice == 'Ollama':
            model_name = settings.get('ollama_model_name', '')
            api_key = None
            endpoint = settings.get('ollama_endpoint', 'http://localhost:11434')
        elif llm_choice == 'Openrouter':
            model_name = settings.get('openrouter_model_name', '')
            api_key = settings.get('openrouter_api_key', '')
            endpoint = 'https://openrouter.ai/api/v1'
        else:
            model_name = 'local-model'
            api_key = None
            endpoint = 'http://localhost:1234/v1'

        return {
            'llm_choice': llm_choice,
            'model_name': model_name,
            'api_key': api_key,
            'endpoint': endpoint
        }

class PromptManager:
    @staticmethod
    def get_prompt(variant: str, role: str) -> str:
        filename = f"llm_prompt_{role}_{variant}.txt"

        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                logger.info(f"Loaded custom prompt: {filename}")
                return content
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}, using factory")

        if role == 'system':
            return PromptManager.get_default_system_prompt(variant)
        elif role == 'assistant':
            return PromptManager.get_default_assistant_prompt(variant)
        elif role == 'user':
            return PromptManager.get_default_user_prompt(variant)
        elif role == 'ollama':
            return PromptManager.get_default_ollama_prompt(variant)
        else:
            logger.error(f"Unknown prompt role: {role}")
            return ""

    @staticmethod
    def save_prompt(variant: str, role: str, content: str) -> str:
        filename = f"llm_prompt_{role}_{variant}.txt"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved prompt: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            raise

    @staticmethod
    def reset_to_factory(variant: str) -> None:
        logger.info(f"Factory reset requested for variant: {variant} (files preserved)")

    @staticmethod
    def hard_reset(variant: str) -> int:
        return PromptManager.delete_prompts_for_variant(variant)

    @staticmethod
    def load_prompts_for_variant(variant: str) -> Dict[str, str]:
        result = {}

        system_file = f"llm_prompt_system_{variant}.txt"
        if os.path.exists(system_file):
            try:
                with open(system_file, 'r', encoding='utf-8') as f:
                    result['system'] = f.read().strip()
                logger.info(f"Loaded custom system prompt: {system_file}")
            except Exception as e:
                logger.warning(f"Failed to load {system_file}: {e}, using factory")
                result['system'] = PromptManager.get_default_system_prompt(variant)
        else:
            result['system'] = PromptManager.get_default_system_prompt(variant)

        assistant_file = f"llm_prompt_assistant_{variant}.txt"
        if os.path.exists(assistant_file):
            try:
                with open(assistant_file, 'r', encoding='utf-8') as f:
                    result['assistant'] = f.read().strip()
                logger.info(f"Loaded custom assistant prompt: {assistant_file}")
            except Exception as e:
                logger.warning(f"Failed to load {assistant_file}: {e}, using factory")
                result['assistant'] = PromptManager.get_default_assistant_prompt(variant)
        else:
            result['assistant'] = PromptManager.get_default_assistant_prompt(variant)

        user_file = f"llm_prompt_user_{variant}.txt"
        if os.path.exists(user_file):
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    result['user'] = f.read().strip()
                logger.info(f"Loaded custom user prompt: {user_file}")
            except Exception as e:
                logger.warning(f"Failed to load {user_file}: {e}, using factory")
                result['user'] = PromptManager.get_default_user_prompt(variant)
        else:
            result['user'] = PromptManager.get_default_user_prompt(variant)

        ollama_file = f"llm_prompt_ollama_{variant}.txt"
        if os.path.exists(ollama_file):
            try:
                with open(ollama_file, 'r', encoding='utf-8') as f:
                    result['ollama'] = f.read().strip()
                logger.info(f"Loaded custom Ollama prompt: {ollama_file}")
            except Exception as e:
                logger.warning(f"Failed to load {ollama_file}: {e}, using factory")
                result['ollama'] = PromptManager.get_default_ollama_prompt(variant)
        else:
            result['ollama'] = PromptManager.get_default_ollama_prompt(variant)

        json_payload_file = f"llm_prompt_json_payload_{variant}.txt"
        if os.path.exists(json_payload_file):
            try:
                with open(json_payload_file, 'r', encoding='utf-8') as f:
                    result['json_payload'] = f.read().strip()
                logger.info(f"Loaded JSON payload template: {json_payload_file}")
            except Exception as e:
                logger.warning(f"Failed to load {json_payload_file}: {e}, using factory")
                result['json_payload'] = PromptManager.get_default_json_payload_prompt(variant)
        else:
            result['json_payload'] = PromptManager.get_default_json_payload_prompt(variant)

        return result

    @staticmethod
    def get_default_system_prompt(variant: str) -> str:
        base = """You are a professional translator from English to Polish.

1. Return ONLY translated text in this format:
<translated>Your translation here</translated>

2. CRITICAL: No text before/after tags. No explanations.

3. Translate ONLY text inside <text_to_translate> tags.
Context in <context_before>/<context_after> is for reference only."""

        if variant == "txt":
            return base + """

4. Preserve natural paragraph breaks and text structure.

5. Keep URLs, emails, proper nouns as-is.

6. Maintain author's meaning and natural Polish flow."""

        elif variant == "srt":
            return base + """

4. LINE PRESERVATION:
- Maintain line breaks (\\n) from original
- If original has 2 lines, translation must have 2 lines
- Split text naturally at phrase boundaries

5. DIALOGUE DASHES:
- Keep dialogue dashes (- ) at line start if present in original

6. Keep URLs, emails, proper nouns as-is.

7. Maintain natural Polish subtitle flow."""

        elif variant == "epub_legacy":
            return base + """

4. RESERVE ELEMENTS (structural):
Format: <id_00>, <id_01>, etc.
These mark images, <br> tags, non-translatable elements.
- Copy EXACTLY as shown
- Keep approximate position in text
- Do NOT modify or remove

5. Keep URLs, emails, proper nouns as-is.

6. Maintain author's meaning and natural Polish flow."""

        elif variant == "epub_inline":
            return base + """

4. PLACEHOLDERS — copy all exactly, preserve positions:
- <id_00>, <id_01>… — images/<br>/structural (do NOT modify)
- <p_00>…</p_00>… — inline formatting (<i>/<b>/etc.); every opening needs matching closing, keep around same words
- <nt_00/>, <nt_01/>… — non-translatable spaces/anchors (self-closing, do NOT move)
- <ps> — paragraph break; count in translation must match original EXACTLY

5. Keep URLs, emails, proper nouns as-is.
6. Maintain author's meaning and natural Polish flow."""

        return base

    @staticmethod
    def get_default_assistant_prompt(variant: str) -> str:
        base = """Context for reference only. Do NOT translate.

<context_before>{context_before}</context_before>

<context_after>{context_after}</context_after>

I understand. I will translate ONLY <text_to_translate>, using context for narrative consistency."""

        if variant == "txt":
            return base

        elif variant == "srt":
            return base

        elif variant == "epub_legacy":
            return base

        elif variant == "epub_inline":
            return base

        return base

    @staticmethod
    def get_default_user_prompt(variant: str) -> str:
        base = """<text_to_translate>{core_text}</text_to_translate>

Translate FULL text inside <text_to_translate> tags. Do not skip, shorten, or omit any part. Keep all placeholders and formatting intact, and preserve the tone, including any strong language.

Return format:
<translated>Translation here</translated>"""

        if variant == "txt":
            return base

        elif variant == "srt":
            return base

        elif variant == "epub_inline":
            return base

        return base

    @staticmethod
    def get_default_ollama_prompt(variant: str) -> str:
        system = PromptManager.get_default_system_prompt(variant)
        assistant = PromptManager.get_default_assistant_prompt(variant)
        user = PromptManager.get_default_user_prompt(variant)

        return f"""{system}

{assistant}

{user}"""

    @staticmethod
    def get_default_json_payload_prompt(variant: str) -> str:
        base_instructions = (
            "Translate the following text to Polish.\\n"
            "Return ONLY valid JSON in this exact format: {\\\"translation\\\":\\\"...\\\"}\\n"
            "Do not add any explanations, comments, markdown or extra text outside the JSON.\\n\\n"
        )

        if variant == "txt":
            variant_instructions = (
                "Preserve natural paragraph breaks and text structure.\\n"
                "Keep URLs, emails and proper nouns as-is.\\n"
                "Maintain the author's meaning and natural Polish flow.\\n\\n"
            )

        elif variant == "srt":
            variant_instructions = (
                "LINE PRESERVATION: maintain line breaks (\\\\n) from the original.\\n"
                "If the original has 2 lines, the translation must have exactly 2 lines.\\n"
                "Keep dialogue dashes (- ) at line start if present in original.\\n"
                "Keep URLs, emails and proper nouns as-is.\\n\\n"
            )

        elif variant == "epub_legacy":
            variant_instructions = (
                "RESERVE ELEMENTS — copy exactly, keep approximate position:\\n"
                "  <id_00>, <id_01>, ... — images, <br> tags, non-translatable elements.\\n"
                "Do NOT modify or remove them.\\n"
                "Keep URLs, emails and proper nouns as-is.\\n\\n"
            )

        elif variant == "epub_inline":
            variant_instructions = (
                "PLACEHOLDERS — copy all exactly and preserve their positions:\\n"
                "  <id_00>, <id_01>... — images/<br>/structural (do NOT modify)\\n"
                "  <p_00>...</p_00>... — inline formatting (<i>/<b>/etc.); every opening needs matching closing\\n"
                "  <nt_00/>, <nt_01/>... — non-translatable anchors (self-closing, do NOT move)\\n"
                "  <ps> — paragraph break; count in translation must match original EXACTLY\\n"
                "NEVER use raw HTML tags like <i>, <b>, <em> — use only <p_XX> placeholders.\\n"
                "Keep URLs, emails and proper nouns as-is.\\n\\n"
            )

        else:
            variant_instructions = ""

        auto_fix_ready = (
            "IMPORTANT: You may receive additional instructions starting with:\\n"
            "=== AUTO-FIX INSTRUCTIONS - CRITICAL ===\\n"
            "If you see them, you MUST fix the listed issues in your previous translation.\\n"
            "Then return the corrected translation in the exact same JSON format.\\n"
            "Do not mention the auto-fix instructions in your response.\\n\\n"
        )

        content = (
            base_instructions
            + variant_instructions
            + auto_fix_ready
            + "Context before (for reference only, do NOT translate):\\n"
            "{context_before}\\n\\n"
            "Text to translate:\\n"
            "{core_text}\\n\\n"
            "Context after (for reference only, do NOT translate):\\n"
            "{context_after}"
        )

        return (
            '{\n'
            '    "messages": [\n'
            '        {\n'
            '            "role": "user",\n'
            f'            "content": "{content}"\n'
            '        }\n'
            '    ]\n'
            '}'
        )

    @staticmethod
    def get_default_json_response_field(variant: str) -> str:
        return ""

    @staticmethod
    def save_prompts_for_variant(
        variant: str,
        system: str = None,
        assistant: str = None,
        user: str = None,
        ollama: str = None
    ) -> List[str]:
        saved_files = []

        try:
            if system is not None:
                filename = f"llm_prompt_system_{variant}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(system)
                saved_files.append(filename)
                logger.info(f"Saved: {filename}")

            if assistant is not None:
                filename = f"llm_prompt_assistant_{variant}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(assistant)
                saved_files.append(filename)
                logger.info(f"Saved: {filename}")

            if user is not None:
                filename = f"llm_prompt_user_{variant}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(user)
                saved_files.append(filename)
                logger.info(f"Saved: {filename}")

            if ollama is not None:
                filename = f"llm_prompt_ollama_{variant}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(ollama)
                saved_files.append(filename)
                logger.info(f"Saved: {filename}")

            return saved_files

        except Exception as e:
            logger.error(f"Failed to save prompts for variant {variant}: {e}")
            raise

    @staticmethod
    def delete_prompts_for_variant(variant: str) -> int:
        files_to_delete = [
            f"llm_prompt_system_{variant}.txt",
            f"llm_prompt_assistant_{variant}.txt",
            f"llm_prompt_user_{variant}.txt",
            f"llm_prompt_ollama_{variant}.txt",
            f"llm_prompt_json_payload_{variant}.txt",
        ]

        deleted_count = 0
        for filename in files_to_delete:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    deleted_count += 1
                    logger.info(f"Deleted: {filename}")
            except Exception as e:
                logger.error(f"Failed to delete {filename}: {e}")

        return deleted_count

class EPUBConstants:
    PRIORITY_TAGS = ['p', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote']

    NON_INLINE_ELEMENTS = {
        'address', 'blockquote', 'dialog', 'div', 'figure', 'figcaption',
        'footer', 'header', 'legend', 'main', 'p', 'pre', 'search', 'article',
        'aside', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hgroup', 'nav',
        'section', 'dd', 'dl', 'dt', 'menu', 'ol', 'ul', 'table', 'caption',
        'colgroup', 'col', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th', 'li'
    }

    CONTAINER_TAGS = {
        'ul', 'ol', 'dl',
        'table', 'tbody', 'thead', 'tfoot', 'tr',
        'div', 'section', 'article', 'aside', 'nav', 'main',
        'header', 'footer', 'figure', 'body'
    }

    NOISE_TAGS = ['rt', 'rp']

    RESERVE_TAGS = [
        'img', 'code', 'br', 'hr', 'sub', 'sup', 'kbd',
        'abbr', 'wbr', 'var', 'canvas', 'svg', 'script',
        'style', 'math'
    ]

    INLINE_FORMATTING_TAGS = ['i', 'b', 'em', 'strong', 'u', 'sup', 'sub', 'small', 'span']

    STRUCTURAL_SPAN_CLASSES = {
        'first-letter',
        'last-word',
        'item-number',
        'element-number',
    }

class SRTConstants:
    MAX_LINE_LENGTH = 42

    MAX_LINES_PER_BLOCK = 2

    DIALOGUE_DASHES = ['-', '–', '—']

class QuoteConstants:
    ALL_QUOTES = r'["\'\u2018\u2019\u201C\u201D\u201E\u201F\u00AB\u00BB\u2039\u203A\u201A\u201B\u2032\u2033\u301D\u301E\u301F\uFF02]'

    DOUBLE_QUOTES = r'["\u201C\u201D\u201E\u201F\u00AB\u00BB\u301D\u301E\u301F\uFF02]'

    SINGLE_QUOTES = r'[\'\u2018\u2019\u201A\u201B\u2032\u2039\u203A]'

    POLISH_OPENING = '\u201e'
    POLISH_CLOSING = '\u201d'

class LanguageConstants:
    SOURCE_LANGUAGES = [
        ("Auto", None),
        ("Bulgarian", "BG"),
        ("Czech", "CS"),
        ("Danish", "DA"),
        ("German", "DE"),
        ("Greek", "EL"),
        ("English", "EN"),
        ("Spanish", "ES"),
        ("Estonian", "ET"),
        ("Finnish", "FI"),
        ("French", "FR"),
        ("Hungarian", "HU"),
        ("Indonesian", "ID"),
        ("Italian", "IT"),
        ("Japanese", "JA"),
        ("Korean", "KO"),
        ("Lithuanian", "LT"),
        ("Latvian", "LV"),
        ("Norwegian (Bokmål)", "NB"),
        ("Dutch", "NL"),
        ("Polish", "PL"),
        ("Portuguese", "PT"),
        ("Romanian", "RO"),
        ("Russian", "RU"),
        ("Slovak", "SK"),
        ("Slovenian", "SL"),
        ("Swedish", "SV"),
        ("Turkish", "TR"),
        ("Ukrainian", "UK"),
        ("Chinese", "ZH"),
    ]

    TARGET_LANGUAGES = [
        ("Bulgarian", "BG"),
        ("Czech", "CS"),
        ("Danish", "DA"),
        ("German", "DE"),
        ("Greek", "EL"),
        ("English", "EN"),
        ("English (British)", "EN-GB"),
        ("English (American)", "EN-US"),
        ("Spanish", "ES"),
        ("Estonian", "ET"),
        ("Finnish", "FI"),
        ("French", "FR"),
        ("Hungarian", "HU"),
        ("Indonesian", "ID"),
        ("Italian", "IT"),
        ("Japanese", "JA"),
        ("Korean", "KO"),
        ("Lithuanian", "LT"),
        ("Latvian", "LV"),
        ("Norwegian (Bokmål)", "NB"),
        ("Dutch", "NL"),
        ("Polish", "PL"),
        ("Portuguese", "PT"),
        ("Portuguese (Portugal)", "PT-PT"),
        ("Portuguese (Brazil)", "PT-BR"),
        ("Romanian", "RO"),
        ("Russian", "RU"),
        ("Slovak", "SK"),
        ("Slovenian", "SL"),
        ("Swedish", "SV"),
        ("Turkish", "TR"),
        ("Ukrainian", "UK"),
        ("Chinese", "ZH"),
    ]

__all__ = [
    'SessionManager',
    'AppSettingsManager',
    'PromptManager',
    'EPUBConstants',
    'SRTConstants',
    'QuoteConstants',
    'LanguageConstants'
]
