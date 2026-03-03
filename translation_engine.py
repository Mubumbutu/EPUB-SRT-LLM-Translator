# translation_engine.py
import json
import logging
import re
import requests
import time
import threading
from abc import ABC, abstractmethod
from openrouter import OpenRouter
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class LLMClient(ABC):
    @abstractmethod
    def translate(
        self,
        prompt: Union[str, List[Dict]],
        temperature: float,
        timeout_seconds: int
    ) -> str:
        pass

class LMStudioClient(LLMClient):
    def __init__(self, endpoint: str = "http://localhost:1234/v1/chat/completions"):
        self.endpoint = endpoint
        self._active_session = None
        self._session_lock = threading.Lock()
        logger.info(f"Initialized LM Studio client: {endpoint}")

    def abort(self):
        with self._session_lock:
            if self._active_session is not None:
                try:
                    self._active_session.close()
                    logger.info("LMStudioClient: sesja zamknięta (hard cancel)")
                except Exception as e:
                    logger.warning(f"LMStudioClient: błąd zamykania sesji: {e}")
                self._active_session = None

    def translate(
        self,
        prompt,
        temperature: float,
        timeout_seconds: int
    ) -> str:
        headers = {"Content-Type": "application/json"}

        session = requests.Session()
        with self._session_lock:
            self._active_session = session

        try:
            if isinstance(prompt, dict) and "__json_payload__" in prompt:
                payload = prompt["__json_payload__"]
                response_field = prompt.get("__response_field__", "").strip()

                payload["temperature"] = temperature

                logger.debug(f"JSON payload request: timeout={timeout_seconds}s, response_field='{response_field}'")

                response = session.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=timeout_seconds
                )
                response.raise_for_status()
                data = response.json()

                if response_field:
                    try:
                        parts = response_field.split(".")
                        val = data
                        for part in parts:
                            if isinstance(val, str):
                                try:
                                    val = json.loads(val)
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            if isinstance(val, list):
                                val = val[int(part)]
                            else:
                                val = val[part]
                        if isinstance(val, str):
                            content = val.strip()
                        else:
                            content = json.dumps(val, ensure_ascii=False).strip()
                    except (KeyError, IndexError, ValueError, TypeError) as e:
                        raw = json.dumps(data)[:400]
                        raise ValueError(
                            f"Could not extract field '{response_field}' from server response.\n"
                            f"Check the 'Response field' setting.\n"
                            f"Raw response: {raw}"
                        )
                else:
                    choices = data.get("choices", [])
                    if choices:
                        raw_content = choices[0].get("message", {}).get("content", "")
                    else:
                        raw_content = (
                            data.get("response", "")
                            or data.get("translation", "")
                            or data.get("text", "")
                        )

                    if raw_content:
                        stripped = raw_content.strip()
                        if stripped.startswith("{") and stripped.endswith("}"):
                            try:
                                parsed = json.loads(stripped)
                                translation_keys = [
                                    "translation", "Translation", "TRANSLATION",
                                    "text", "Text", "result", "Result",
                                    "output", "Output", "translated", "Translated"
                                ]
                                extracted = None
                                for k in translation_keys:
                                    if k in parsed:
                                        extracted = str(parsed[k]).strip()
                                        break
                                content = extracted if extracted else stripped
                            except (json.JSONDecodeError, ValueError):
                                content = stripped
                        else:
                            content = stripped
                    else:
                        content = ""

                if not content:
                    raw = json.dumps(data)[:400]
                    raise ValueError(
                        f"Server returned 200 OK but the extracted content is empty.\n"
                        f"Check the 'Response field' setting or verify the model output format.\n"
                        f"Raw response: {raw}"
                    )

                logger.debug(f"JSON payload response: {len(content)} chars")
                return content

            else:
                payload = {
                    "messages": prompt,
                    "temperature": temperature,
                    "max_tokens": -1,
                    "stream": False
                }

                logger.debug(f"LM Studio request: temp={temperature}, timeout={timeout_seconds}s")

                response = session.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=timeout_seconds
                )
                response.raise_for_status()
                data = response.json()

                choices = data.get("choices", [])
                if not choices:
                    raw = json.dumps(data)[:400]
                    raise ValueError(
                        f"Server returned 200 OK but 'choices' is empty or missing.\n"
                        f"Possible cause: model is not loaded or the server does not support the chat/completions endpoint.\n"
                        f"Raw response: {raw}"
                    )

                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raw = json.dumps(data)[:400]
                    raise ValueError(
                        f"Server returned 200 OK but 'content' is empty.\n"
                        f"Make sure the model is fully loaded and supports the chat/completions format.\n"
                        f"Raw response: {raw}"
                    )

                logger.debug(f"LM Studio response: {len(content)} chars")
                return content

        except requests.Timeout:
            logger.error(f"LM Studio timeout after {timeout_seconds}s")
            raise TimeoutError(f"LM Studio request timed out after {timeout_seconds}s")

        except requests.RequestException as e:
            logger.error(f"LM Studio request failed: {e}")
            raise

        finally:
            with self._session_lock:
                if self._active_session is session:
                    self._active_session = None

class OllamaClient(LLMClient):
    def __init__(
        self,
        model_name: str,
        endpoint: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.endpoint = endpoint
        self.api_url = f"{endpoint}/api/generate"
        self._active_session = None
        self._session_lock = threading.Lock()
        logger.info(f"Initialized Ollama client: {endpoint} (model: {model_name})")

    def abort(self):
        with self._session_lock:
            if self._active_session is not None:
                try:
                    self._active_session.close()
                    logger.info("OllamaClient: sesja zamknięta (hard cancel)")
                except Exception as e:
                    logger.warning(f"OllamaClient: błąd zamykania sesji: {e}")
                self._active_session = None

    def translate(
        self,
        prompt: str,
        temperature: float,
        timeout_seconds: int
    ) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}

        session = requests.Session()
        with self._session_lock:
            self._active_session = session

        try:
            logger.debug(f"Ollama request: model={self.model_name}, temp={temperature}, timeout={timeout_seconds}s")
            response = session.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=timeout_seconds
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("response", "")
            if not content:
                logger.warning("Ollama returned empty response")
                return ""
            logger.debug(f"Ollama response: {len(content)} chars")
            return content

        except requests.Timeout:
            logger.error(f"Ollama timeout after {timeout_seconds}s")
            raise TimeoutError(f"Ollama request timed out after {timeout_seconds}s")

        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise

        finally:
            with self._session_lock:
                if self._active_session is session:
                    self._active_session = None

class OpenRouterClient(LLMClient):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

        self.is_free_model = model_name.strip().endswith(":free")

        self._last_request_time: Optional[float] = None
        self._min_interval = 5.0

        logger.info(
            f"Initialized OpenRouter client (model: {model_name}, "
            f"free_model: {self.is_free_model})"
        )

    def _wait_for_rate_limit(self):
        if not self.is_free_model:
            return

        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            wait_needed = self._min_interval - elapsed

            if wait_needed > 0:
                logger.info(
                    f"[OpenRouter] Free model rate limit: waiting {wait_needed:.1f}s "
                    f"(limit: 20 req/min)"
                )
                time.sleep(wait_needed)

    def _is_rate_limit_error(self, e: Exception) -> bool:
        for attr in ('http_res', 'raw_response', 'response'):
            obj = getattr(e, attr, None)
            if obj is not None:
                try:
                    if getattr(obj, 'status_code', None) == 429:
                        return True
                except Exception:
                    pass

        err_str = str(e)
        return (
            "429" in err_str
            or "too many requests" in err_str.lower()
            or "rate limit" in err_str.lower()
            or "ratelimit" in err_str.lower()
        )

    def translate(
        self,
        prompt: List[Dict],
        temperature: float,
        timeout_seconds: int,
        max_retries: int = 5
    ) -> str:
        try:
            pass
        except ImportError:
            raise ImportError(
                "openrouter package is not installed.\n"
                "Run: pip install openrouter"
            )

        for attempt in range(1, max_retries + 1):
            self._wait_for_rate_limit()

            self._last_request_time = time.time()

            try:
                logger.debug(
                    f"OpenRouter SDK request (attempt {attempt}/{max_retries}): "
                    f"model={self.model_name}, temp={temperature}, "
                    f"timeout={timeout_seconds}s, free={self.is_free_model}"
                )

                with OpenRouter(api_key=self.api_key) as client:
                    response = client.chat.send(
                        model=self.model_name,
                        messages=prompt,
                        temperature=temperature
                    )

                if not response.choices:
                    logger.warning("OpenRouter returned no choices")
                    return ""

                content = response.choices[0].message.content

                if not content:
                    logger.warning("OpenRouter returned empty content")
                    return ""

                logger.debug(f"OpenRouter response: {len(content)} chars")
                return content

            except Exception as e:
                err_str = str(e)

                if self._is_rate_limit_error(e):
                    self._last_request_time = time.time()

                    wait_seconds = min(10.0 * attempt, 120.0)

                    if attempt < max_retries:
                        logger.warning(
                            f"[OpenRouter] Rate limited 429 "
                            f"(attempt {attempt}/{max_retries}). "
                            f"Waiting {wait_seconds:.0f}s before retry..."
                        )
                        time.sleep(wait_seconds)
                        continue
                    else:
                        logger.error(
                            f"[OpenRouter] Rate limited - "
                            f"max retries ({max_retries}) reached. Giving up."
                        )
                        raise Exception(
                            f"OpenRouter rate limit exceeded after {max_retries} attempts"
                        )

                is_timeout = (
                    "timeout" in err_str.lower()
                    or "timed out" in err_str.lower()
                )

                if is_timeout:
                    logger.error(f"OpenRouter timeout after {timeout_seconds}s")
                    raise TimeoutError(
                        f"OpenRouter request timed out after {timeout_seconds}s"
                    )

                is_not_found = (
                    "404" in err_str
                    or "not found" in err_str.lower()
                    or ("model" in err_str.lower() and "not" in err_str.lower())
                )

                if is_not_found:
                    raise Exception(
                        f"OpenRouter model not found: '{self.model_name}'\n"
                        f"Check the model name at openrouter.ai/models\n"
                        f"Details: {err_str}"
                    )

                is_auth_error = (
                    "401" in err_str
                    or "403" in err_str
                    or "unauthorized" in err_str.lower()
                    or "forbidden" in err_str.lower()
                    or "invalid api key" in err_str.lower()
                )

                if is_auth_error:
                    raise Exception(
                        f"OpenRouter authentication error.\n"
                        f"Check your API key in Options tab.\n"
                        f"Details: {err_str}"
                    )

                logger.error(f"OpenRouter request failed: {err_str}")
                raise Exception(f"OpenRouter error: {err_str}")

        raise Exception(f"OpenRouter: all {max_retries} attempts failed")

class LLMClientFactory:
    @staticmethod
    def create_client(
        llm_choice: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> LLMClient:
        if llm_choice == "LM Studio":
            if endpoint:
                return LMStudioClient(endpoint)
            return LMStudioClient()

        elif llm_choice == "Ollama":
            if not model_name:
                raise ValueError("Ollama requires model_name")

            if endpoint:
                return OllamaClient(model_name, endpoint)
            return OllamaClient(model_name)

        elif llm_choice == "Openrouter":
            if not api_key or not model_name:
                raise ValueError("OpenRouter requires api_key and model_name")

            return OpenRouterClient(api_key, model_name)

        else:
            raise ValueError(f"Unknown LLM choice: {llm_choice}")

class PromptBuilder:
    def __init__(
        self,
        variant: str,
        system_template: Optional[str] = None,
        assistant_template: Optional[str] = None,
        user_template: Optional[str] = None,
        ollama_template: Optional[str] = None,
        single_prompt_mode: bool = False,
        json_payload_template: Optional[str] = None,
        json_response_field: str = ""
    ):
        self.variant = variant
        self.system_template = system_template
        self.assistant_template = assistant_template
        self.user_template = user_template
        self.ollama_template = ollama_template
        self.single_prompt_mode = single_prompt_mode
        self.json_payload_template = json_payload_template
        self.json_response_field = json_response_field

        logger.info(f"PromptBuilder initialized: variant={variant}, single_prompt={single_prompt_mode}, json_payload={bool(json_payload_template)}")

    def build_prompt(
        self,
        core_text: str,
        context_before: str = "",
        context_after: str = "",
        auto_fix_section: str = ""
    ) -> Union[str, List[Dict]]:
        if self.json_payload_template is not None:
            def json_escape(value: str) -> str:
                return json.dumps(value)[1:-1]

            filled = self.json_payload_template.replace("{core_text}", json_escape(core_text))
            filled = filled.replace("{context_before}", json_escape(context_before))
            filled = filled.replace("{context_after}", json_escape(context_after))

            if auto_fix_section and auto_fix_section.strip():
                auto_fix_formatted = (
                    "\\n\\n"
                    "=== AUTO-FIX INSTRUCTIONS - CRITICAL ===\\n"
                    "You MUST fix the following issues in your previous translation:\\n\\n"
                    + json_escape(auto_fix_section.strip()) +
                    "\\n\\n"
                    "=== END OF AUTO-FIX INSTRUCTIONS ===\\n"
                    "Return ONLY the corrected translation in the same JSON format as before."
                )

                try:
                    payload_dict = json.loads(filled)
                    if isinstance(payload_dict, dict):
                        # Najpierw próbujemy dodać do pola "messages"
                        if "messages" in payload_dict and isinstance(payload_dict["messages"], list):
                            for msg in payload_dict["messages"]:
                                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                                    msg["content"] = msg["content"].rstrip() + auto_fix_formatted
                                    break
                        elif isinstance(payload_dict.get("content"), str):
                            payload_dict["content"] = payload_dict["content"].rstrip() + auto_fix_formatted
                        # Fallback
                        else:
                            filled = filled.rstrip() + auto_fix_formatted
                    else:
                        filled = filled.rstrip() + auto_fix_formatted
                except (json.JSONDecodeError, TypeError):
                    filled = filled.rstrip() + auto_fix_formatted

            try:
                payload_dict = json.loads(filled)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON Payload Template: {e}\n"
                    f"Check the template in the LLM Instructions Editor."
                )

            return {
                "__json_payload__": payload_dict,
                "__response_field__": self.json_response_field
            }

        if self.ollama_template is not None:
            prompt = self.ollama_template.format(
                core_text=core_text,
                context_before=context_before,
                context_after=context_after
            )
            if auto_fix_section:
                prompt += "\n\n" + auto_fix_section
            return prompt

        else:
            system_content = self.system_template.format(
                core_text=core_text,
                context_before=context_before,
                context_after=context_after
            ) if self.system_template else ""

            assistant_content = self.assistant_template.format(
                core_text=core_text,
                context_before=context_before,
                context_after=context_after
            ) if self.assistant_template else ""

            user_content = self.user_template.format(
                core_text=core_text,
                context_before=context_before,
                context_after=context_after
            ) if self.user_template else ""

            if auto_fix_section:
                user_content += "\n\n" + auto_fix_section

            if self.single_prompt_mode:
                merged = ""
                if system_content:
                    merged += system_content + "\n\n"
                if assistant_content:
                    merged += assistant_content + "\n\n"
                if user_content:
                    merged += user_content
                return [{"role": "user", "content": merged}]
            else:
                messages = []
                if system_content:
                    messages.append({"role": "system", "content": system_content})
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})
                if user_content:
                    messages.append({"role": "user", "content": user_content})
                return messages

class AutoFixManager:
    def __init__(self, max_attempts: int, base_temperature: float):
        self.max_attempts = max_attempts
        self.base_temperature = base_temperature
        self.current_temperature = base_temperature
        self.history: List[Dict] = []
        self.best_attempt: Optional[Dict] = None

        logger.info(f"AutoFixManager initialized: max={max_attempts}, base_temp={base_temperature}")

    def record_attempt(
        self,
        translation: str,
        mismatch_flags: Dict
    ) -> None:
        error_count = sum(1 for v in mismatch_flags.values() if v)

        attempt = {
            'attempt': len(self.history) + 1,
            'translation': translation,
            'mismatch_flags': mismatch_flags.copy(),
            'error_count': error_count,
            'temperature': self.current_temperature
        }

        self.history.append(attempt)

        if self.best_attempt is None or error_count < self.best_attempt['error_count']:
            self.best_attempt = attempt
            logger.info(f"New best result! (errors: {error_count})")

        logger.debug(f"Recorded attempt {attempt['attempt']}: {error_count} errors")

    def should_retry(self) -> bool:
        return len(self.history) < self.max_attempts

    def get_current_attempt_number(self) -> int:
        return len(self.history) + 1

    def get_next_temperature(self) -> float:
        if len(self.history) >= 2:
            prev_errors = self.history[-2]['error_count']
            current_errors = self.history[-1]['error_count']

            if current_errors >= prev_errors:
                self.current_temperature = min(self.current_temperature + 0.1, 1.0)
                logger.info(f"No progress - increasing temperature to {self.current_temperature:.1f}")

        return self.current_temperature

    def get_best_translation(self) -> Optional[str]:
        return self.best_attempt['translation'] if self.best_attempt else None

    def build_auto_fix_section(
        self,
        original_text: str,
        current_translation: str,
        current_flags: Dict,
        use_ps_markers: bool = False
    ) -> str:
        def _flag_is_positioning_only(flag_key: str, flag_value) -> bool:
            if flag_key not in ('reserve_elements', 'nt_markers'):
                return False
            if not isinstance(flag_value, dict):
                return False
            has_structural = any(
                flag_value.get(k) for k in ('missing', 'extra', 'spurious_closing')
            )
            has_positioning = bool(flag_value.get('positioning'))
            return has_positioning and not has_structural

        lines = [
            "",
            f"AUTO-FIX ATTEMPT {self.get_current_attempt_number()}/{self.max_attempts}",
            "",
            "TRANSLATION PROGRESS:"
        ]

        if self.best_attempt:
            best_error_count    = self.best_attempt['error_count']
            current_error_count = sum(1 for v in current_flags.values() if v)

            if len(self.history) >= 1:
                last_error_count = self.history[-1]['error_count']

                if current_error_count > best_error_count:
                    lines.append("")
                    lines.append(f"⚠️ WARNING: Last attempt performed WORSE ({last_error_count} errors)")
                    lines.append(f"✓ BEST RESULT SO FAR (attempt {self.best_attempt['attempt']}): {best_error_count} errors")
                    lines.append("")
                    lines.append("Use this REFERENCE translation as a baseline for corrections:")
                    lines.append(f"<reference_translation>{self.best_attempt['translation']}</reference_translation>")
                    lines.append("")
                    lines.append("Your task: FIX remaining issues based on the REFERENCE translation above.")
                    lines.append("")

                elif len(self.history) >= 2 and last_error_count == self.history[-2]['error_count']:
                    lines.append("")
                    lines.append(f"⚠️ NO PROGRESS: Still {last_error_count} errors (same as previous attempt)")
                    lines.append("Try changing phrasing, structure, or approach to reduce errors.")
                    lines.append("")
                    lines.append("Previous translation (requires a different approach):")
                    lines.append(f"<previous_translation>{self.history[-1]['translation']}</previous_translation>")
                    lines.append("")

                else:
                    if len(self.history) >= 2:
                        prev_error_count = self.history[-2]['error_count']
                        lines.append("")
                        lines.append(f"✓ PROGRESS: {prev_error_count} → {last_error_count} errors")
                        lines.append("")
                    lines.append("Previous translation (incorrect, for reference):")
                    lines.append(f"<previous_translation>{self.history[-1]['translation']}</previous_translation>")
                    lines.append("")

        critical_issues  = []
        important_issues = []
        minor_issues     = []

        for flag_key, flag_value in current_flags.items():
            if not flag_value:
                continue

            if flag_key in ('reserve_elements', 'nt_markers'):
                if _flag_is_positioning_only(flag_key, flag_value):
                    important_issues.append((flag_key, flag_value))
                else:
                    critical_issues.append((flag_key, flag_value))

            elif flag_key in ('inline_formatting', 'untranslated'):
                critical_issues.append((flag_key, flag_value))

            elif flag_key in ('paragraphs', 'first_char', 'last_char', 'ps_markers'):
                important_issues.append((flag_key, flag_value))

            else:
                minor_issues.append((flag_key, flag_value))

        lines.append("CURRENT ISSUES TO FIX:")
        lines.append("")

        if critical_issues:
            lines.append("🔴 CRITICAL (must fix):")
            for flag_key, flag_value in critical_issues:
                lines.extend(self._format_issue_details(
                    flag_key, flag_value, original_text, current_translation,
                    use_ps_markers=use_ps_markers
                ))
            lines.append("")

        if important_issues:
            lines.append("🟡 IMPORTANT:")
            for flag_key, flag_value in important_issues:
                lines.extend(self._format_issue_details(
                    flag_key, flag_value, original_text, current_translation,
                    use_ps_markers=use_ps_markers
                ))
            lines.append("")

        if minor_issues:
            lines.append("🟢 MINOR (fix if possible):")
            for flag_key, flag_value in minor_issues:
                lines.extend(self._format_issue_details(
                    flag_key, flag_value, original_text, current_translation,
                    use_ps_markers=use_ps_markers
                ))
            lines.append("")

        lines.append("INSTRUCTIONS:")
        if critical_issues:
            lines.append("- Focus on CRITICAL issues first - they break the document structure")
        if important_issues and not critical_issues:
            lines.append("- Fix IMPORTANT issues — misplaced tags break formatting in the output file")
        if self.best_attempt:
            current_error_count = sum(1 for v in current_flags.values() if v)
            if current_error_count > self.best_attempt['error_count']:
                lines.append("- Use the BEST translation above as your starting point")
        lines.append("- Keep any parts that were already correct")
        lines.append("- Return ONLY the corrected translation in <translated>...</translated> tags")

        return "\n".join(lines)

    def _format_issue_details(
        self,
        flag_key: str,
        flag_value,
        original: str,
        translation: str,
        use_ps_markers: bool = False
    ) -> List[str]:
        lines = []

        if flag_key == 'reserve_elements':
            if isinstance(flag_value, dict):
                missing   = flag_value.get('missing', [])
                extra     = flag_value.get('extra', [])
                spurious  = flag_value.get('spurious_closing', [])
                pos_errors = flag_value.get('positioning', [])

                if missing:
                    lines.append(f"  ❌ Missing placeholders: {', '.join(missing[:4])}")
                    if len(missing) > 4:
                        lines.append(f"     ... and {len(missing) - 4} more")
                if extra:
                    lines.append(f"  ❌ Extra placeholders: {', '.join(extra[:4])}")
                    if len(extra) > 4:
                        lines.append(f"     ... and {len(extra) - 4} more")
                if spurious:
                    lines.append(f"  ❌ Spurious CLOSING tags (LLM bug): {', '.join(spurious)}")
                    lines.append(f"     → <id_xx> are self-closing — remove all closing tags!")

                if pos_errors:
                    lines.append("")
                    lines.append(f"  ❌ PLACEHOLDER POSITION ERRORS:")
                    for error in pos_errors[:4]:
                        tag_id = error.get('tag_id', '??')
                        orig_pct = int(error.get('orig_rel_pos', 0) * 100)
                        trans_pct = int(error.get('trans_rel_pos', 0) * 100)
                        lines.append(f"     • <id_{tag_id}> should be ~{orig_pct}% but is at ~{trans_pct}%")

        elif flag_key == 'nt_markers':
            if isinstance(flag_value, dict):
                lines.append("  ❌ Non-translatable markers (<nt_xx/>) are wrong or missing")
                if flag_value.get('missing'):
                    lines.append(f"     Missing: {', '.join(flag_value['missing'][:4])}")
                if flag_value.get('positioning'):
                    lines.append("     Wrong position — must stay exactly where they were")

        elif flag_key == 'inline_formatting':
            lines.append("  ❌ Inline formatting tags (<p_xx>) are broken or mismatched")

            if isinstance(flag_value, dict) and flag_value:
                # Opening tags
                if 'opening_tags' in flag_value:
                    ot = flag_value['opening_tags']
                    if ot.get('missing'):
                        lines.append(f"     ❌ Missing opening: {', '.join(ot['missing'][:6])}")
                    if ot.get('extra'):
                        lines.append(f"     ❌ Extra opening: {', '.join(ot['extra'][:6])}")

                # Closing tags
                if 'closing_tags' in flag_value:
                    ct = flag_value['closing_tags']
                    if ct.get('missing'):
                        lines.append(f"     ❌ Missing closing: {', '.join(ct['missing'][:6])}")
                    if ct.get('extra'):
                        lines.append(f"     ❌ Extra closing: {', '.join(ct['extra'][:6])}")

                if 'unexpected_tags' in flag_value:
                    lines.append(f"     Found raw HTML tags: {', '.join(flag_value['unexpected_tags'])}")
                    lines.append("     → Use ONLY <p_00>, <p_01>... placeholders")

                if flag_value.get('unpaired_tags'):
                    lines.append("     Unpaired opening/closing tags")

                if flag_value.get('positioning'):
                    lines.append("     Tags are in wrong places")

            else:
                lines.append("     (no detailed info – probably missing/extra <p_XX> tags)")

            # Specjalna rada dla tytułów rozdziałów (bardzo częsty przypadek błędu)
            if len(original.strip()) < 80 and '<p_' not in original:
                lines.append("     💡 This looks like a chapter title — LLM often adds phantom <p_xx> tags here.")

        elif flag_key == 'ps_markers':
            ps_in_orig = original.count('<ps>')
            ps_in_trans = translation.count('<ps>')
            lines.append(f"  ❌ Paragraph structure mismatch")
            lines.append(f"     Original: {ps_in_orig} <ps> markers")
            lines.append(f"     Translation: {ps_in_trans} <ps> markers")
            if ps_in_orig > ps_in_trans:
                lines.append(f"  → MISSING {ps_in_orig - ps_in_trans} <ps> marker(s)!")
            elif ps_in_orig < ps_in_trans:
                lines.append(f"  → TOO MANY {ps_in_trans - ps_in_orig} <ps> marker(s)!")

        elif flag_key == 'length':
            if isinstance(flag_value, dict):
                orig_len = flag_value.get("orig_chars", 0)
                trans_len = flag_value.get("trans_chars", 0)
                ratio = flag_value.get("ratio", 1.0)
                lines.append(f"  ❌ Significant length difference:")
                lines.append(f"     Original: {orig_len} characters")
                lines.append(f"     Translation: {trans_len} characters")
                lines.append(f"     Ratio: {ratio:.2f}x")
                if ratio > 1.7 and orig_len < 80:
                    lines.append("     → This looks like a chapter title — keep it short!")

        elif flag_key == 'content_drift':
            lines.append("  ❌ CONTENT DRIFT detected")
            lines.append("     Translation started writing completely different content")
            lines.append("     → Stay focused only on the text inside <text_to_translate>")

        elif flag_key == 'unexpected_html':
            lines.append("  ❌ Raw HTML tags found in output")
            lines.append(f"     Tags: {', '.join(flag_value) if isinstance(flag_value, list) else str(flag_value)}")
            lines.append("     → Use ONLY <p_00>, <p_01> placeholders")

        elif flag_key == 'first_char':
            orig_first_char, orig_desc   = self._get_first_char_details(original)
            trans_first_char, trans_desc = self._get_first_char_details(translation)
            if orig_first_char and trans_first_char:
                lines.append(f"  ❌ First character mismatch:")
                lines.append(f"     Original starts with {orig_desc}: '{orig_first_char}'")
                lines.append(f"     Translation starts with {trans_desc}: '{trans_first_char}'")
                lines.append(f"  → Translation should start with {orig_desc}")

        elif flag_key == 'last_char':
            orig_last_info  = self._get_last_char_details(original)
            trans_last_info = self._get_last_char_details(translation)
            if orig_last_info['char'] and trans_last_info['char']:
                lines.append(f"  ❌ Last character/punctuation mismatch:")
                lines.append(f"     Original ends with: {orig_last_info['description']}")
                lines.append(f"     Translation ends with: {trans_last_info['description']}")
                lines.append(f"  → Fix: Translation should end with {orig_last_info['type']}")

        elif flag_key == 'quote_parity':
            lines.append(f"  ❌ Unpaired quotation marks (odd number)")
            lines.append(f"  → Add or remove ONE quote to make even number")

        elif flag_key == 'untranslated':
            lines.append(f"  ❌ Translation appears identical to original")
            lines.append(f"  → TRANSLATE the text, do not return it unchanged")

        else:
            lines.append(f"  - {flag_key}")

        return lines

    def _get_first_char_details(self, text: str) -> tuple:
        clean_text = re.sub(r'</?p_\d{2}>|<id_\d{2}>|</id_\d{2}>|<nt_\d{2}/>|</?ps>', '', text)
        clean_text = clean_text.lstrip()

        if not clean_text:
            return None, "empty"

        ALL_QUOTES = r'["\'\\u2018\\u2019\\u201C\\u201D\\u201E\\u201F\\u00AB\\u00BB\\u2039\\u203A\\u201A\\u201B\\u2032\\u2033\\u301D\\u301E\\u301F\\uFF02]'
        text_no_quotes = re.sub(f'^{ALL_QUOTES}+', '', clean_text).lstrip()

        if not text_no_quotes:
            return '"', "quotation mark"

        first_char = text_no_quotes[0]

        if first_char.isdigit():
            desc = f"digit ({first_char})"
        elif first_char.isupper():
            desc = f"uppercase letter ({first_char})"
        elif first_char.islower():
            desc = f"lowercase letter ({first_char})"
        elif first_char == '<':
            desc = "HTML/placeholder tag"
        else:
            desc = f"special character ({first_char})"

        return first_char, desc

    def _get_last_char_details(self, text: str) -> dict:
        clean_text = re.sub(r'</?p_\d{2}>|<id_\d{2}>|</id_\d{2}>|<nt_\d{2}/>|</?ps>', '', text)
        clean_text = clean_text.rstrip()

        if not clean_text:
            return {'char': None, 'type': 'empty', 'description': 'empty text'}

        ALL_QUOTES = r'["\'\\u2018\\u2019\\u201C\\u201D\\u201E\\u201F\\u00AB\\u00BB\\u2039\\u203A\\u201A\\u201B\\u2032\\u2033\\u301D\\u301E\\u301F\\uFF02]'
        text_no_quotes = re.sub(f'{ALL_QUOTES}+$', '', clean_text).rstrip()

        if not text_no_quotes:
            return {'char': '"', 'type': 'quote', 'description': 'quotation mark only'}

        last_char = text_no_quotes[-1]

        punct_map = {
            '.': ('period',      'period (.)'),
            ',': ('comma',       'comma (,)'),
            '!': ('exclamation', 'exclamation mark (!)'),
            '?': ('question',    'question mark (?)'),
            '…': ('ellipsis',    'ellipsis (…)'),
            ';': ('semicolon',   'semicolon (;)'),
            ':': ('colon',       'colon (:)'),
            '-': ('dash',        'dash (-)'),
            '–': ('dash',        'en-dash (–)'),
            '—': ('dash',        'em-dash (—)')
        }

        if last_char in punct_map:
            char_type, desc = punct_map[last_char]
            return {'char': last_char, 'type': char_type, 'description': desc}
        elif last_char.isalpha():
            return {'char': last_char, 'type': 'letter',  'description': f'letter ({last_char})'}
        elif last_char.isdigit():
            return {'char': last_char, 'type': 'digit',   'description': f'digit ({last_char})'}
        else:
            return {'char': last_char, 'type': 'other',   'description': f'special character ({last_char})'}

    def reset_temperature(self) -> None:
        self.current_temperature = self.base_temperature
        logger.info(f"Temperature reset to {self.base_temperature:.1f}")

class TranslationOrchestrator:
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
        formatting_sync=None,
        timeout_minutes: int = 10
    ):
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.formatting_sync = formatting_sync
        self.timeout_seconds = timeout_minutes * 60
        self._cancelled = False

        logger.info(f"TranslationOrchestrator initialized: timeout={timeout_minutes}min")

    def cancel(self):
        self._cancelled = True
        logger.info("TranslationOrchestrator: cancellation requested")

    def hard_cancel(self):
        self._cancelled = True
        logger.info("TranslationOrchestrator: HARD CANCEL requested")
        if hasattr(self.llm_client, 'abort'):
            self.llm_client.abort()

    def _log_prompt_details(
        self,
        prompt: Union[str, List[Dict]],
        fragment: Dict,
        context_before: List[Dict],
        context_after: List[Dict],
        attempt_num: int,
        temperature: float,
        has_auto_fix: bool = False
    ):
        separator = "=" * 80
        logger.info("")
        logger.info(separator)

        if has_auto_fix:
            logger.info(f"🔵 LLM REQUEST - Fragment {fragment.get('index', '?')+1} - Attempt {attempt_num} 🔧 AUTO-FIX ACTIVE")
        else:
            logger.info(f"🔵 LLM REQUEST - Fragment {fragment.get('index', '?')+1} - Attempt {attempt_num}")

        logger.info(separator)

        logger.info("")
        logger.info("📄 FRAGMENT TO TRANSLATE:")
        logger.info("-" * 80)
        original_text = fragment.get('original_text', '')
        logger.info(f"Length: {len(original_text)} chars")
        logger.info(f"Text: {original_text[:200]}{'...' if len(original_text) > 200 else ''}")
        logger.info("-" * 80)

        logger.info("")
        logger.info("📚 CONTEXT:")
        logger.info("-" * 80)
        logger.info(f"Before: {len(context_before)} paragraph(s)")
        if context_before:
            for i, para in enumerate(context_before, 1):
                text = para.get('translated_text') if para.get('is_translated') else para.get('original_text', '')
                logger.info(f"  [{i}] {text[:100]}{'...' if len(text) > 100 else ''}")

        logger.info(f"After: {len(context_after)} paragraph(s)")
        if context_after:
            for i, para in enumerate(context_after, 1):
                text = para.get('translated_text') if para.get('is_translated') else para.get('original_text', '')
                logger.info(f"  [{i}] {text[:100]}{'...' if len(text) > 100 else ''}")
        logger.info("-" * 80)

        logger.info("")
        logger.info("⚙️ PARAMETERS:")
        logger.info("-" * 80)
        logger.info(f"Temperature: {temperature:.2f}")
        logger.info(f"Timeout: {self.timeout_seconds}s ({self.timeout_seconds/60:.1f} min)")
        logger.info(f"Variant: {self.prompt_builder.variant}")
        logger.info(f"Single-prompt mode: {self.prompt_builder.single_prompt_mode}")
        logger.info(f"Auto-fix active: {'YES 🔧' if has_auto_fix else 'NO'}")
        logger.info("-" * 80)

        logger.info("")
        logger.info("💬 FULL PROMPT SENT TO LLM:")
        logger.info("-" * 80)

        if isinstance(prompt, str):
            logger.info("Format: Single string (Ollama)")
            logger.info("")

            if len(prompt) > 2000:
                logger.info(prompt[:1000])
                logger.info("")
                logger.info(f"... [TRUNCATED {len(prompt) - 2000} chars] ...")
                logger.info("")
                logger.info(prompt[-1000:])
            else:
                logger.info(prompt)

        elif isinstance(prompt, list):
            logger.info(f"Format: Messages array ({len(prompt)} message(s))")
            logger.info("")

            for i, msg in enumerate(prompt, 1):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')

                if has_auto_fix and role == 'user':
                    logger.info(f"Message {i}/{len(prompt)} - Role: {role.upper()} 🔧 CONTAINS AUTO-FIX INSTRUCTIONS")
                else:
                    logger.info(f"Message {i}/{len(prompt)} - Role: {role.upper()}")

                logger.info("~" * 80)

                if len(content) > 1000:
                    logger.info(content[:500])
                    logger.info("")
                    logger.info(f"... [TRUNCATED {len(content) - 1000} chars] ...")
                    logger.info("")
                    logger.info(content[-500:])
                else:
                    logger.info(content)

                logger.info("~" * 80)
                logger.info("")

        logger.info("-" * 80)
        logger.info(separator)
        logger.info("")

    def _restore_paragraph_structure(self, text: str, fragment: Dict) -> str:
        original_text = fragment.get('original_text', '')
        import re as _re
        orig_has_ps = bool(_re.search(r'<ps>', original_text))

        if not fragment.get('_had_newlines', False):
            if not orig_has_ps and '<ps>' in text:
                cleaned = _re.sub(r'<ps>', '', text).strip()
                logger.warning(
                    f"Stripped spurious <ps> from single-part fragment "
                    f"(original has no paragraph breaks)"
                )
                return cleaned
            return text

        original_parts = fragment.get('_original_parts', [])
        original_separators = fragment.get('_original_separators', [])
        n_parts = len(original_parts)

        if n_parts <= 1:
            if not orig_has_ps and '<ps>' in text:
                cleaned = _re.sub(r'<ps>', '', text).strip()
                logger.warning("Stripped spurious <ps> from single-part fragment")
                return cleaned
            return text

        text = text.strip()
        if not text:
            return text

        if fragment.get('_used_ps_marker', False):
            trans_parts = [p.strip() for p in text.split('<ps>')]
            trans_parts = [p for p in trans_parts if p]

            if len(trans_parts) == n_parts:
                result = ''
                for i, part in enumerate(trans_parts):
                    result += part
                    if i < len(original_separators):
                        result += original_separators[i]
                logger.debug(
                    f"Restored inline structure: {n_parts} parts via <ps> markers"
                )
                return result

            elif len(trans_parts) > 1:
                dominant_sep = original_separators[0] if original_separators else '\n'
                result = dominant_sep.join(trans_parts)
                logger.warning(
                    f"<ps> count mismatch: expected {n_parts}, got {len(trans_parts)} "
                    f"— joining with {repr(dominant_sep)}"
                )
                return result

            else:
                flat = _re.sub(r'<ps>', '', text).strip()
                logger.warning(
                    f"LLM dropped all <ps> markers (expected {n_parts} parts) "
                    f"— returning flat text without spurious <ps>"
                )
                return flat

        else:
            orig_lengths = [len(p.strip()) for p in original_parts]
            total_orig = sum(orig_lengths)

            if total_orig == 0:
                return text

            split_points = []
            cumulative = 0
            prev_split = 0

            for length in orig_lengths[:-1]:
                cumulative += length
                target_pos = int(len(text) * cumulative / total_orig)
                target_pos = min(target_pos, len(text) - 1)

                best_pos = None
                search_range = min(80, max(20, len(text) // (n_parts * 2)))

                for offset in range(search_range + 1):
                    pos_right = target_pos + offset
                    if pos_right < len(text) and text[pos_right] == ' ':
                        best_pos = pos_right
                        break

                    pos_left = target_pos - offset
                    if pos_left > prev_split and text[pos_left] == ' ':
                        best_pos = pos_left
                        break

                if best_pos is None:
                    fallback = text.find(' ', target_pos)
                    best_pos = fallback if fallback != -1 else target_pos

                split_points.append(best_pos)
                prev_split = best_pos

            parts = []
            prev = 0
            for sp in split_points:
                parts.append(text[prev:sp].strip())
                prev = sp + 1
            parts.append(text[prev:].strip())

            result = ''
            for i, part in enumerate(parts):
                result += part
                if i < len(original_separators):
                    result += original_separators[i]

            return result

    def translate_fragment(
        self,
        fragment: Dict,
        context_before: List[Dict],
        context_after: List[Dict],
        temperature: float,
        auto_fix_manager: Optional[AutoFixManager] = None,
        mismatch_checker: Optional['MismatchChecker'] = None,
        progress_callback: Optional[callable] = None
    ) -> str:
        core_text = fragment['original_text']

        if '\n' in core_text:
            use_ps_markers = fragment.get('use_ps_markers', True)
            processing_mode = fragment.get('processing_mode', 'inline')

            tokens = re.split(r'(\n\n|\n)', core_text)
            original_parts = []
            original_separators = []
            for i, token in enumerate(tokens):
                if i % 2 == 0:
                    original_parts.append(token)
                else:
                    original_separators.append(token)

            if not use_ps_markers:
                fragment['_had_newlines'] = False
                fragment['_original_parts'] = []
                fragment['_original_separators'] = []
                fragment['_used_ps_marker'] = False
                core_text = ' '.join(p.strip() for p in original_parts if p.strip())
            else:
                fragment['_had_newlines'] = True
                fragment['_original_parts'] = original_parts
                fragment['_original_separators'] = original_separators

                if processing_mode == 'inline':
                    fragment['_used_ps_marker'] = True
                    core_text = '<ps>'.join(p.strip() for p in original_parts if p.strip())
                else:
                    fragment['_used_ps_marker'] = False
                    core_text = ' '.join(p.strip() for p in original_parts if p.strip())
        else:
            fragment['_had_newlines'] = False
            fragment['_original_parts'] = []
            fragment['_original_separators'] = []
            fragment['_used_ps_marker'] = False

        context_before_str = self._build_context_string(context_before)
        context_after_str = self._build_context_string(context_after)

        while True:
            if progress_callback and auto_fix_manager:
                attempt_num = auto_fix_manager.get_current_attempt_number()
                max_attempts = auto_fix_manager.max_attempts
                current_temp = auto_fix_manager.current_temperature

                try:
                    progress_callback(attempt_num, max_attempts, current_temp)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")

            auto_fix_section = ""
            if auto_fix_manager and len(auto_fix_manager.history) > 0:
                auto_fix_section = auto_fix_manager.build_auto_fix_section(
                    fragment['original_text'],
                    fragment.get('translated_text', ''),
                    fragment.get('mismatch_flags', {}),
                    use_ps_markers=fragment.get('_used_ps_marker', False)
                )

                logger.info("")
                logger.info("=" * 80)
                logger.info("🔧 AUTO-FIX INSTRUCTIONS BEING ADDED TO PROMPT")
                logger.info("=" * 80)
                logger.info(f"Attempt: {auto_fix_manager.get_current_attempt_number()}/{auto_fix_manager.max_attempts}")
                logger.info(f"Previous attempts: {len(auto_fix_manager.history)}")
                if auto_fix_manager.best_attempt:
                    logger.info(
                        f"Best result so far: Attempt {auto_fix_manager.best_attempt['attempt']} "
                        f"({auto_fix_manager.best_attempt['error_count']} errors)"
                    )
                logger.info("-" * 80)
                logger.info("Full auto-fix section content:")
                logger.info("-" * 80)
                logger.info(auto_fix_section)
                logger.info("-" * 80)
                logger.info("=" * 80)
                logger.info("")

                temperature = auto_fix_manager.get_next_temperature()

            prompt = self.prompt_builder.build_prompt(
                core_text,
                context_before_str,
                context_after_str,
                auto_fix_section
            )

            attempt_num = auto_fix_manager.get_current_attempt_number() if auto_fix_manager else 1

            self._log_prompt_details(
                prompt=prompt,
                fragment=fragment,
                context_before=context_before,
                context_after=context_after,
                attempt_num=attempt_num,
                temperature=temperature,
                has_auto_fix=bool(auto_fix_section)
            )

            start_time = time.time()

            try:
                response = self.llm_client.translate(
                    prompt,
                    temperature,
                    self.timeout_seconds
                )

                elapsed = time.time() - start_time

                logger.info("")
                logger.info("=" * 80)
                logger.info(f"✅ LLM RESPONSE - Fragment {fragment.get('index', '?')+1} - Attempt {attempt_num}")
                logger.info("=" * 80)
                logger.info(f"Response time: {elapsed:.1f}s")
                logger.info(f"Response length: {len(response)} chars")
                logger.info("-" * 80)
                if len(response) > 500:
                    logger.info(f"{response[:250]}")
                    logger.info(f"... [TRUNCATED {len(response) - 500} chars] ...")
                    logger.info(f"{response[-250:]}")
                else:
                    logger.info(response)
                logger.info("-" * 80)
                logger.info("=" * 80)
                logger.info("")

            except TimeoutError as e:
                logger.error("")
                logger.error("=" * 80)
                logger.error(f"⏱️ TIMEOUT - Fragment {fragment.get('index', '?')+1} - Attempt {attempt_num}")
                logger.error("=" * 80)
                logger.error(f"Error: {e}")
                logger.error("=" * 80)
                logger.error("")
                raise

            except requests.RequestException as e:
                logger.error("")
                logger.error("=" * 80)
                logger.error(f"❌ REQUEST FAILED - Fragment {fragment.get('index', '?')+1} - Attempt {attempt_num}")
                logger.error("=" * 80)
                logger.error(f"Error: {e}")
                logger.error("=" * 80)
                logger.error("")
                raise

            cleaned = self._extract_translation_from_response(response)

            cleaned = self._restore_paragraph_structure(cleaned, fragment)

            if self.formatting_sync:
                cleaned = self.formatting_sync.sync_formatting(
                    original=fragment['original_text'],
                    translated=cleaned,
                    para=fragment
                )

            logger.info("")
            logger.info("🔧 EXTRACTED TRANSLATION (restored):")
            logger.info("-" * 80)
            logger.info(f"Length: {len(cleaned)} chars")
            if len(cleaned) > 300:
                logger.info(f"{cleaned[:150]}")
                logger.info(f"... [TRUNCATED {len(cleaned) - 300} chars] ...")
                logger.info(f"{cleaned[-150:]}")
            else:
                logger.info(cleaned)
            logger.info("-" * 80)
            logger.info("")

            if auto_fix_manager and mismatch_checker:
                fragment['translated_text'] = cleaned
                fragment['is_translated'] = True

                has_mismatch, mismatch_flags = mismatch_checker.check_mismatch(fragment)
                auto_fix_manager.record_attempt(cleaned, mismatch_flags)

                if has_mismatch:
                    error_count = sum(1 for v in mismatch_flags.values() if v)
                    logger.warning("")
                    logger.warning("⚠️ MISMATCH DETECTED:")
                    logger.warning("-" * 80)
                    logger.warning(f"Total errors: {error_count}")
                    for flag_key, flag_value in mismatch_flags.items():
                        if flag_value:
                            logger.warning(f"  - {flag_key}: {flag_value}")
                    logger.warning("-" * 80)
                    logger.warning("")
                else:
                    logger.info("")
                    logger.info("✅ NO MISMATCH - Translation is valid!")
                    logger.info("")

                if not has_mismatch:
                    logger.info(f"✓ Auto-fix SUCCESS on attempt {attempt_num}")
                    return cleaned

                elif auto_fix_manager.should_retry():
                    error_count = sum(1 for v in mismatch_flags.values() if v)
                    logger.warning(f"⚠ Attempt {attempt_num} has {error_count} mismatch errors - retrying...")
                    fragment['mismatch_flags'] = mismatch_flags
                    continue

                else:
                    best = auto_fix_manager.get_best_translation()
                    if best:
                        best_attempt_num = auto_fix_manager.best_attempt['attempt']
                        best_errors = auto_fix_manager.best_attempt['error_count']
                        logger.warning("")
                        logger.warning("=" * 80)
                        logger.warning("⚠️ AUTO-FIX FAILED - MAX ATTEMPTS REACHED")
                        logger.warning("=" * 80)
                        logger.warning(f"Total attempts: {attempt_num}")
                        logger.warning(f"Best result: Attempt {best_attempt_num} ({best_errors} errors)")
                        logger.warning("=" * 80)
                        logger.warning("")
                        return best
                    else:
                        logger.warning(f"⚠ Auto-fix FAILED - returning last attempt")
                        return cleaned

            else:
                return cleaned

    def _build_context_string(self, paragraphs: List[Dict]) -> str:
        if not paragraphs:
            return ""

        texts = []
        for para in paragraphs:
            if para.get('is_translated', False):
                texts.append(para['translated_text'])
            else:
                texts.append(para['original_text'])

        return "\n\n".join(texts)

    def _extract_translation_from_response(self, response: str) -> str:
        text = response.strip()

        matches = re.findall(
            r'<(?:translated|translation)>(.*?)</(?:translated|translation)>',
            text,
            re.DOTALL | re.IGNORECASE
        )

        if matches:
            best = max(matches, key=len)
            text = best.strip()
            logger.debug("Extracted text from <translated>/<translation> tags")

        if '\\n' in text:
            text = text.replace('\\n', '\n')
            logger.debug("Unescaped \\n sequences")

        if '\\"' in text:
            text = text.replace('\\"', '"')
            logger.debug("Unescaped \\\" sequences")

        return text

def build_context_section(
    context_before: List[Dict],
    context_after: List[Dict]
) -> Tuple[str, str]:
    def build_text(paragraphs):
        if not paragraphs:
            return ""
        texts = []
        for para in paragraphs:
            if para.get('is_translated', False):
                texts.append(para['translated_text'])
            else:
                texts.append(para['original_text'])
        return "\n\n".join(texts)

    return build_text(context_before), build_text(context_after)
