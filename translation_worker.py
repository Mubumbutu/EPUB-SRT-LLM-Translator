# translation_worker.py
import logging
import re
import requests
import time
import traceback
from PyQt6.QtCore import pyqtSignal, QThread

def normalize_quotes(text: str) -> str:
    if not text:
        return text

    replacements = {
        "„": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "‟": '"',
        "‹": "'",
        "›": "'",
        "‘": "'",
        "’": "'",
    }

    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)

    if result != text:
        logging.debug("Quote normalization applied")

    return result

def parse_llm_response(raw_response: str) -> str:
    raw = raw_response.strip()
    logging.debug(f"Raw LLM response (first 500 chars): {raw[:500]}")

    raw = re.sub(r"^```(?:xml)?\s*", "", raw, flags=re.IGNORECASE | re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    match = re.search(r'<translated>(.*?)</translated>', raw, re.DOTALL | re.IGNORECASE)

    if match:
        result = match.group(1)

        if result:
            logging.debug(f"Extracted from <translated> tag: {result[:100]}")
            return result
        else:
            logging.warning("Empty content inside <translated> tag - using fallback")

            logging.info("Fallback: returning entire response as translation")
            return raw

    logging.warning("No <translated> tag found - using entire response as fallback")
    logging.debug(f"Fallback content: {raw[:200]}")
    return raw

class TimeoutException(Exception):
    pass

def timeout_handler(func, timeout_seconds, *args, **kwargs):
    result = [TimeoutException("Function timeout")]

    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            result[0] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        logging.error(f"⏱ TIMEOUT after {timeout_seconds}s")
        raise TimeoutException(f"Translation timeout ({timeout_seconds}s)")

    if isinstance(result[0], Exception):
        raise result[0]

    return result[0]

class TranslationWorker(QThread):
    progress = pyqtSignal(int, str, bool)
    finished = pyqtSignal()

    def __init__(
        self,
        paragraphs_to_translate,
        context_before,
        context_after,
        temperature,
        all_paragraphs,
        llm_choice="LM Studio",
        model_name="local-model",
        openrouter_api_key=None,
        ollama_full_prompt_template=None,
        system_prompt_template=None,
        assistant_prompt_template=None,
        user_prompt_template=None,
        single_prompt_mode=False,
        timeout_minutes=10
    ):
        super().__init__()
        self.paragraphs_to_translate = paragraphs_to_translate
        self.context_before = context_before
        self.context_after = context_after
        self.temperature = temperature
        self.all_paragraphs = all_paragraphs
        self.llm_choice = llm_choice
        self.model_name = model_name
        self.openrouter_api_key = openrouter_api_key
        self.ollama_full_prompt_template = ollama_full_prompt_template
        self.system_prompt_template = system_prompt_template
        self.assistant_prompt_template = assistant_prompt_template
        self.user_prompt_template = user_prompt_template
        self.single_prompt_mode = single_prompt_mode
        self.timeout_seconds = timeout_minutes * 60

        self._cancelled = False

    def request_cancel(self):
        logging.info("Worker received cancel request")
        self._cancelled = True

    def split_prefix_suffix(self, text: str):
        m = re.match(r'^(\s*\d+[\.\)]\s*)(.*?)([\.\?!]?)(\s*)$', text)
        if m:
            prefix, core, punct, trail = m.groups()
            return prefix, core, punct + trail
        return "", text, ""

    def build_single_merged_prompt(self, system_prompt, assistant_prompt, user_prompt):
        sections = []

        if system_prompt and system_prompt.strip():
            sections.append("INSTRUCTIONS:")
            sections.append(system_prompt.strip())
            sections.append("")

        if assistant_prompt and assistant_prompt.strip():
            sections.append("CONTEXT AND ADDITIONAL GUIDELINES:")
            sections.append(assistant_prompt.strip())
            sections.append("")

        if user_prompt and user_prompt.strip():
            sections.append("TASK TO PERFORM NOW:")
            sections.append(user_prompt.strip())

        sections.append("\nYour translation:")

        merged = "\n".join(sections)
        return merged

    def call_ollama_api(self, prompt):
        headers = {"Content-Type": "application/json"}
        base_url = "http://localhost:11434"

        try:
            test_response = requests.get(f"{base_url}", timeout=5)
            logging.debug(f"Ollama base URL response: {test_response.status_code}")
        except Exception as e:
            raise Exception(f"Ollama server not responding at {base_url}. Run: ollama serve. Error: {e}")

        try:
            health_response = requests.get(f"{base_url}/api/tags", timeout=5)
            logging.debug(f"Ollama /api/tags response: {health_response.status_code}")
            if health_response.status_code != 200:
                raise Exception(f"Ollama /api/tags not responding. Status: {health_response.status_code}. Run: ollama serve")
        except requests.exceptions.ConnectionError:
            raise Exception("Ollama server not running. Run: ollama serve")
        except requests.exceptions.Timeout:
            raise Exception("Connection timeout to Ollama. Run: ollama serve")

        try:
            models_data = health_response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            logging.debug(f"Available models: {available_models}")
            if self.model_name not in available_models:
                model_base = self.model_name.split(':')[0]
                matching_models = [m for m in available_models if m.startswith(model_base)]
                if matching_models:
                    logging.warning(f"Using model: {matching_models[0]} instead of {self.model_name}")
                    self.model_name = matching_models[0]
                else:
                    raise Exception(f"Model '{self.model_name}' not available. Available: {available_models}")
        except Exception as e:
            logging.warning(f"Cannot check models: {e}")

        ollama_payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
            "stop": ["<|im_end|>", "Human:", "Assistant:"]
        }

        generate_url = f"{base_url}/api/generate"
        logging.debug(f"Calling Ollama: {generate_url} with model: {self.model_name}")

        response = requests.post(
            generate_url,
            headers=headers,
            json=ollama_payload,
            timeout=self.timeout_seconds
        )

        logging.debug(f"Ollama response status: {response.status_code}")

        if response.status_code != 200:
            logging.error(f"Ollama error {response.status_code}: {response.text}")

        response.raise_for_status()
        data = response.json()

        if 'response' not in data:
            raise Exception(f"Invalid response from Ollama: {data}")

        response_text = data.get('response', '').strip()

        response_text = response_text.replace('<|im_sep|>', '')
        response_text = response_text.replace('<|im_end|>', '')
        response_text = response_text.replace('<|im_start|>', '')
        response_text = response_text.strip('-').strip()

        return response_text

    def call_lm_studio_api(self, system_prompt, assistant_prompt, user_prompt):
        headers = {"Content-Type": "application/json"}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": assistant_prompt},
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "model": "local-model",
            "messages": messages,
            "temperature": self.temperature,
            "stop": ["<|im_end|>", "Human:", "Assistant:"]
        }

        logging.debug("LM Studio payload messages:")
        logging.debug(" SYSTEM: %s", messages[0]["content"][:200])
        logging.debug(" ASSISTANT: %s", messages[1]["content"][:200])
        logging.debug("  USER: %s", messages[2]["content"][:200])

        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds
        )

        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()

    def call_lm_studio_single_prompt(self, merged_prompt):
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": "local-model",
            "prompt": merged_prompt,
            "temperature": max(0.1, self.temperature),
            "max_tokens": 128000,
            "stop": ["</translated>", "<|im_end|>", "==="],
            "stream": False,
            "repeat_penalty": 1.1,
            "top_p": 0.95,
            "top_k": 40
        }

        prompt_length = len(merged_prompt)
        logging.debug(f"Single-prompt length: {prompt_length} chars (~{prompt_length // 4} tokens)")

        if prompt_length > 12000:
            logging.warning(f"⚠ Very long prompt ({prompt_length} chars) - may exceed context window!")

        try:
            response = requests.post(
                "http://localhost:1234/v1/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds
            )

            response.raise_for_status()
            data = response.json()

            generated_text = data["choices"][0]["text"].strip()

            if not generated_text:
                logging.error("⚠ LM Studio returned EMPTY response!")
                logging.error(f"Full response data: {data}")
                raise ValueError("Model returned empty text (possible context overflow)")

            finish_reason = data["choices"][0].get("finish_reason", "unknown")
            logging.debug(f"Finish reason: {finish_reason}")

            if finish_reason == "length":
                logging.warning("⚠ Response cut off due to max_tokens limit")

            return generated_text

        except requests.exceptions.Timeout:
            raise TimeoutException(f"LM Studio timeout ({self.timeout_seconds}s)")
        except requests.exceptions.ConnectionError:
            raise ValueError("Cannot connect to LM Studio - is it running?")
        except Exception as e:
            logging.error(f"LM Studio API error: {e}")
            raise

    def call_openrouter_api(self, system_prompt, assistant_prompt, user_prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openrouter_api_key}"
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": assistant_prompt},
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stop": ["<|im_end|>", "Human:", "Assistant:"]
        }

        logging.debug("Openrouter payload messages:")
        logging.debug(" SYSTEM: %s", messages[0]["content"][:200])
        logging.debug(" ASSISTANT: %s", messages[1]["content"][:200])
        logging.debug("  USER: %s", messages[2]["content"][:200])

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds
        )

        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()

    def call_openrouter_single_prompt(self, merged_prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openrouter_api_key}"
        }

        messages = [
            {"role": "user", "content": merged_prompt}
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stop": ["<|im_end|>", "Human:", "Assistant:", "==="]
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds
        )

        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()

    def build_context(self, current_idx: int) -> tuple:
        context_before_paragraphs = []
        start_idx = max(0, current_idx - self.context_before)

        for i in range(start_idx, current_idx):
            para = self.all_paragraphs[i]

            if para.get('is_translated') and not para.get('has_mismatch', False):
                text = para['translated_text']
            else:
                text = para['original_text']

            context_before_paragraphs.append(text)

        context_before = "\n".join(context_before_paragraphs)

        context_after_paragraphs = []
        end_idx = min(len(self.all_paragraphs), current_idx + self.context_after + 1)

        for i in range(current_idx + 1, end_idx):
            para = self.all_paragraphs[i]

            if para.get('is_translated') and not para.get('has_mismatch', False):
                text = para['translated_text']
            else:
                text = para['original_text']

            context_after_paragraphs.append(text)

        context_after = "\n".join(context_after_paragraphs)

        return context_before, context_after

    def run(self):
        for idx in self.paragraphs_to_translate:
            if self._cancelled:
                logging.info(f"Translation cancelled - skipping fragment {idx}")
                break

            try:
                original_text = self.all_paragraphs[idx]['original_text']

                prefix, core_text, suffix = self.split_prefix_suffix(original_text)

                context_before, context_after = self.build_context(idx)

                logging.debug(f"Context BEFORE paragraph {idx}: \n{context_before}")
                logging.debug(f"Context AFTER paragraph {idx}: \n{context_after}")
                logging.debug(f"Core text to translate: {core_text[:100]}...")

                start_time = time.time()

                if self.llm_choice == "Ollama":
                    if not self.ollama_full_prompt_template:
                        raise ValueError("No Ollama prompt template provided")

                    full_prompt = self.ollama_full_prompt_template.format(
                        context_before=context_before,
                        context_after=context_after,
                        core_text=core_text
                    )

                    logging.debug(f"Ollama full prompt (first 500 chars): {full_prompt[:500]}")
                    translated_core = self.call_ollama_api(full_prompt)

                elif self.llm_choice == "Openrouter":
                    if not self.system_prompt_template or not self.assistant_prompt_template or not self.user_prompt_template:
                        raise ValueError("No system/assistant/user prompt templates provided")

                    if '{' in self.system_prompt_template:
                        system_prompt = self.system_prompt_template.format(
                            context_before=context_before,
                            context_after=context_after,
                            core_text=core_text
                        )
                    else:
                        system_prompt = self.system_prompt_template

                    assistant_prompt = self.assistant_prompt_template.format(
                        context_before=context_before,
                        context_after=context_after,
                        core_text=core_text
                    )

                    user_prompt = self.user_prompt_template.format(
                        core_text=core_text,
                        context_before="",
                        context_after=""
                    )

                    if self.single_prompt_mode:
                        merged_prompt = self.build_single_merged_prompt(
                            system_prompt, assistant_prompt, user_prompt
                        )
                        logging.debug(f"OpenRouter merged prompt (first 800 chars): {merged_prompt[:800]}")
                        translated_core = self.call_openrouter_single_prompt(merged_prompt)
                    else:
                        logging.debug(f"OpenRouter SYSTEM: {system_prompt[:200]}")
                        logging.debug(f"OpenRouter ASSISTANT: {assistant_prompt[:200]}")
                        logging.debug(f"OpenRouter USER: {user_prompt[:200]}")
                        translated_core = self.call_openrouter_api(system_prompt, assistant_prompt, user_prompt)

                    time.sleep(3)

                else:
                    if not self.system_prompt_template or not self.assistant_prompt_template or not self.user_prompt_template:
                        raise ValueError("No system/assistant/user prompt templates provided")

                    if '{' in self.system_prompt_template:
                        system_prompt = self.system_prompt_template.format(
                            context_before=context_before,
                            context_after=context_after,
                            core_text=core_text
                        )
                    else:
                        system_prompt = self.system_prompt_template

                    assistant_prompt = self.assistant_prompt_template.format(
                        context_before=context_before,
                        context_after=context_after,
                        core_text=core_text
                    )

                    user_prompt = self.user_prompt_template.format(
                        core_text=core_text,
                        context_before="",
                        context_after=""
                    )

                    if self.single_prompt_mode:
                        merged_prompt = self.build_single_merged_prompt(
                            system_prompt, assistant_prompt, user_prompt
                        )

                        prompt_tokens_estimate = len(merged_prompt) // 4
                        if prompt_tokens_estimate > 3000:
                            logging.warning(f"⚠ Prompt very long ({prompt_tokens_estimate} tokens) - reducing context...")

                        logging.debug(f"LM Studio merged prompt (first 800 chars): {merged_prompt[:800]}")
                        translated_core = self.call_lm_studio_single_prompt(merged_prompt)
                    else:
                        logging.debug(f"LM Studio SYSTEM: {system_prompt[:200]}")
                        logging.debug(f"LM Studio ASSISTANT: {assistant_prompt[:200]}")
                        logging.debug(f"LM Studio USER: {user_prompt[:200]}")
                        translated_core = self.call_lm_studio_api(system_prompt, assistant_prompt, user_prompt)

                if self._cancelled:
                    logging.info(f"Translation cancelled AFTER generation for idx={idx} - ACCEPTING result and stopping")

                elapsed = int(time.time() - start_time)
                logging.info(f"✓ Translation completed in {elapsed}s")

                translated_core = translated_core.strip()
                translated_core = translated_core.replace('\x00', '').replace('\ufffd', '')

                translated_core = parse_llm_response(translated_core)

                if not translated_core:
                    raise ValueError("Empty translation received")

                translated_core = normalize_quotes(translated_core)

                full_translation = f"{prefix}{translated_core}{suffix}"

                self.all_paragraphs[idx]['translated_text'] = full_translation
                self.all_paragraphs[idx]['is_translated'] = True
                self.progress.emit(idx, full_translation, False)

                if self._cancelled:
                    logging.info(f"Stopping translation loop after fragment {idx}")
                    break

            except TimeoutException as e:
                error_msg = f"⏱ TIMEOUT: {e}"
                logging.error(f"Timeout for idx={idx}: {e}")
                self.all_paragraphs[idx]['translated_text'] = ""
                self.all_paragraphs[idx]['is_translated'] = False
                self.progress.emit(idx, "", True)

            except Exception as e:
                error_msg = f"ERROR: {e}"
                logging.error(f"Translation error for idx={idx}: {e}")
                logging.error(traceback.format_exc())
                self.all_paragraphs[idx]['translated_text'] = error_msg
                self.all_paragraphs[idx]['is_translated'] = False
                self.progress.emit(idx, error_msg, True)

        self.finished.emit()
