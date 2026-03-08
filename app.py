# app.py
import ctypes
import deepl
import logging
import os
import re
import sys
import threading
import time
import traceback
import unicodedata
from app_utils import (
    AppSettingsManager,
    LanguageConstants,
    PromptManager,
    SessionManager,
)
from deep_translator import GoogleTranslator
from epub_creator_lxml import EPUBCreatorLxml
from file_processors import FileProcessorFactory
from format_alignment import (
    FormatAlignmentEngine,
    INLINE_TRANSFER_TAGS,
    download_model as _download_alignment_model_fn,
    download_model as _fa_download_model,
    get_local_model_path,
    is_model_downloaded,
    MODELS_SUBDIR,
)
from formatting import (
    ALL_QUOTES_CHARS,
    DOUBLE_QUOTES_CHARS,
    FormattingSynchronizer,
    MismatchChecker,
    SINGLE_QUOTES_CHARS,
)
from epub_preview import EPUBPreviewEngine, EPUBPreviewToolbar, EPUBReaderWindow
from PyQt6.QtCore import pyqtSignal, Qt, QThread, QTimer, QUrl
from PyQt6.QtGui import QColor, QCursor, QIcon, QPalette
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QGridLayout,
)
from translation_engine import (
    AutoFixManager,
    BatchSplitError,
    LLMClientFactory,
    PromptBuilder,
    SentenceBatcher,
    TranslationOrchestrator,
)
from typing import Dict, List, Optional

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreserveForegroundDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        fg = index.data(Qt.ItemDataRole.ForegroundRole)
        if fg is not None:
            option.palette.setColor(QPalette.ColorRole.HighlightedText, fg.color())
        super().paint(painter, option, index)

class TranslationWorkerThread(QThread):
    progress = pyqtSignal(int, str, bool)
    retry_progress = pyqtSignal(int, int, int, float)
    finished = pyqtSignal()

    def __init__(
        self,
        orchestrator,
        fragment,
        context_before,
        context_after,
        temperature,
        auto_fix_manager=None,
        mismatch_checker=None
    ):
        super().__init__()
        self.orchestrator = orchestrator
        self.fragment = fragment
        self.context_before = context_before
        self.context_after = context_after
        self.temperature = temperature
        self.auto_fix_manager = auto_fix_manager
        self.mismatch_checker = mismatch_checker
        self._cancelled = False

    def request_cancel(self):
        self._cancelled = True
        if self.orchestrator:
            self.orchestrator.cancel()

    def hard_cancel(self):
        self._cancelled = True
        if self.orchestrator:
            self.orchestrator.hard_cancel()
        if self.isRunning():
            self.terminate()
            self.wait(2000)
            logging.info("TranslationWorkerThread: thread terminated (hard cancel)")

    def _on_retry_attempt(self, attempt: int, max_attempts: int, temperature: float):
        idx = self.fragment.get('index', 0)
        self.retry_progress.emit(idx, attempt, max_attempts, temperature)

    def run(self):
        try:
            translation = self.orchestrator.translate_fragment(
                fragment=self.fragment,
                context_before=self.context_before,
                context_after=self.context_after,
                temperature=self.temperature,
                auto_fix_manager=self.auto_fix_manager,
                mismatch_checker=self.mismatch_checker,
                progress_callback=self._on_retry_attempt
            )

            idx = self.fragment.get('index', 0)
            self.progress.emit(idx, translation, False)

        except Exception as e:
            idx = self.fragment.get('index', 0)
            error_msg = f"Translation failed: {e}"
            logging.error(f"Translation error: {e}")
            logging.error(traceback.format_exc())
            self.progress.emit(idx, error_msg, True)

        finally:
            self.finished.emit()


class TranslationBatchWorkerThread(QThread):
    batch_progress = pyqtSignal(list, list, bool)
    retry_progress = pyqtSignal(int, int, float)
    finished = pyqtSignal()

    def __init__(
        self,
        orchestrator,
        fragments,
        indices,
        context_before,
        context_after,
        temperature,
        auto_fix_manager=None,
        mismatch_checker=None,
        batcher=None,
        batch_hint_template=""
    ):
        super().__init__()
        self.orchestrator = orchestrator
        self.fragments = fragments
        self.indices = indices
        self.context_before = context_before
        self.context_after = context_after
        self.temperature = temperature
        self.auto_fix_manager = auto_fix_manager
        self.mismatch_checker = mismatch_checker
        self.batcher = batcher
        self.batch_hint_template = batch_hint_template
        self._cancelled = False

    def request_cancel(self):
        self._cancelled = True
        if self.orchestrator:
            self.orchestrator.cancel()

    def hard_cancel(self):
        self._cancelled = True
        if self.orchestrator:
            self.orchestrator.hard_cancel()
        if self.isRunning():
            self.terminate()
            self.wait(2000)

    def run(self):
        try:
            batch_text, offsets = self.batcher.prepare_batch(self.fragments)
            n = len(self.fragments)
            max_attempts = self.auto_fix_manager.max_attempts if self.auto_fix_manager else 3
            last_error = None
            last_response = ""

            for attempt in range(1, max_attempts + 1):
                if self._cancelled:
                    return

                try:
                    batch_hint = self.batcher.build_batch_hint(n, self.batch_hint_template)

                    if attempt > 1 and last_error is not None:
                        actual = SentenceBatcher.count_sentence_markers(last_response)
                        extra = (
                            f"\n\nPREVIOUS ATTEMPT FAILED: found {actual} <z> marker(s) instead of {n - 1}.\n"
                            f"You MUST output EXACTLY {n - 1} <z> marker(s). Count them carefully before responding."
                        )
                        batch_hint = batch_hint + extra

                    self.orchestrator.prompt_builder._pending_batch_hint = batch_hint

                    prompt = self.orchestrator.prompt_builder.build_prompt(
                        core_text=batch_text,
                        context_before="",
                        context_after=""
                    )

                    response = self.orchestrator.llm_client.translate(
                        prompt,
                        self.temperature,
                        self.orchestrator.timeout_seconds
                    )

                    if self._cancelled:
                        return

                    last_response = response.strip()

                    outer_match = re.match(
                        r'^<(?:translated|translation)>(.*)</(?:translated|translation)>$',
                        last_response,
                        re.DOTALL | re.IGNORECASE
                    )
                    if outer_match:
                        last_response = outer_match.group(1).strip()

                    translations = self.batcher.split_response(last_response, n, offsets)
                    self.batch_progress.emit(self.indices, translations, False)
                    return

                except BatchSplitError as e:
                    if self._cancelled:
                        return
                    last_error = e
                    logging.warning(f"Batch attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt == max_attempts:
                        raise

            raise last_error

        except Exception as e:
            if self._cancelled:
                return
            error_msg = f"Batch translation failed: {e}"
            logging.error(error_msg)
            self.batch_progress.emit(self.indices, [error_msg] * len(self.indices), True)

        finally:
            self.finished.emit()


class AlignmentWorkerThread(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()

    def __init__(self, paragraphs, model_name, device, models_dir):
        super().__init__()
        self.paragraphs = paragraphs
        self.model_name = model_name
        self.device = device
        self.models_dir = models_dir
        self._cancelled = False

    def request_cancel(self):
        self._cancelled = True

    def hard_cancel(self):
        self._cancelled = True
        if self.isRunning():
            self.terminate()
            logging.info("AlignmentWorkerThread: terminated (hard cancel)")

    def run(self):
        try:
            engine = FormatAlignmentEngine(
                model_name=self.model_name,
                device=self.device,
                local_models_dir=self.models_dir,
            )
            engine.load_model()

            total = len(self.paragraphs)
            completed = 0

            for para in self.paragraphs:
                if self._cancelled:
                    break
                engine.align_batch([para])
                completed += 1
                self.progress.emit(completed, total)

            try:
                engine.unload_model()
            except Exception as exc:
                logging.warning(f"AlignmentWorkerThread: unload_model error: {exc}")

        except Exception as e:
            logging.error(f"AlignmentWorkerThread error: {e}")
            logging.error(traceback.format_exc())
        finally:
            self.finished.emit()

class SRTCreator(QThread):
    finished = pyqtSignal(str, bool)

    def __init__(self, paragraphs, output_path, translator_app):
        super().__init__()
        self.paragraphs = paragraphs
        self.output_path = output_path
        self.translator_app = translator_app

    def run(self):
        try:
            def block_key(para):
                try:
                    return int(para.get('subtitle_block', 0))
                except:
                    return 0

            sorted_paragraphs = sorted(self.paragraphs, key=block_key)

            with open(self.output_path, 'w', encoding='utf-8') as f:
                for para in sorted_paragraphs:
                    block_num = para.get('subtitle_block', '1')
                    timestamp = para.get('timestamp', '00:00:00,000 --> 00:00:00,000')

                    if para.get('is_translated'):
                        raw_text = para.get('translated_text', '').strip()
                    else:
                        raw_text = para.get('original_text', '').strip()

                    if '\n' in raw_text:
                        lines = raw_text.split('\n')
                    else:
                        if self.translator_app:
                            lines = self.translator_app.split_translated_text_into_lines(raw_text, para)
                        else:
                            lines = [raw_text]

                    final_lines = []
                    for i, line in enumerate(lines):
                        if 'srt_tags_by_line' in para and i < len(para['srt_tags_by_line']):
                            tags_dict = para['srt_tags_by_line'][i]

                            for pos in sorted(tags_dict.keys(), reverse=True):
                                for tag_type, tag_val in reversed(tags_dict[pos]):
                                    if pos <= len(line):
                                        line = line[:pos] + tag_val + line[pos:]
                        final_lines.append(line)

                    f.write(f"{block_num}\n{timestamp}\n" + "\n".join(final_lines) + "\n\n")

            self.finished.emit(self.output_path, False)

        except Exception as e:
            logging.error(traceback.format_exc())
            self.finished.emit(str(e), True)

class TXTCreator(QThread):
    finished = pyqtSignal(str, bool)

    def __init__(self, paragraphs, output_path):
        super().__init__()
        self.paragraphs = paragraphs
        self.output_path = output_path

    def run(self):
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for i, para in enumerate(self.paragraphs):
                    text = para['translated_text'] if para['is_translated'] else para['original_text']
                    f.write(text)
                    if i < len(self.paragraphs) - 1:
                        f.write('\n\n')

            self.finished.emit(self.output_path, False)
        except Exception as e:
            self.finished.emit(str(e), True)

class AlignmentSaveDialog(QDialog):
    """
    Dialog shown before saving a Legacy-mode EPUB that has aligned fragments.
    Lets the user choose which alignment quality levels (dot colours) to
    include in the saved file.  Fragments whose dot is not selected are saved
    with plain translated text instead of the aligned HTML.
    """

    _DOT_DEFS = [
        ('🟢', 'Green',  'Auto-wrap / manually confirmed',  True),
        ('🟡', 'Yellow', 'Neural model — probably OK',      True),
        ('🟠', 'Orange', 'Neural model — worth checking',   False),
    ]

    def __init__(self, counts, parent=None):
        """
        Parameters
        ----------
        counts : dict  {emoji: int}  number of aligned paragraphs per dot
        """
        super().__init__(parent)
        self.setWindowTitle("Save EPUB — Alignment Filter")
        self.setMinimumWidth(460)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #cccccc; }
            QLabel  { color: #cccccc; }
            QCheckBox { color: #cccccc; font-size: 12px; spacing: 8px; }
            QCheckBox::indicator {
                width: 15px; height: 15px;
                border-radius: 3px; border: 2px solid #555;
                background-color: #1e1e1e;
            }
            QCheckBox::indicator:checked {
                border-color: #2a6aaa; background-color: #1a4a7a;
            }
            QPushButton {
                padding: 6px 18px; border-radius: 4px; font-size: 12px;
                background-color: #2a2a2a; color: #ccc;
                border: 1px solid #444;
            }
            QPushButton:hover { background-color: #3a3a3a; color: white; }
            QPushButton#save_btn {
                background-color: #1e5fa8; color: white;
                border: 1px solid #2a6aaa; font-weight: bold;
            }
            QPushButton#save_btn:hover { background-color: #2a72c8; }
            QFrame#sep { background-color: #333; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(18, 14, 18, 14)

        intro = QLabel(
            "This EPUB uses <b>Legacy mode</b> with alignment data.<br>"
            "Choose which alignment quality levels to include in the saved file.<br>"
            "<span style='color:#888;font-size:11px;'>"
            "Excluded fragments will be saved with plain translated text.</span>"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        sep1 = QFrame()
        sep1.setObjectName('sep')
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setFixedHeight(1)
        layout.addWidget(sep1)

        self._checkboxes = {}

        for emoji, short, desc, default in self._DOT_DEFS:
            n = counts.get(emoji, 0)
            if n == 0:
                continue
            cb = QCheckBox(f"{emoji}  {short} — {desc}   ({n} fragment{'s' if n != 1 else ''})")
            cb.setChecked(default)
            layout.addWidget(cb)
            self._checkboxes[emoji] = cb

        info_rows = []
        n_red   = counts.get('🔴', 0)
        n_black = counts.get('⚫', 0)
        n_none  = counts.get('○',  0)
        if n_red:
            info_rows.append(
                f"🔴  Flagged as bad — <b>always excluded</b> ({n_red} fragment{'s' if n_red != 1 else ''})"
            )
        if n_black:
            info_rows.append(
                f"⚫  No inline tags — not applicable ({n_black} fragment{'s' if n_black != 1 else ''})"
            )
        if n_none:
            info_rows.append(
                f"○  Not yet aligned — will save plain text ({n_none} fragment{'s' if n_none != 1 else ''})"
            )

        if info_rows:
            sep2 = QFrame()
            sep2.setObjectName('sep')
            sep2.setFrameShape(QFrame.Shape.HLine)
            sep2.setFixedHeight(1)
            layout.addWidget(sep2)
            for row in info_rows:
                lbl = QLabel(row)
                lbl.setStyleSheet("color: #666; font-size: 11px;")
                layout.addWidget(lbl)

        layout.addSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        btn_save = QPushButton("Save EPUB")
        btn_save.setObjectName("save_btn")
        btn_save.setDefault(True)
        btn_save.clicked.connect(self.accept)
        btn_row.addWidget(btn_save)
        layout.addLayout(btn_row)

    def selected_dots(self):
        """Return set of dot emojis the user wants to INCLUDE."""
        return {emoji for emoji, cb in self._checkboxes.items() if cb.isChecked()}


class EPUBPreviewPage(QWebEnginePage):
    """
    Custom QWebEnginePage that intercepts epub-preview://select/<element_id>
    navigation requests triggered by JS click handlers in the preview.
    Emits fragment_selected(str) with the element id.
    Also intercepts epub-preview://contextmenu/<element_id> and emits
    contextmenu_requested(str) so Python can show a native context menu.
    """
    fragment_selected    = pyqtSignal(str)
    contextmenu_requested = pyqtSignal(str)

    def acceptNavigationRequest(self, url, nav_type, is_main_frame):
        from PyQt6.QtCore import QUrl
        if isinstance(url, QUrl):
            scheme = url.scheme()
            host   = url.host()
            path   = url.path().lstrip('/')
        else:
            return True
        if scheme == 'epub-preview' and host == 'select':
            elem_id = path
            if elem_id:
                self.fragment_selected.emit(elem_id)
            return False
        if scheme == 'epub-preview' and host == 'contextmenu':
            elem_id = path
            if elem_id:
                self.contextmenu_requested.emit(elem_id)
            return False
        return True


def _html_strip_outer_tag(html: str) -> str:
    """Return the inner content of an HTML element, stripping the outermost tag.

    E.g.  '<p xmlns="..." id="x">foo <b>bar</b></p>'  →  'foo <b>bar</b>'
    If the outer tag cannot be detected the original string is returned unchanged.
    """
    html = html.strip()
    m = re.match(r'^<[^>]+>(.*)</\w+>\s*$', html, re.DOTALL)
    if m:
        return m.group(1)
    return html


def _html_restore_outer_tag(inner: str, original_html: str) -> str:
    """Re-wrap *inner* with the opening/closing tag taken from *original_html*.

    This is the inverse of _html_strip_outer_tag and is used when saving
    user edits back to the paragraph dict — the stored value must always be
    a full element (outer tag included) so that the EPUB writer can use it
    directly.
    """
    original_html = original_html.strip()
    m_open = re.match(r'^(<(\w+)[^>]*>)', original_html, re.DOTALL)
    if not m_open:
        return inner
    opening_tag = m_open.group(1)
    tag_name    = m_open.group(2)
    return f'{opening_tag}{inner}</{tag_name}>'


class TranslatorApp(QMainWindow):
    _download_done_signal = pyqtSignal(str, str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EPUB and SRT Translator with LLM by Mubumbutu")
        self.setWindowIcon(QIcon("icon.ico"))

        self.resize(1500, 900)
        self.setMinimumSize(1200, 700)

        self.file_processor: Optional[any] = None
        self.mismatch_checker: Optional[MismatchChecker] = None
        self.formatting_sync: Optional[FormattingSynchronizer] = None
        self.translation_orchestrator: Optional[TranslationOrchestrator] = None
        self.prompt_manager: PromptManager = PromptManager()
        self.current_prompts_cache: Dict[str, Dict[str, str]] = {}

        self.paragraphs: List[Dict] = []
        self.para_to_row_map = {}
        self.row_to_para_map = {}
        self.original_file_path: Optional[str] = None
        self.file_type: Optional[str] = None

        self.app_settings = AppSettingsManager.load_settings()

        self.translation_queue: List[int] = []
        self.current_translation_idx: Optional[int] = None
        self.current_auto_fix_attempt: int = 0
        self.completed_translations: int = 0
        self.total_to_translate: int = 0
        self.translation_cancelled: bool = False
        self.last_checked_row: Optional[int] = None
        self.is_session_loaded: bool = False

        self._eta_chars_done: int = 0
        self._eta_chars_total: int = 0
        self._eta_session_start: Optional[float] = None
        self._eta_recent_times: List[float] = []

        self.translation_timer = QTimer()
        self.translation_timer.timeout.connect(self.update_translation_time)
        self.translation_start_time: Optional[float] = None
        self.current_fragment_index: Optional[int] = None

        self.current_worker: Optional[QThread] = None
        self._alignment_worker: Optional[AlignmentWorkerThread] = None
        self._alignment_cancelled: bool = False
        self._alignment_total: int = 0
        self._alignment_completed: int = 0
        self.epub_creator: Optional[EPUBCreatorLxml] = None
        self.srt_creator: Optional[SRTCreator] = None
        self.txt_creator: Optional[TXTCreator] = None

        self._download_done_signal.connect(self._on_download_done_slot)

        self._preview_engine = EPUBPreviewEngine()
        self._preview_show_original_ids: set = set()
        self._preview_dark_mode: bool = False
        self._reader_window: EPUBReaderWindow = None

        self.init_ui()

        self._initialize_components()

    def _get_models_dir(self) -> str:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(app_dir, MODELS_SUBDIR)

    def _initialize_components(self):
        if self.file_type:
            self.mismatch_checker = MismatchChecker(
                self.file_type,
                mismatch_settings=self.app_settings
            )
        else:
            self.mismatch_checker = None

        if self.file_type:
            self.formatting_sync = FormattingSynchronizer(self.file_type)
        else:
            self.formatting_sync = None

        if hasattr(self, 'mismatch_check_checkboxes'):
            self._on_mismatch_check_toggled()

    def init_ui(self):
        translator_widget = QWidget()
        translator_layout = QVBoxLayout(translator_widget)
        translator_layout.setSpacing(6)
        translator_layout.setContentsMargins(8, 8, 8, 8)

        TOP_BTN_STYLE = """
            QPushButton {
                background-color: #2a2a2a;
                color: #cccccc;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px 14px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #3a3a3a; border-color: #666; color: white; }
            QPushButton:pressed { background-color: #1a1a1a; }
        """

        SMALL_BTN = """
            QPushButton {
                background-color: #252525;
                color: #bbbbbb;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #333; color: white; border-color: #555; }
            QPushButton:pressed { background-color: #1a1a1a; }
        """

        COMBO_STYLE = """
            QComboBox {
                padding: 4px 8px; border: 1px solid #444; border-radius: 3px;
                background-color: #252525; color: white; font-size: 11px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a; color: white;
                selection-background-color: #3a5a8a;
            }
        """

        SPINBOX_STYLE = """
            QSpinBox, QDoubleSpinBox {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 12px;
                min-height: 26px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 22px; height: 13px;
                background-color: #333;
                border-left: 1px solid #444;
                border-bottom: 1px solid #444;
                border-top-right-radius: 3px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 22px; height: 13px;
                background-color: #333;
                border-left: 1px solid #444;
                border-bottom-right-radius: 3px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #2a5a9a;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background-color: #1a3a6a;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: none; width: 0; height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #aaa;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none; width: 0; height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #aaa;
            }
        """

        PARAM_LABEL = "color: #888; font-size: 11px;"
        LABEL_STYLE = "color: #888; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;"
        TEXT_VIEW_STYLE = """
            QTextEdit {
                background-color: #161616;
                color: #e0e0e0;
                border: 1px solid #2e2e2e;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
                selection-background-color: #1e3a5f;
            }
        """
        EDITOR_BTN = "padding: 6px 12px; font-weight: bold; border-radius: 3px; color: white;"

        CHECKBOX_STYLE = """
            QCheckBox {
                color: #cccccc;
                font-size: 12px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid #555555;
                background-color: #1e1e1e;
            }
            QCheckBox::indicator:hover {
                border-color: #2a6aaa;
                background-color: #252535;
            }
            QCheckBox::indicator:checked {
                border-color: #2a6aaa;
                background-color: #1a4a7a;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #2a5a9a;
            }
        """

        top_panel = QHBoxLayout()
        top_panel.setSpacing(6)

        btn_open = QPushButton("📂  Open File")
        btn_open.clicked.connect(self.open_file)
        btn_open.setStyleSheet(TOP_BTN_STYLE)

        btn_save_session = QPushButton("💾  Save Session")
        btn_save_session.clicked.connect(self.save_session)
        btn_save_session.setStyleSheet(TOP_BTN_STYLE)

        btn_load_session = QPushButton("📥  Load Session")
        btn_load_session.clicked.connect(self.load_session)
        btn_load_session.setStyleSheet(TOP_BTN_STYLE)

        top_panel.addWidget(btn_open)
        top_panel.addWidget(btn_save_session)
        top_panel.addWidget(btn_load_session)
        top_panel.addStretch()
        translator_layout.addLayout(top_panel)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(5)

        left_widget = QWidget()
        left_widget.setMinimumWidth(220)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(4)
        left_layout.setContentsMargins(0, 0, 4, 0)

        file_label_layout = QHBoxLayout()
        file_label_layout.setSpacing(4)

        self.file_label = QLabel("📄  No file loaded")
        self.file_label.setStyleSheet("""
            QLabel {
                color: #888888;
                background-color: #1a1a1a;
                padding: 6px 10px;
                border-radius: 4px;
                border: 1px dashed #3a3a3a;
                font-size: 11px;
                font-style: italic;
            }
        """)
        file_label_layout.addWidget(self.file_label)

        self.btn_unload_file = QPushButton("✕")
        self.btn_unload_file.setMaximumWidth(28)
        self.btn_unload_file.setMinimumHeight(28)
        self.btn_unload_file.setToolTip("Unload file")
        self.btn_unload_file.setStyleSheet("""
            QPushButton {
                background-color: #5a1a1a; color: #ff8888;
                font-weight: bold; padding: 3px;
                border-radius: 4px; border: 1px solid #882222;
            }
            QPushButton:hover { background-color: #aa2222; color: white; }
        """)
        self.btn_unload_file.clicked.connect(self.unload_file)
        self.btn_unload_file.setVisible(False)
        file_label_layout.addWidget(self.btn_unload_file)
        left_layout.addLayout(file_label_layout)

        filter_buttons = QHBoxLayout()
        filter_buttons.setSpacing(4)
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(lambda: self.toggle_all_selection(True))
        self.btn_select_all.setStyleSheet(SMALL_BTN)
        self.btn_deselect_all = QPushButton("Deselect All")
        self.btn_deselect_all.clicked.connect(lambda: self.toggle_all_selection(False))
        self.btn_deselect_all.setStyleSheet(SMALL_BTN)
        filter_buttons.addWidget(self.btn_select_all)
        filter_buttons.addWidget(self.btn_deselect_all)
        left_layout.addLayout(filter_buttons)

        select_mismatch_layout = QHBoxLayout()
        select_mismatch_layout.setSpacing(4)
        btn_select_untranslated = QPushButton("Select Untranslated")
        btn_select_untranslated.clicked.connect(lambda: self.toggle_selection_by_translated(False))
        btn_select_untranslated.setStyleSheet(SMALL_BTN)
        btn_select_mismatch = QPushButton("Select Mismatch")
        btn_select_mismatch.clicked.connect(lambda: self.toggle_selection_mismatch(True))
        btn_select_mismatch.setStyleSheet(SMALL_BTN)
        select_mismatch_layout.addWidget(btn_select_untranslated)
        select_mismatch_layout.addWidget(btn_select_mismatch)
        left_layout.addLayout(select_mismatch_layout)

        show_buttons = QHBoxLayout()
        show_buttons.setSpacing(4)
        btn_show_all = QPushButton("All")
        btn_show_all.clicked.connect(lambda: self.filter_list(None))
        btn_show_all.setStyleSheet(SMALL_BTN)
        btn_show_translated = QPushButton("Translated")
        btn_show_translated.clicked.connect(lambda: self.filter_list(True))
        btn_show_translated.setStyleSheet(SMALL_BTN)
        btn_show_untranslated = QPushButton("Untranslated")
        btn_show_untranslated.clicked.connect(lambda: self.filter_list(False))
        btn_show_untranslated.setStyleSheet(SMALL_BTN)
        btn_show_mismatch = QPushButton("Mismatch")
        btn_show_mismatch.clicked.connect(lambda: self.filter_mismatch(True))
        btn_show_mismatch.setStyleSheet(SMALL_BTN)
        show_buttons.addWidget(btn_show_all)
        show_buttons.addWidget(btn_show_translated)
        show_buttons.addWidget(btn_show_untranslated)
        show_buttons.addWidget(btn_show_mismatch)
        left_layout.addLayout(show_buttons)

        ALIGN_BTN = """
            QPushButton {
                background-color: #252525;
                color: #bbbbbb;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #333; color: white; border-color: #555; }
            QPushButton:pressed { background-color: #1a1a1a; }
        """
        self._alignment_filter_row = QWidget()
        align_filter_layout = QHBoxLayout(self._alignment_filter_row)
        align_filter_layout.setSpacing(3)
        align_filter_layout.setContentsMargins(0, 0, 0, 0)

        btn_af_green = QPushButton("🟢")
        btn_af_green.setToolTip("Auto-wrap or manually confirmed (100% certain)")
        btn_af_green.setStyleSheet(ALIGN_BTN)
        btn_af_green.clicked.connect(lambda: self._filter_alignment_dot('🟢'))

        btn_af_yellow = QPushButton("🟡")
        btn_af_yellow.setToolTip("Neural model, < 2 corrections (probably OK)")
        btn_af_yellow.setStyleSheet(ALIGN_BTN)
        btn_af_yellow.clicked.connect(lambda: self._filter_alignment_dot('🟡'))

        btn_af_orange = QPushButton("🟠")
        btn_af_orange.setToolTip("Neural model, ≥ 2 corrections (worth checking)")
        btn_af_orange.setStyleSheet(ALIGN_BTN)
        btn_af_orange.clicked.connect(lambda: self._filter_alignment_dot('🟠'))

        btn_af_black = QPushButton("⚫")
        btn_af_black.setToolTip("No inline tags in original (alignment N/A)")
        btn_af_black.setStyleSheet(ALIGN_BTN)
        btn_af_black.clicked.connect(lambda: self._filter_alignment_dot('⚫'))

        btn_af_red = QPushButton("🔴")
        btn_af_red.setToolTip("Manually flagged as bad")
        btn_af_red.setStyleSheet(ALIGN_BTN)
        btn_af_red.clicked.connect(lambda: self._filter_alignment_dot('🔴'))

        btn_af_pending = QPushButton("○")
        btn_af_pending.setToolTip("Translated but alignment not yet run")
        btn_af_pending.setStyleSheet(ALIGN_BTN)
        btn_af_pending.clicked.connect(lambda: self._filter_alignment_dot('none'))

        for btn in (btn_af_green, btn_af_yellow, btn_af_orange,
                    btn_af_black, btn_af_red, btn_af_pending):
            align_filter_layout.addWidget(btn)

        self._alignment_filter_row.setVisible(False)
        left_layout.addWidget(self._alignment_filter_row)

        search_layout = QHBoxLayout()
        search_layout.setSpacing(4)

        self.search_mode_combo = QComboBox()
        self.search_mode_combo.addItems(["Original", "Translation"])
        self.search_mode_combo.setStyleSheet(COMBO_STYLE)
        self.search_mode_combo.setMaximumWidth(100)
        search_layout.addWidget(self.search_mode_combo)

        search_container = QFrame()
        search_container.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
            }
            QFrame:focus-within {
                border-color: #2a6aaa;
            }
        """)
        search_inner = QHBoxLayout(search_container)
        search_inner.setContentsMargins(6, 0, 6, 0)
        search_inner.setSpacing(4)

        lbl_search_icon = QLabel("🔍")
        lbl_search_icon.setStyleSheet("background: transparent; border: none; font-size: 11px;")
        search_inner.addWidget(lbl_search_icon)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search fragments…")
        self.search_edit.setStyleSheet("""
            QLineEdit {
                background-color: transparent;
                color: white;
                border: none;
                padding: 4px 2px;
                font-size: 12px;
            }
        """)
        self.search_edit.textChanged.connect(self.filter_search)
        search_inner.addWidget(self.search_edit)
        search_layout.addWidget(search_container)
        left_layout.addLayout(search_layout)

        self.list_widget = QListWidget()
        self.list_widget.setItemDelegate(PreserveForegroundDelegate(self.list_widget))
        self.list_widget.setAutoScroll(False)
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #181818;
                border: 1px solid #333;
                border-radius: 3px;
                color: #dddddd;
                outline: none;
            }
            QListWidget::item { padding: 4px 8px; border-bottom: 1px solid #222; }
            QListWidget::item:selected { background-color: #1e3a5f; border-left: 3px solid #5a9aff; }
            QListWidget::item:hover { background-color: #252525; }
        """)
        self.list_widget.currentItemChanged.connect(self.display_selected_fragment)
        self.list_widget.itemClicked.connect(self.on_list_item_clicked)

        self._drag_scrolling = False
        self._drag_start_y = 0
        self._drag_start_scroll = 0
        self.list_widget.viewport().installEventFilter(self)

        left_layout.addWidget(self.list_widget)

        bottom_left_layout = QVBoxLayout()
        bottom_left_layout.setSpacing(4)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(6)
        grid_layout.setColumnStretch(3, 1)

        self.auto_fix_checkbox = QCheckBox("Auto-fix mismatch")
        self.auto_fix_checkbox.setToolTip(
            "Automatically retry translation if mismatch is detected.\n\n"
            "How it works:\n"
            "• After each translation, checks for formatting/structural errors\n"
            "• If mismatch found, retries with error details added to prompt\n"
            "• Temperature increases slightly each failed attempt (encourages variation)\n"
            "• Returns the best result across all attempts"
        )
        self.auto_fix_checkbox.setStyleSheet(CHECKBOX_STYLE)
        grid_layout.addWidget(self.auto_fix_checkbox, 0, 0)

        lbl_auto_fix_tries = QLabel("Attempts:")
        lbl_auto_fix_tries.setStyleSheet("color: #888; font-size: 11px;")
        grid_layout.addWidget(lbl_auto_fix_tries, 0, 1)

        self.auto_fix_spinbox = QSpinBox()
        self.auto_fix_spinbox.setRange(1, 10)
        self.auto_fix_spinbox.setValue(3)
        self.auto_fix_spinbox.setFixedWidth(70)
        self.auto_fix_spinbox.setStyleSheet(SPINBOX_STYLE)
        grid_layout.addWidget(self.auto_fix_spinbox, 0, 2)

        self.btn_cancel = QPushButton("✖ Cancel")
        self._hard_cancel_mode = False
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        self.btn_cancel.setToolTip("Click: finish the current section and stop")
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #4a1a1a; color: #ff9999;
                border: 1px solid #772222; border-radius: 4px;
                padding: 5px 12px; font-size: 11px; font-weight: bold;
            }
            QPushButton:hover { background-color: #6a2222; color: white; }
        """)
        grid_layout.addWidget(self.btn_cancel, 0, 3, 1, 2)

        self.sentence_batch_checkbox = QCheckBox("Sentence batching")
        self.sentence_batch_checkbox.setToolTip(
            "Combine multiple paragraphs into one LLM request.\n"
            "Faster translation by reducing API call overhead.\n"
            "Works in both Legacy and Inline mode.\n"
            "Paragraphs from different chapters are never merged."
        )
        self.sentence_batch_checkbox.setStyleSheet(CHECKBOX_STYLE)
        grid_layout.addWidget(self.sentence_batch_checkbox, 1, 0)

        self._lbl_batch_size = QLabel("Batch size:")
        self._lbl_batch_size.setStyleSheet("color: #888; font-size: 11px;")
        grid_layout.addWidget(self._lbl_batch_size, 1, 1)

        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(2, 20)
        self.batch_size_spinbox.setValue(5)
        self.batch_size_spinbox.setFixedWidth(70)
        self.batch_size_spinbox.setStyleSheet(SPINBOX_STYLE)
        grid_layout.addWidget(self.batch_size_spinbox, 1, 2)

        def _on_batch_checkbox_changed(state):
            enabled = self.sentence_batch_checkbox.isChecked()
            self.batch_size_spinbox.setEnabled(enabled)
            self._lbl_batch_size.setEnabled(enabled)
            self._update_context_visibility()
            if hasattr(self, 'llm_editor_container') and self.llm_editor_container.isVisible():
                self.update_llm_editor_content()

        self.sentence_batch_checkbox.stateChanged.connect(_on_batch_checkbox_changed)
        _on_batch_checkbox_changed(False)

        bottom_left_layout.addLayout(grid_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a1a; border: 1px solid #333;
                border-radius: 3px; height: 8px; text-align: center; color: transparent;
            }
            QProgressBar::chunk { background-color: #2a6aaa; border-radius: 2px; }
        """)
        bottom_left_layout.addWidget(self.progress_bar)

        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet("color: #888; font-size: 10px;")
        self.eta_label.setVisible(False)
        bottom_left_layout.addWidget(self.eta_label)

        left_layout.addLayout(bottom_left_layout)

        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(6)
        right_layout.setContentsMargins(4, 0, 0, 0)

        self.panel_tabs = QTabWidget()
        self.panel_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #2e2e2e; border-radius: 3px; }
            QTabBar::tab {
                background-color: #1e1e1e; color: #777;
                padding: 5px 14px;
                border-top-left-radius: 3px; border-top-right-radius: 3px;
                border: 1px solid #2e2e2e; border-bottom: none; margin-right: 2px;
            }
            QTabBar::tab:selected { background-color: #252525; color: #ccc; }
            QTabBar::tab:hover { background-color: #2a2a2a; color: #aaa; }
        """)

        translation_tab = QWidget()
        translation_tab_layout = QVBoxLayout(translation_tab)
        translation_tab_layout.setContentsMargins(0, 4, 0, 0)
        translation_tab_layout.setSpacing(0)

        preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        preview_splitter.setHandleWidth(5)

        orig_container = QWidget()
        orig_layout = QVBoxLayout(orig_container)
        orig_layout.setContentsMargins(0, 0, 0, 0)
        orig_layout.setSpacing(4)
        orig_lbl = QLabel("Original")
        orig_lbl.setStyleSheet(LABEL_STYLE)
        orig_layout.addWidget(orig_lbl)
        self.original_text_view = QTextEdit()
        self.original_text_view.setReadOnly(True)
        self.original_text_view.setStyleSheet(TEXT_VIEW_STYLE)
        orig_layout.addWidget(self.original_text_view)
        preview_splitter.addWidget(orig_container)

        trans_container = QWidget()
        trans_layout = QVBoxLayout(trans_container)
        trans_layout.setContentsMargins(0, 0, 0, 0)
        trans_layout.setSpacing(4)
        self.trans_lbl = QLabel("TRANSLATION  (EDITABLE)")
        self.trans_lbl.setStyleSheet(LABEL_STYLE)
        trans_layout.addWidget(self.trans_lbl)
        self.translated_text_view = QTextEdit()
        self.translated_text_view.textChanged.connect(self.update_translation_from_edit)
        self.translated_text_view.setStyleSheet(TEXT_VIEW_STYLE)
        trans_layout.addWidget(self.translated_text_view)
        preview_splitter.addWidget(trans_container)

        translation_tab_layout.addWidget(preview_splitter, stretch=1)
        self.panel_tabs.addTab(translation_tab, "Translation")

        alignment_tab = QWidget()
        alignment_tab_layout = QVBoxLayout(alignment_tab)
        alignment_tab_layout.setContentsMargins(0, 4, 0, 0)
        alignment_tab_layout.setSpacing(0)

        alignment_splitter = QSplitter(Qt.Orientation.Horizontal)
        alignment_splitter.setHandleWidth(5)

        orig_html_container = QWidget()
        orig_html_layout = QVBoxLayout(orig_html_container)
        orig_html_layout.setContentsMargins(0, 0, 0, 0)
        orig_html_layout.setSpacing(4)
        orig_html_lbl = QLabel("Original HTML (source of tags — read only)")
        orig_html_lbl.setStyleSheet(LABEL_STYLE)
        orig_html_layout.addWidget(orig_html_lbl)
        self.alignment_original_view = QPlainTextEdit()
        self.alignment_original_view.setReadOnly(True)
        self.alignment_original_view.setStyleSheet("""
            QPlainTextEdit {
                background-color: #161616; color: #9090a0;
                border: 1px solid #2e2e2e; border-radius: 4px;
                padding: 8px; font-size: 12px; font-family: monospace;
            }
        """)
        orig_html_layout.addWidget(self.alignment_original_view)
        alignment_splitter.addWidget(orig_html_container)

        aligned_result_container = QWidget()
        aligned_result_layout = QVBoxLayout(aligned_result_container)
        aligned_result_layout.setContentsMargins(0, 0, 0, 0)
        aligned_result_layout.setSpacing(4)
        aligned_result_lbl = QLabel("Alignment result — translation with inserted tags")
        aligned_result_lbl.setStyleSheet(LABEL_STYLE)
        aligned_result_layout.addWidget(aligned_result_lbl)
        self.alignment_result_view = QTextEdit()
        self.alignment_result_view.setStyleSheet(TEXT_VIEW_STYLE)
        self.alignment_result_view.setPlaceholderText("Run Alignment to see results")
        self.alignment_result_view.textChanged.connect(self._on_alignment_result_edited)
        aligned_result_layout.addWidget(self.alignment_result_view)
        alignment_splitter.addWidget(aligned_result_container)

        alignment_tab_layout.addWidget(alignment_splitter, stretch=1)
        self.panel_tabs.addTab(alignment_tab, "Alignment")
        self.panel_tabs.setTabVisible(1, False)

        preview_tab = QWidget()
        preview_tab_layout = QVBoxLayout(preview_tab)
        preview_tab_layout.setContentsMargins(0, 4, 0, 0)
        preview_tab_layout.setSpacing(0)

        self._preview_toolbar = EPUBPreviewToolbar()
        self._preview_toolbar.refresh_requested.connect(self.refresh_preview)
        self._preview_toolbar.chapter_changed.connect(self._on_preview_chapter_changed)
        self._preview_toolbar.dark_mode_toggled.connect(self._on_preview_dark_mode_toggled)
        self._preview_toolbar.reader_requested.connect(self._on_open_reader)
        preview_tab_layout.addWidget(self._preview_toolbar)

        self.epub_preview_view = QWebEngineView()
        self._preview_page = EPUBPreviewPage(self.epub_preview_view)
        self._preview_page.fragment_selected.connect(self._on_preview_fragment_clicked)
        self._preview_page.contextmenu_requested.connect(self._on_preview_contextmenu)
        self.epub_preview_view.setPage(self._preview_page)
        self.epub_preview_view.setStyleSheet(
            "QWebEngineView { background-color: #ffffff; border: none; }"
        )
        self.epub_preview_view.setHtml(
            '<html><body style="background:#1a1a1a;margin:0;padding:24px;">'
            '<p style="color:#666;font-family:sans-serif;font-size:13px;">'
            'Select an EPUB fragment from the list to see the visual preview.'
            '</p></body></html>'
        )
        preview_tab_layout.addWidget(self.epub_preview_view, stretch=1)
        self.panel_tabs.addTab(preview_tab, "Preview")
        self.panel_tabs.setTabVisible(2, False)

        self.panel_tabs.currentChanged.connect(self._on_panel_tab_changed)

        right_layout.addWidget(self.panel_tabs, stretch=1)

        self.controls_bar = QWidget()
        self.controls_bar.setStyleSheet(
            "background-color: #1c1c1c; border: 1px solid #2e2e2e; border-radius: 5px;"
        )
        controls_bar_layout = QVBoxLayout(self.controls_bar)
        controls_bar_layout.setContentsMargins(8, 6, 8, 6)
        controls_bar_layout.setSpacing(6)

        qt_row = QHBoxLayout()
        qt_row.setSpacing(6)

        self.btn_copy_original = QPushButton("→  Copy Original")
        self.btn_copy_original.setToolTip("Copy original text to translation field")
        self.btn_copy_original.clicked.connect(self.copy_original_to_translation)
        self.btn_copy_original.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a; color: #aaa;
                border: 1px solid #444; border-radius: 3px; padding: 4px 10px; font-size: 11px;
            }
            QPushButton:hover { background-color: #333; color: white; }
        """)
        qt_row.addWidget(self.btn_copy_original)

        self._qt_sep1 = QFrame()
        self._qt_sep1.setFrameShape(QFrame.Shape.VLine)
        self._qt_sep1.setStyleSheet("color: #333;")
        qt_row.addWidget(self._qt_sep1)

        self.quick_translate_service_combo = QComboBox()
        self.quick_translate_service_combo.addItems(["Google (Free)", "DeepL Free", "DeepL Pro"])
        self.quick_translate_service_combo.setMinimumWidth(130)
        self.quick_translate_service_combo.setStyleSheet(COMBO_STYLE)
        qt_row.addWidget(self.quick_translate_service_combo)

        self._qt_lbl_from = QLabel("From:")
        self._qt_lbl_from.setStyleSheet("color: #777; font-size: 11px;")
        qt_row.addWidget(self._qt_lbl_from)

        self.source_lang_combo = QComboBox()
        for n, c in LanguageConstants.SOURCE_LANGUAGES:
            self.source_lang_combo.addItem(n)
            self.source_lang_combo.setItemData(self.source_lang_combo.count() - 1, c, Qt.ItemDataRole.UserRole)
        self.source_lang_combo.setCurrentText("English")
        self.source_lang_combo.setMaximumWidth(180)
        self.source_lang_combo.setStyleSheet(COMBO_STYLE)
        qt_row.addWidget(self.source_lang_combo)

        self._qt_lbl_arrow = QLabel("→")
        self._qt_lbl_arrow.setStyleSheet("color: #555; font-size: 13px;")
        qt_row.addWidget(self._qt_lbl_arrow)

        self._qt_lbl_to = QLabel("To:")
        self._qt_lbl_to.setStyleSheet("color: #777; font-size: 11px;")
        qt_row.addWidget(self._qt_lbl_to)

        self.target_lang_combo = QComboBox()
        for n, c in LanguageConstants.TARGET_LANGUAGES:
            self.target_lang_combo.addItem(n)
            self.target_lang_combo.setItemData(self.target_lang_combo.count() - 1, c, Qt.ItemDataRole.UserRole)
        self.target_lang_combo.setCurrentText("Polish")
        self.target_lang_combo.setMaximumWidth(180)
        self.target_lang_combo.setStyleSheet(COMBO_STYLE)
        qt_row.addWidget(self.target_lang_combo)

        self.btn_quick_translate = QPushButton("Translate")
        self.btn_quick_translate.setStyleSheet("""
            QPushButton {
                background-color: #1a4a7a; color: white;
                font-weight: bold; padding: 5px 14px;
                border-radius: 3px; border: 1px solid #2a6aaa; font-size: 12px;
            }
            QPushButton:hover { background-color: #2a5a9a; }
            QPushButton:pressed { background-color: #0a3a6a; }
        """)
        self.btn_quick_translate.clicked.connect(self.translate_with_quick_service)
        qt_row.addWidget(self.btn_quick_translate)

        qt_row.addStretch()

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #333;")
        qt_row.addWidget(sep2)

        self.btn_mark_correct = QPushButton("✓  Mark Correct")
        self.btn_mark_correct.setToolTip("Mark selected fragment as correct (ignore mismatch)")
        self.btn_mark_correct.setStyleSheet("""
            QPushButton {
                background-color: #1e3d0a; color: #88dd44;
                border: 1px solid #3a6a15; border-radius: 3px; padding: 4px 10px;
                font-weight: bold; font-size: 11px;
            }
            QPushButton:hover { background-color: #2a5a12; color: #aaff66; }
        """)
        self.btn_mark_correct.clicked.connect(self.mark_fragment_as_correct)
        qt_row.addWidget(self.btn_mark_correct)

        self.btn_unmark_correct = QPushButton("✗  Unmark")
        self.btn_unmark_correct.setToolTip("Remove 'correct' flag and recheck mismatch")
        self.btn_unmark_correct.setStyleSheet("""
            QPushButton {
                background-color: #3a0e0e; color: #dd6666;
                border: 1px solid #662020; border-radius: 3px; padding: 4px 10px;
                font-weight: bold; font-size: 11px;
            }
            QPushButton:hover { background-color: #5a1515; color: #ff8888; }
        """)
        self.btn_unmark_correct.clicked.connect(self.unmark_fragment_as_correct)
        qt_row.addWidget(self.btn_unmark_correct)

        controls_bar_layout.addLayout(qt_row)
        right_layout.addWidget(self.controls_bar)

        self.llm_options_widget = QWidget()
        llm_options_layout = QVBoxLayout(self.llm_options_widget)
        llm_options_layout.setContentsMargins(0, 0, 0, 0)
        llm_options_layout.setSpacing(4)

        llm_buttons_layout = QHBoxLayout()
        btn_toggle_llm_editor = QPushButton("⚙  LLM Options & Instructions")
        btn_toggle_llm_editor.setStyleSheet("""
            QPushButton {
                font-weight: bold; padding: 8px 16px;
                background-color: #2e2e2e; color: #cccccc;
                border: 1px solid #444; border-radius: 4px;
            }
            QPushButton:hover { background-color: #3a3a3a; color: white; }
            QPushButton:checked { background-color: #1e3a5f; border-color: #2a6aaa; color: white; }
        """)
        btn_toggle_llm_editor.clicked.connect(self.toggle_llm_editor)
        llm_buttons_layout.addWidget(btn_toggle_llm_editor)
        llm_buttons_layout.addStretch()
        llm_options_layout.addLayout(llm_buttons_layout)

        self.llm_editor_container = QWidget()
        self.llm_editor_container.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                border: 1px solid #2e2e2e;
                border-radius: 4px;
            }
        """)
        editor_container_layout = QVBoxLayout(self.llm_editor_container)
        editor_container_layout.setContentsMargins(10, 8, 10, 8)
        editor_container_layout.setSpacing(8)

        params_group = QFrame()
        params_group.setStyleSheet("""
            QFrame {
                background-color: #222;
                border: 1px solid #333;
                border-radius: 3px;
            }
        """)
        params_group_layout = QHBoxLayout(params_group)
        params_group_layout.setContentsMargins(10, 6, 10, 6)
        params_group_layout.setSpacing(12)

        def param_pair(label_text, widget):
            lbl = QLabel(label_text)
            lbl.setStyleSheet(PARAM_LABEL + " background: transparent; border: none;")
            params_group_layout.addWidget(lbl)
            params_group_layout.addWidget(widget)
            return lbl

        self.temperature_spinbox = QDoubleSpinBox()
        self.temperature_spinbox.setRange(0.0, 1.0)
        self.temperature_spinbox.setSingleStep(0.05)
        self.temperature_spinbox.setValue(0.8)
        self.temperature_spinbox.setFixedWidth(78)
        self.temperature_spinbox.setStyleSheet(SPINBOX_STYLE)
        param_pair("🌡  Temperature:", self.temperature_spinbox)

        sep_v1 = QFrame()
        sep_v1.setFrameShape(QFrame.Shape.VLine)
        sep_v1.setStyleSheet("background-color: #333; border: none; max-width: 1px;")
        params_group_layout.addWidget(sep_v1)

        self.timeout_spinbox = QSpinBox()
        self.timeout_spinbox.setRange(1, 60)
        self.timeout_spinbox.setValue(10)
        self.timeout_spinbox.setFixedWidth(78)
        self.timeout_spinbox.setToolTip("Max time per fragment before skipping (default: 10 min)")
        self.timeout_spinbox.setStyleSheet(SPINBOX_STYLE)
        param_pair("⏱  Timeout (min):", self.timeout_spinbox)

        sep_v2 = QFrame()
        sep_v2.setFrameShape(QFrame.Shape.VLine)
        sep_v2.setStyleSheet("background-color: #333; border: none; max-width: 1px;")
        params_group_layout.addWidget(sep_v2)

        self.context_before_spinbox = QSpinBox()
        self.context_before_spinbox.setRange(0, 99999)
        self.context_before_spinbox.setValue(3)
        self.context_before_spinbox.setFixedWidth(78)
        self.context_before_spinbox.setStyleSheet(SPINBOX_STYLE)
        self._lbl_context_before = param_pair("📄  Prev paragraphs:", self.context_before_spinbox)

        self.context_after_spinbox = QSpinBox()
        self.context_after_spinbox.setRange(0, 99999)
        self.context_after_spinbox.setValue(2)
        self.context_after_spinbox.setFixedWidth(78)
        self.context_after_spinbox.setStyleSheet(SPINBOX_STYLE)
        self._lbl_context_after = param_pair("Next paragraphs:", self.context_after_spinbox)

        params_group_layout.addStretch()
        editor_container_layout.addWidget(params_group)

        editor_header_layout = QHBoxLayout()
        editor_header_layout.setSpacing(6)

        lbl_editor = QLabel("LLM Instructions Editor:")
        lbl_editor.setStyleSheet("color: #999; font-size: 11px; background: transparent; border: none;")
        editor_header_layout.addWidget(lbl_editor)
        editor_header_layout.addStretch()

        self.single_prompt_checkbox = QCheckBox("Single-prompt mode")
        self.single_prompt_checkbox.setToolTip(
            "Merge all instructions into one prompt (no system/assistant/user roles).\n"
            "Use for instruct-only models (e.g., Gemma) or to avoid Channel Errors."
        )
        self.single_prompt_checkbox.setStyleSheet(CHECKBOX_STYLE)
        editor_header_layout.addWidget(self.single_prompt_checkbox)

        self.json_payload_checkbox = QCheckBox("JSON Payload mode")
        self.json_payload_checkbox.setToolTip(
            "Send a raw JSON payload instead of chat messages.\n"
            "Use for models that require a custom JSON input format.\n\n"
            "Available variables:\n"
            "  {core_text}       — text to translate\n"
            "  {context_before}  — previous paragraphs (for context)\n"
            "  {context_after}   — following paragraphs (for context)\n\n"
            "Temperature is injected automatically from the UI slider.\n"
            "Do not include 'temperature' in the template.\n\n"
            "Response field (leave empty for auto-detect):\n"
            "  translation                     → {\"translation\": \"...\"}\n"
            "  choices.0.message.content       → standard chat response\n"
            "  choices.0.message.content.translation → JSON inside content"
        )
        self.json_payload_checkbox.setStyleSheet(CHECKBOX_STYLE)
        self.json_payload_checkbox.stateChanged.connect(self._on_json_payload_toggled)
        editor_header_layout.addWidget(self.json_payload_checkbox)

        self.json_response_field_label = QLabel("Response field:")
        self.json_response_field_label.setStyleSheet(
            "color: #999; font-size: 11px; background: transparent; border: none;"
        )
        self.json_response_field_label.setVisible(False)
        editor_header_layout.addWidget(self.json_response_field_label)

        self.json_response_field_edit = QLineEdit()
        self.json_response_field_edit.setPlaceholderText("leave empty to auto-detect")
        self.json_response_field_edit.setFixedWidth(160)
        self.json_response_field_edit.setToolTip(
            "JSON path to extract the translation from the server response.\n"
            "Supports dot notation and traversal through JSON strings.\n\n"
            "Examples:\n"
            "  translation                          → {\"translation\": \"...\"}\n"
            "  choices.0.message.content            → standard chat content\n"
            "  choices.0.message.content.translation → JSON string inside content\n\n"
            "Leave empty to auto-detect common keys:\n"
            "translation, Translation, text, result, output, translated"
        )
        self.json_response_field_edit.setVisible(False)
        editor_header_layout.addWidget(self.json_response_field_edit)

        self.btn_hard_reset = QPushButton("🗑  Hard Reset")
        self.btn_hard_reset.setToolTip("Permanently deletes your instruction files and restores factory defaults.")
        self.btn_hard_reset.setStyleSheet(f"background-color: #880000; {EDITOR_BTN}")
        self.btn_hard_reset.clicked.connect(self.hard_reset_llm_instruction)
        self.btn_hard_reset.setVisible(False)
        editor_header_layout.addWidget(self.btn_hard_reset)

        self.btn_reset_llm = QPushButton("↩  Reset to Default")
        self.btn_reset_llm.setToolTip("Reverts unsaved changes to last saved state.")
        self.btn_reset_llm.setStyleSheet(f"background-color: #cc6600; {EDITOR_BTN}")
        self.btn_reset_llm.clicked.connect(self.reset_llm_instruction)
        self.btn_reset_llm.setVisible(False)
        editor_header_layout.addWidget(self.btn_reset_llm)

        self.btn_save_llm_instruction = QPushButton("💾  Save LLM Instruction")
        self.btn_save_llm_instruction.setStyleSheet(f"background-color: #007700; {EDITOR_BTN}")
        self.btn_save_llm_instruction.clicked.connect(self.save_llm_instruction)
        self.btn_save_llm_instruction.setVisible(False)
        editor_header_layout.addWidget(self.btn_save_llm_instruction)

        editor_container_layout.addLayout(editor_header_layout)

        self.llm_editor_widget = QWidget()
        self.llm_editor_widget.setStyleSheet("background: transparent; border: none;")
        self.llm_editor_layout = QVBoxLayout(self.llm_editor_widget)
        self.llm_editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_container_layout.addWidget(self.llm_editor_widget)

        self.llm_editor_container.setVisible(False)
        llm_options_layout.addWidget(self.llm_editor_container)
        right_layout.addWidget(self.llm_options_widget)

        self.action_buttons_widget = QWidget()
        action_buttons_layout = QHBoxLayout(self.action_buttons_widget)
        action_buttons_layout.setContentsMargins(0, 0, 0, 0)
        action_buttons_layout.setSpacing(8)

        self.btn_translate = QPushButton("▶  Translate Selected")
        self.btn_translate.clicked.connect(self.start_translation)
        self.btn_translate.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_translate.setMinimumHeight(38)
        self.btn_translate.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e5fa8, stop:1 #133f74);
                color: white; font-weight: bold; font-size: 13px;
                padding: 8px 24px; border-radius: 5px; border: 1px solid #2a6aaa;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a72c8, stop:1 #1a4e8c);
                border-color: #4a8acc;
            }
            QPushButton:pressed { background: #0e3060; }
        """)

        self.btn_save_file = QPushButton("💾  Save as New File")
        self.btn_save_file.clicked.connect(self.save_file)
        self.btn_save_file.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_file.setMinimumHeight(38)
        self.btn_save_file.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e7a3a, stop:1 #125225);
                color: white; font-weight: bold; font-size: 13px;
                padding: 8px 24px; border-radius: 5px; border: 1px solid #2a9a4a;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a9a4a, stop:1 #1a6a30);
                border-color: #4acc6a;
            }
            QPushButton:pressed { background: #0e4020; }
        """)

        self.btn_alignment = QPushButton("🔡  Run Alignment")
        self.btn_alignment.clicked.connect(self.start_alignment)
        self.btn_alignment.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_alignment.setMinimumHeight(38)
        self.btn_alignment.setVisible(False)
        self.btn_alignment.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a2a7a, stop:1 #2e1a50);
                color: white; font-weight: bold; font-size: 13px;
                padding: 8px 24px; border-radius: 5px; border: 1px solid #7a4aaa;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6a3a9a, stop:1 #3e2a70);
                border-color: #9a6acc;
            }
            QPushButton:pressed { background: #1e0e40; }
        """)

        action_buttons_layout.addWidget(self.btn_translate)
        action_buttons_layout.addWidget(self.btn_save_file)
        action_buttons_layout.addWidget(self.btn_alignment)
        right_layout.addWidget(self.action_buttons_widget)

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900])
        translator_layout.addWidget(splitter)

        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; border-radius: 3px; }
            QTabBar::tab {
                background-color: #222; color: #888;
                padding: 7px 20px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                border: 1px solid #333; border-bottom: none; margin-right: 2px;
            }
            QTabBar::tab:selected { background-color: #2a2a2a; color: white; border-bottom: 1px solid #2a2a2a; }
            QTabBar::tab:hover { background-color: #2e2e2e; color: #ccc; }
        """)
        tab_widget.addTab(translator_widget, "Translator")
        tab_widget.addTab(self.init_options_tab(), "Options")

        self.setCentralWidget(tab_widget)

    def init_options_tab(self):
        INPUT_STYLE = """
            QLineEdit {
                background-color: #161616; color: #e0e0e0;
                border: 1px solid #2e2e2e; border-radius: 3px;
                padding: 5px 8px; font-size: 12px;
            }
            QLineEdit:focus { border-color: #2a6aaa; }
        """
        COMBO_STYLE = """
            QComboBox {
                padding: 4px 8px; border: 1px solid #444; border-radius: 3px;
                background-color: #252525; color: white; font-size: 11px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a; color: white;
                selection-background-color: #3a5a8a;
            }
        """
        LABEL_STYLE = "color: #888888; font-size: 11px;"
        CHECKBOX_STYLE = """
            QCheckBox { color: #cccccc; font-size: 12px; spacing: 8px; }
            QCheckBox::indicator {
                width: 16px; height: 16px; border-radius: 3px;
                border: 2px solid #555555; background-color: #1e1e1e;
            }
            QCheckBox::indicator:hover { border-color: #2a6aaa; background-color: #252535; }
            QCheckBox::indicator:checked { border-color: #2a6aaa; background-color: #1a4a7a; }
            QCheckBox::indicator:checked:hover { background-color: #2a5a9a; }
            QCheckBox::indicator:disabled { border-color: #333; background-color: #1a1a1a; }
        """
        CHECKBOX_SMALL_STYLE = """
            QCheckBox { color: #bbbbbb; font-size: 11px; spacing: 6px; }
            QCheckBox::indicator {
                width: 14px; height: 14px; border-radius: 3px;
                border: 2px solid #555555; background-color: #1e1e1e;
            }
            QCheckBox::indicator:hover { border-color: #2a6aaa; background-color: #252535; }
            QCheckBox::indicator:checked { border-color: #2a6aaa; background-color: #1a4a7a; }
            QCheckBox::indicator:checked:hover { background-color: #2a5a9a; }
        """
        SPINBOX_STYLE = """
            QSpinBox, QDoubleSpinBox {
                background-color: #1e1e1e; color: white;
                border: 1px solid #3c3c3c; border-radius: 3px;
                padding: 3px 6px; font-size: 12px; min-height: 26px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border; subcontrol-position: top right;
                width: 22px; height: 13px; background-color: #333;
                border-left: 1px solid #444; border-bottom: 1px solid #444;
                border-top-right-radius: 3px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border; subcontrol-position: bottom right;
                width: 22px; height: 13px; background-color: #333;
                border-left: 1px solid #444; border-bottom-right-radius: 3px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #2a5a9a;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background-color: #1a3a6a;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: none; width: 0; height: 0;
                border-left: 4px solid transparent; border-right: 4px solid transparent;
                border-bottom: 5px solid #aaa;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none; width: 0; height: 0;
                border-left: 4px solid transparent; border-right: 4px solid transparent;
                border-top: 5px solid #aaa;
            }
        """
        TOGGLE_BTN_STYLE = """
            QToolButton {{
                background-color: {bg}; color: {fg};
                border: 1px solid {border}; border-radius: 5px;
                padding: 7px 12px; font-size: 9pt; font-weight: bold;
                text-align: left;
            }}
            QToolButton:hover {{
                background-color: {bg_h}; border-color: {border_h}; color: {fg_h};
            }}
            QToolButton:checked {{
                background-color: {bg_c}; border-color: {border_c}; color: {fg_c};
                border-bottom-left-radius: 0px; border-bottom-right-radius: 0px;
            }}
        """
        SECTION_BODY_STYLE = """
            QWidget#sectionBody {{
                background-color: {bg};
                border: 1px solid {border};
                border-top: none;
                border-bottom-left-radius: 5px;
                border-bottom-right-radius: 5px;
            }}
        """

        def form_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(LABEL_STYLE)
            return lbl

        def make_collapsible(title, content_widget, expanded=False,
                             color_scheme="default"):
            schemes = {
                "default": dict(bg="#222222", fg="#aaaaaa", border="#333333",
                                bg_h="#2a2a2a", border_h="#444444", fg_h="#cccccc",
                                bg_c="#1c2025", border_c="#3a4a5a", fg_c="#99bbdd",
                                body_bg="#1c1c1c", body_border="#2e2e2e"),
                "blue":    dict(bg="#1a1f2a", fg="#7ab8dd", border="#2a4a6a",
                                bg_h="#1e2535", border_h="#3a6a9a", fg_h="#9acfee",
                                bg_c="#141e28", border_c="#2a6aaa", fg_c="#aad4f0",
                                body_bg="#141e28", body_border="#2a4a6a"),
                "green":   dict(bg="#1a221a", fg="#88aa88", border="#2a4a2a",
                                bg_h="#1e281e", border_h="#3a6a3a", fg_h="#aaccaa",
                                bg_c="#162216", border_c="#2a6a2a", fg_c="#aadcaa",
                                body_bg="#1c1c1c", body_border="#2a4a3a"),
                "amber":   dict(bg="#221e10", fg="#b89a44", border="#4a3e1a",
                                bg_h="#282210", border_h="#6a5a2a", fg_h="#d4b860",
                                bg_c="#1c1a0e", border_c="#6a5a2a", fg_c="#d4b860",
                                body_bg="#1c1c1c", body_border="#3a3a2a"),
                "purple":  dict(bg="#1e1a28", fg="#9977cc", border="#3a2e5a",
                                bg_h="#251e32", border_h="#5a4a8a", fg_h="#bbaaee",
                                bg_c="#1a1525", border_c="#5a4a8a", fg_c="#ccbbee",
                                body_bg="#1a1822", body_border="#3a2e4a"),
            }
            s = schemes.get(color_scheme, schemes["default"])

            container = QWidget()
            container.setStyleSheet("background: transparent;")
            c_layout = QVBoxLayout(container)
            c_layout.setContentsMargins(0, 0, 0, 0)
            c_layout.setSpacing(0)

            btn = QToolButton()
            btn.setCheckable(True)
            btn.setChecked(expanded)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            def _update_btn_text(checked):
                arrow = "▼" if checked else "▶"
                btn.setText(f"  {arrow}  {title}")

            _update_btn_text(expanded)
            btn.setStyleSheet(TOGGLE_BTN_STYLE.format(**s))
            c_layout.addWidget(btn)

            body = QWidget()
            body.setObjectName("sectionBody")
            body.setStyleSheet(SECTION_BODY_STYLE.format(
                bg=s["body_bg"], border=s["body_border"]
            ))
            body.setVisible(expanded)
            c_layout.addWidget(body)

            body_layout = QVBoxLayout(body)
            body_layout.setContentsMargins(12, 10, 12, 12)
            body_layout.setSpacing(8)
            body_layout.addWidget(content_widget)

            btn.toggled.connect(_update_btn_text)
            btn.toggled.connect(body.setVisible)

            return container

        tab_root = QWidget()
        tab_root.setStyleSheet("background-color: #1a1a1a; color: #cccccc;")

        scroll_area = QScrollArea(tab_root)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea { border: none; background-color: #1a1a1a; }
            QScrollBar:vertical {
                background: #1e1e1e; width: 8px; margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #3a3a3a; border-radius: 4px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background: #4a4a4a; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: #1a1a1a;")
        outer_layout = QVBoxLayout(scroll_content)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.setSpacing(6)

        scroll_area.setWidget(scroll_content)

        tab_layout = QVBoxLayout(tab_root)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll_area)

        info_widget = QWidget()
        info_widget.setStyleSheet("""
            QWidget {
                background-color: #141e28;
                border: 1px solid #2a4a6a;
                border-radius: 6px;
            }
        """)
        info_inner = QVBoxLayout(info_widget)
        info_inner.setContentsMargins(10, 8, 10, 10)
        info_inner.setSpacing(3)

        info_title = QLabel("ℹ️  How Settings Work")
        info_title.setStyleSheet("color: #7ab8dd; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        info_inner.addWidget(info_title)

        info_rows = [
            ("#8aaabb",
             "• <b>Processing Mode</b> and <b>Skip Tags</b> apply to EPUB files only — ignored for SRT and TXT."),
            ("#8aaabb",
             "• After switching Inline ↔ Legacy on an already-loaded file, reload the file to avoid inconsistencies."),
            ("#3a9a5a",
             "🟢 <b>Instant (no save needed):</b>  Mismatch checkboxes · Length / untranslated ratio thresholds · "
             "Tag position shift threshold &lt;id_xx&gt; &lt;nt_xx/&gt; · Inline position shift &lt;p_xx&gt; · "
             "Restore paragraph structure · Show &lt;ps&gt; in UI · Prompt text fields."),
            ("#b8902a",
             "💾 <b>Requires Save Settings:</b>  LLM provider · API keys."),
            ("#b8902a",
             "🔄 <b>Requires file reload:</b>  Switching Inline ↔ Legacy after file is loaded "
             "(save first, then reload the file)."),
        ]
        for color, text in info_rows:
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                f"color: {color}; font-size: 11px; font-weight: normal; "
                f"padding: 1px 0; background: transparent; border: none;"
            )
            info_inner.addWidget(lbl)

        outer_layout.addWidget(info_widget)

        engine_content = QWidget()
        engine_content.setStyleSheet("background: transparent;")
        engine_form = QFormLayout(engine_content)
        engine_form.setContentsMargins(0, 0, 0, 0)
        engine_form.setSpacing(10)

        self.llm_choice_combo = QComboBox()
        self.llm_choice_combo.addItems(["LM Studio", "Ollama", "Openrouter"])
        self.llm_choice_combo.setStyleSheet(COMBO_STYLE)
        self.llm_choice_combo.currentTextChanged.connect(self.update_model_name_visibility)
        engine_form.addRow(form_label("Select LLM:"), self.llm_choice_combo)

        self.server_url_label = form_label("Server URL:")
        self.server_url_edit = QLineEdit()
        self.server_url_edit.setPlaceholderText("e.g., http://localhost:1234/v1/chat/completions")
        self.server_url_edit.setStyleSheet(INPUT_STYLE)
        self.server_url_edit.setToolTip(
            "Custom endpoint URL for LM Studio, Lemonade, or any OpenAI-compatible server.\n"
            "Leave empty to use the default: http://localhost:1234/v1/chat/completions"
        )
        engine_form.addRow(self.server_url_label, self.server_url_edit)

        self.ollama_model_label = form_label("Ollama Model Name:")
        self.ollama_model_edit = QLineEdit()
        self.ollama_model_edit.setPlaceholderText("e.g., llama3.2:3b")
        self.ollama_model_edit.setStyleSheet(INPUT_STYLE)
        engine_form.addRow(self.ollama_model_label, self.ollama_model_edit)

        self.openrouter_api_key_label = form_label("Openrouter API Key:")
        self.openrouter_api_key_edit = QLineEdit()
        self.openrouter_api_key_edit.setPlaceholderText("Enter your Openrouter API key")
        self.openrouter_api_key_edit.setStyleSheet(INPUT_STYLE)
        engine_form.addRow(self.openrouter_api_key_label, self.openrouter_api_key_edit)

        self.openrouter_model_label = form_label("Openrouter Model Name:")
        self.openrouter_model_edit = QLineEdit()
        self.openrouter_model_edit.setPlaceholderText("e.g., openai/gpt-4o or openai/gpt-4o:free")
        self.openrouter_model_edit.setStyleSheet(INPUT_STYLE)
        engine_form.addRow(self.openrouter_model_label, self.openrouter_model_edit)

        self.openrouter_free_warning = QFrame()
        self.openrouter_free_warning.setStyleSheet("""
            QFrame { background-color: #1a1500; border: 1px solid #7a6000; border-radius: 4px; margin-top: 2px; }
        """)
        free_warning_layout = QVBoxLayout(self.openrouter_free_warning)
        free_warning_layout.setContentsMargins(10, 8, 10, 8)
        free_warning_layout.setSpacing(4)
        free_warning_title = QLabel("⚠️  Using free models (model name ending with <b>:free</b>)?")
        free_warning_title.setStyleSheet(
            "color: #ddaa00; font-size: 11px; font-weight: bold; background: transparent; border: none;"
        )
        free_warning_title.setWordWrap(True)
        free_warning_layout.addWidget(free_warning_title)
        free_warning_text = QLabel(
            "Free models on OpenRouter require you to enable the following option in your account settings:<br>"
            "<b>\"Enable free endpoints that may publish prompts\"</b><br>"
            "<span style='color: #888;'>Allow free model providers to publish your prompts and completions to public datasets.</span><br><br>"
            "Without this, every request will fail with a 404 error — even if the model name is correct.<br>"
            "Click the link below to configure:"
        )
        free_warning_text.setStyleSheet(
            "color: #ccaa55; font-size: 11px; background: transparent; border: none;"
        )
        free_warning_text.setWordWrap(True)
        free_warning_layout.addWidget(free_warning_text)
        free_warning_link = QLabel(
            '<a href="https://openrouter.ai/settings/privacy" style="color: #4a9eff; font-size: 12px; font-weight: bold;">'
            '🔗 openrouter.ai/settings/privacy</a>'
        )
        free_warning_link.setStyleSheet("background: transparent; border: none;")
        free_warning_link.setOpenExternalLinks(True)
        free_warning_link.setTextFormat(Qt.TextFormat.RichText)
        free_warning_layout.addWidget(free_warning_link)
        engine_form.addRow(self.openrouter_free_warning)

        separator1 = QLabel()
        separator1.setFixedHeight(1)
        separator1.setStyleSheet("background-color: #2e2e2e; margin: 4px 0;")
        engine_form.addRow(separator1)

        self.deepl_free_api_key_edit = QLineEdit()
        self.deepl_free_api_key_edit.setPlaceholderText("DeepL Free API Key")
        self.deepl_free_api_key_edit.setStyleSheet(INPUT_STYLE)
        engine_form.addRow(form_label("DeepL Free Key:"), self.deepl_free_api_key_edit)

        self.deepl_pro_api_key_edit = QLineEdit()
        self.deepl_pro_api_key_edit.setPlaceholderText("DeepL Pro API Key")
        self.deepl_pro_api_key_edit.setStyleSheet(INPUT_STYLE)
        engine_form.addRow(form_label("DeepL Pro Key:"), self.deepl_pro_api_key_edit)

        outer_layout.addWidget(
            make_collapsible("🔧  Engine Settings", engine_content,
                             expanded=True, color_scheme="default")
        )

        para_content = QWidget()
        para_content.setStyleSheet("background: transparent;")
        para_content_layout = QVBoxLayout(para_content)
        para_content_layout.setContentsMargins(0, 0, 0, 0)
        para_content_layout.setSpacing(6)

        self.restore_paragraph_checkbox = QCheckBox(
            "Restore paragraph structure after translation  "
            "(Deepl/Google and inline: <ps> markers,  legacy: proportional split)"
        )
        self.restore_paragraph_checkbox.setStyleSheet(CHECKBOX_STYLE)
        self.restore_paragraph_checkbox.setToolTip(
            "<html>"
            "When <b>enabled</b>, paragraph breaks inside multi-paragraph elements are preserved:<br>"
            "  • <b>EPUB Inline:</b> boundaries sent as <b>&lt;ps&gt;</b> markers, restored after LLM<br>"
            "  • <b>EPUB Legacy / TXT:</b> proportional split at word boundaries after LLM<br><br>"
            "When <b>disabled</b>, all newlines are flattened to a single paragraph before<br>"
            "sending to LLM and NO restoration is attempted."
            "</html>"
        )
        self.restore_paragraph_checkbox.setChecked(
            self.app_settings.get('restore_paragraph_structure', True)
        )
        para_content_layout.addWidget(self.restore_paragraph_checkbox)

        self._para_sub_container = QWidget()
        self._para_sub_container.setStyleSheet("background: transparent;")
        para_sub_layout = QVBoxLayout(self._para_sub_container)
        para_sub_layout.setContentsMargins(28, 2, 0, 0)
        para_sub_layout.setSpacing(4)

        self.show_ps_in_ui_checkbox = QCheckBox(
            "Show <ps> markers in text preview  (display paragraph breaks as <ps>)"
        )
        self.show_ps_in_ui_checkbox.setStyleSheet(CHECKBOX_STYLE)
        self.show_ps_in_ui_checkbox.setToolTip(
            "<html>"
            "When enabled, paragraph breaks in the text preview are shown as literal<br>"
            "<b>&lt;ps&gt;</b> markers so you can see where paragraph boundaries are.<br><br>"
            "This is <b>display only</b> — stored data and saved files are not affected.<br>"
            "Only active in EPUB Inline mode with paragraph structure restoration enabled."
            "</html>"
        )
        self.show_ps_in_ui_checkbox.setChecked(self.app_settings.get('show_ps_in_ui', False))
        para_sub_layout.addWidget(self.show_ps_in_ui_checkbox)
        para_content_layout.addWidget(self._para_sub_container)

        def _update_para_sub_visibility():
            is_inline = self.inline_formatting_checkbox.isChecked()
            is_restore = self.restore_paragraph_checkbox.isChecked()
            visible = is_inline and is_restore
            self._para_sub_container.setVisible(visible)
            if visible:
                self.show_ps_in_ui_checkbox.setChecked(True)
            else:
                self.show_ps_in_ui_checkbox.setChecked(False)

        self.restore_paragraph_checkbox.stateChanged.connect(
            lambda _: self._refresh_current_fragment_display()
        )
        self.restore_paragraph_checkbox.stateChanged.connect(
            lambda _: _update_para_sub_visibility()
        )
        self.show_ps_in_ui_checkbox.stateChanged.connect(
            lambda _: self._refresh_current_fragment_display()
        )

        outer_layout.addWidget(
            make_collapsible("📝  Paragraph Structure Restoration", para_content,
                             expanded=False, color_scheme="green")
        )

        mismatch_content = QWidget()
        mismatch_content.setStyleSheet("background: transparent;")
        mismatch_outer_layout = QVBoxLayout(mismatch_content)
        mismatch_outer_layout.setContentsMargins(0, 0, 0, 0)
        mismatch_outer_layout.setSpacing(10)

        saved_checks = self.app_settings.get("mismatch_checks", {})

        mismatch_checks_info = [
            ("paragraphs", "Paragraph / line count",
             "Detects when the number of paragraphs or lines differs between original and translation."),
            ("first_char", "First character type",
             "Detects when the type of the first character differs (letter, digit, quote, etc.)."),
            ("last_char", "Last character type",
             "Detects when the ending punctuation type differs (period, comma, exclamation, etc.)."),
            ("length", "Length ratio",
             "Detects when the translation is disproportionately longer or shorter than the original.\n"
             "Thresholds are configurable below. Texts ≤20 chars are always skipped."),
            ("quote_parity", "Quote parity",
             "Detects when the translation contains an odd number of double quotation marks."),
            ("untranslated", "Untranslated detection",
             "Detects when the translation is identical to the original text.\n"
             "Short texts, proper nouns, URLs and single tokens are ignored."),
            ("reserve_elements", "Reserve elements  <id_xx>",
             "Detects when reserve elements (images, <br> tags, structural placeholders) are\n"
             "missing or added in the translation. Only relevant for EPUB files."),
            ("nt_markers", "NT markers  <nt_xx/>",
             "Detects when non-translatable markers (padding spaces, empty anchors) are\n"
             "missing or added in the translation.\n"
             "Only relevant for EPUB Inline mode. Should always be enabled for EPUB."),
            ("inline_formatting", "Inline formatting  <p_xx>",
             "Detects when inline formatting tags (italic, bold, etc.) are missing, extra, or\n"
             "unpaired in the translation. Only relevant for EPUB Inline mode."),
        ]

        self.mismatch_check_checkboxes = {}
        checks_grid = QHBoxLayout()
        checks_grid.setSpacing(16)
        col1_layout = QVBoxLayout()
        col1_layout.setSpacing(6)
        col2_layout = QVBoxLayout()
        col2_layout.setSpacing(6)

        for i, (key, label, tooltip) in enumerate(mismatch_checks_info):
            cb = QCheckBox(label)
            cb.setToolTip(tooltip)
            cb.setChecked(saved_checks.get(key, True))
            cb.setStyleSheet(CHECKBOX_STYLE)
            self.mismatch_check_checkboxes[key] = cb
            (col1_layout if i < 5 else col2_layout).addWidget(cb)

        checks_grid.addLayout(col1_layout)
        sep_v = QFrame()
        sep_v.setFrameShape(QFrame.Shape.VLine)
        sep_v.setStyleSheet("color: #333; max-width: 1px;")
        checks_grid.addWidget(sep_v)
        checks_grid.addLayout(col2_layout)
        mismatch_outer_layout.addLayout(checks_grid)

        for cb in self.mismatch_check_checkboxes.values():
            cb.stateChanged.connect(self._on_mismatch_check_toggled)

        sep_h = QFrame()
        sep_h.setFrameShape(QFrame.Shape.HLine)
        sep_h.setStyleSheet("color: #2e2e2e; margin: 2px 0;")
        mismatch_outer_layout.addWidget(sep_h)

        thresholds_label = QLabel("Configurable thresholds  (take effect immediately, saved with Save Settings):")
        thresholds_label.setStyleSheet("color: #999; font-size: 11px; font-weight: bold;")
        mismatch_outer_layout.addWidget(thresholds_label)

        saved_thresholds = self.app_settings.get("mismatch_thresholds", {})
        thresholds_layout = QHBoxLayout()
        thresholds_layout.setSpacing(6)

        length_group = QFrame()
        length_group.setStyleSheet("""
            QFrame { background-color: #181818; border: 1px dashed #333; border-radius: 3px; }
        """)
        length_group_layout = QVBoxLayout(length_group)
        length_group_layout.setContentsMargins(8, 6, 8, 6)
        length_group_layout.setSpacing(4)
        length_group_title = QLabel("Length ratio  (longer ÷ shorter)")
        length_group_title.setStyleSheet("color: #777; font-size: 10px; font-weight: bold;")
        length_group_title.setToolTip(
            "Controls when 'Length ratio' mismatch is triggered.\n"
            "Three thresholds depending on text length:\n\n"
            "• Short  (≤100 chars): higher tolerance\n"
            "• Medium (≤500 chars): moderate tolerance\n"
            "• Long   (>500 chars): strictest\n\n"
            "Lower value = stricter check. Texts ≤20 chars are always ignored."
        )
        length_group_layout.addWidget(length_group_title)

        length_spinboxes_row = QHBoxLayout()
        length_spinboxes_row.setSpacing(10)

        def make_spinbox_with_label(label_text, attr_name, default_val, tooltip,
                                    min_val=1.1, max_val=5.0, step=0.1):
            col = QVBoxLayout()
            col.setSpacing(2)
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #666; font-size: 10px;")
            lbl.setToolTip(tooltip)
            col.addWidget(lbl)
            sb = QDoubleSpinBox()
            sb.setRange(min_val, max_val)
            sb.setSingleStep(step)
            sb.setDecimals(2)
            sb.setValue(saved_thresholds.get(attr_name, default_val))
            sb.setFixedWidth(74)
            sb.setStyleSheet(SPINBOX_STYLE)
            sb.setToolTip(tooltip)
            setattr(self, attr_name + "_spinbox", sb)
            col.addWidget(sb)
            hint = QLabel(f"def: {default_val:.2f}")
            hint.setStyleSheet("color: #444; font-size: 9px;")
            col.addWidget(hint)
            return col

        length_spinboxes_row.addLayout(make_spinbox_with_label(
            "Short (≤100)", "length_ratio_short", 1.6, "Texts up to 100 chars.\nDefault: 1.60"
        ))
        length_spinboxes_row.addLayout(make_spinbox_with_label(
            "Medium (≤500)", "length_ratio_medium", 1.4, "Texts 101–500 chars.\nDefault: 1.40"
        ))
        length_spinboxes_row.addLayout(make_spinbox_with_label(
            "Long (>500)", "length_ratio_long", 1.3, "Texts over 500 chars.\nDefault: 1.30"
        ))
        length_group_layout.addLayout(length_spinboxes_row)

        too_short_row = QHBoxLayout()
        too_short_row.setSpacing(10)
        too_short_row.addLayout(make_spinbox_with_label(
            "Too short (min ratio)", "length_ratio_too_short", 0.45,
            "Translation is flagged when shorter than this fraction of the original.\n"
            "e.g. 0.45 means: if translation < 45% of original length → mismatch.\n"
            "Only applied when original ≥ 8 chars.\nDefault: 0.45",
            min_val=0.05, max_val=0.95, step=0.05
        ))
        too_short_row.addStretch()
        length_group_layout.addLayout(too_short_row)
        thresholds_layout.addWidget(length_group)

        sep_thresh = QFrame()
        sep_thresh.setFrameShape(QFrame.Shape.VLine)
        sep_thresh.setStyleSheet("color: #2e2e2e;")
        thresholds_layout.addWidget(sep_thresh)

        untrans_group = QFrame()
        untrans_group.setStyleSheet("""
            QFrame { background-color: #181818; border: 1px dashed #333; border-radius: 3px; }
        """)
        untrans_group_layout = QVBoxLayout(untrans_group)
        untrans_group_layout.setContentsMargins(8, 6, 8, 6)
        untrans_group_layout.setSpacing(4)
        untrans_title = QLabel("Untranslated word ratio")
        untrans_title.setStyleSheet("color: #777; font-size: 10px; font-weight: bold;")
        untrans_title.setToolTip(
            "Minimum fraction of words starting with a lowercase letter\n"
            "required to trigger the 'untranslated' check.\n"
            "Higher value = less sensitive. Default: 0.30"
        )
        untrans_group_layout.addWidget(untrans_title)
        untrans_sb_row = QHBoxLayout()
        untrans_sb_row.setSpacing(6)
        self.untranslated_ratio_spinbox = QDoubleSpinBox()
        self.untranslated_ratio_spinbox.setRange(0.05, 0.95)
        self.untranslated_ratio_spinbox.setSingleStep(0.05)
        self.untranslated_ratio_spinbox.setDecimals(2)
        self.untranslated_ratio_spinbox.setValue(saved_thresholds.get("untranslated_ratio", 0.3))
        self.untranslated_ratio_spinbox.setFixedWidth(74)
        self.untranslated_ratio_spinbox.setStyleSheet(SPINBOX_STYLE)
        self.untranslated_ratio_spinbox.setToolTip("Default: 0.30")
        untrans_sb_row.addWidget(self.untranslated_ratio_spinbox)
        untrans_sb_row.addStretch()
        untrans_group_layout.addLayout(untrans_sb_row)
        untrans_group_layout.addWidget(QLabel("def: 0.30"))
        untrans_group.findChildren(QLabel)[-1].setStyleSheet("color: #444; font-size: 9px;")
        thresholds_layout.addWidget(untrans_group)

        sep_thresh2 = QFrame()
        sep_thresh2.setFrameShape(QFrame.Shape.VLine)
        sep_thresh2.setStyleSheet("color: #2e2e2e;")
        thresholds_layout.addWidget(sep_thresh2)

        pos_shift_group = QFrame()
        pos_shift_group.setStyleSheet("""
            QFrame { background-color: #181818; border: 1px dashed #2a3a4a; border-radius: 3px; }
        """)
        pos_shift_group_layout = QVBoxLayout(pos_shift_group)
        pos_shift_group_layout.setContentsMargins(8, 6, 8, 6)
        pos_shift_group_layout.setSpacing(6)

        pos_shift_title = QLabel("Tag position shift")
        pos_shift_title.setStyleSheet("color: #5a7a9a; font-size: 10px; font-weight: bold;")
        pos_shift_group_layout.addWidget(pos_shift_title)

        id_row = QHBoxLayout()
        id_row.setSpacing(6)

        id_sb_col = QVBoxLayout()
        id_sb_col.setSpacing(2)
        id_sb_label = QLabel("<id_xx>  <nt_xx/>")
        id_sb_label.setStyleSheet("color: #666; font-size: 10px;")
        id_sb_label.setToolTip(
            "Position shift threshold for reserve elements <id_xx> and <nt_xx/> markers.\n"
            "These are structural — their position should be stable.\n"
            "Default: 0.15"
        )
        id_sb_col.addWidget(id_sb_label)
        self.position_shift_threshold_spinbox = QDoubleSpinBox()
        self.position_shift_threshold_spinbox.setRange(0.01, 0.50)
        self.position_shift_threshold_spinbox.setSingleStep(0.01)
        self.position_shift_threshold_spinbox.setDecimals(2)
        self.position_shift_threshold_spinbox.setValue(
            saved_thresholds.get("position_shift_threshold", 0.15)
        )
        self.position_shift_threshold_spinbox.setFixedWidth(74)
        self.position_shift_threshold_spinbox.setStyleSheet(SPINBOX_STYLE)
        self.position_shift_threshold_spinbox.setToolTip(
            "Threshold for <id_xx> and <nt_xx/>.\nDefault: 0.15"
        )
        id_sb_col.addWidget(self.position_shift_threshold_spinbox)
        id_hint = QLabel("def: 0.15")
        id_hint.setStyleSheet("color: #444; font-size: 9px;")
        id_sb_col.addWidget(id_hint)
        id_row.addLayout(id_sb_col)
        id_row.addStretch()
        pos_shift_group_layout.addLayout(id_row)

        p_row = QHBoxLayout()
        p_row.setSpacing(6)

        p_sb_col = QVBoxLayout()
        p_sb_col.setSpacing(2)
        p_sb_label = QLabel("<p_xx>  (inline formatting)")
        p_sb_label.setStyleSheet("color: #666; font-size: 10px;")
        p_sb_label.setToolTip(
            "Position shift threshold for inline formatting tags <p_xx>.\n"
            "Higher default because translated words naturally have different lengths\n"
            "(e.g. 'Book' → 'Księga' shifts the relative position of the tag).\n"
            "Default: 0.30"
        )
        p_sb_col.addWidget(p_sb_label)
        self.inline_position_shift_threshold_spinbox = QDoubleSpinBox()
        self.inline_position_shift_threshold_spinbox.setRange(0.01, 0.99)
        self.inline_position_shift_threshold_spinbox.setSingleStep(0.01)
        self.inline_position_shift_threshold_spinbox.setDecimals(2)
        self.inline_position_shift_threshold_spinbox.setValue(
            saved_thresholds.get("inline_position_shift_threshold", 0.30)
        )
        self.inline_position_shift_threshold_spinbox.setFixedWidth(74)
        self.inline_position_shift_threshold_spinbox.setStyleSheet(SPINBOX_STYLE)
        self.inline_position_shift_threshold_spinbox.setToolTip(
            "Threshold for <p_xx> inline formatting tags.\nDefault: 0.30"
        )
        p_sb_col.addWidget(self.inline_position_shift_threshold_spinbox)
        p_hint = QLabel("def: 0.30")
        p_hint.setStyleSheet("color: #444; font-size: 9px;")
        p_sb_col.addWidget(p_hint)
        p_row.addLayout(p_sb_col)
        p_row.addStretch()
        pos_shift_group_layout.addLayout(p_row)

        thresholds_layout.addWidget(pos_shift_group)
        thresholds_layout.addStretch()
        mismatch_outer_layout.addLayout(thresholds_layout)

        self.length_ratio_short_spinbox.valueChanged.connect(self._on_mismatch_check_toggled)
        self.length_ratio_medium_spinbox.valueChanged.connect(self._on_mismatch_check_toggled)
        self.length_ratio_long_spinbox.valueChanged.connect(self._on_mismatch_check_toggled)
        self.length_ratio_too_short_spinbox.valueChanged.connect(self._on_mismatch_check_toggled)
        self.untranslated_ratio_spinbox.valueChanged.connect(self._on_mismatch_check_toggled)
        self.position_shift_threshold_spinbox.valueChanged.connect(self._on_mismatch_check_toggled)
        self.inline_position_shift_threshold_spinbox.valueChanged.connect(self._on_mismatch_check_toggled)

        outer_layout.addWidget(
            make_collapsible("🔍  Mismatch Detection Settings", mismatch_content,
                             expanded=False, color_scheme="amber")
        )

        epub_content = QWidget()
        epub_content.setStyleSheet("background: transparent;")
        epub_outer_layout = QVBoxLayout(epub_content)
        epub_outer_layout.setContentsMargins(0, 0, 0, 0)
        epub_outer_layout.setSpacing(10)

        self.inline_formatting_checkbox = QCheckBox("Use inline formatting system  (Inline mode)")
        self.inline_formatting_checkbox.setStyleSheet(CHECKBOX_STYLE)
        self.inline_formatting_checkbox.setToolTip(
            "<html>"
            "Inline: Full inline formatting with &lt;p_XX&gt;...&lt;/p_XX&gt;<br>"
            "(<i>italicized</i>, <b>bold text</b>), etc. placeholders<br><br>"
            "Legacy: Simple reserve elements &lt;id_XX&gt; (images) with position-based insertion<br>"
            "Uses an encoder-only model and is intended exclusively for legacy mode, operating with &lt;p_XX&gt;...&lt;/p_XX&gt; placeholders<br><br>"
            "⚠️ Changing the mode and clicking 'Save Settings' will require reloading the EPUB file."
            "</html>"
        )
        self.inline_formatting_checkbox.setChecked(self.app_settings.get('use_inline_formatting', True))
        self.inline_formatting_checkbox.setEnabled(True)
        self.inline_formatting_checkbox.stateChanged.connect(self._toggle_processing_mode)
        epub_outer_layout.addWidget(self.inline_formatting_checkbox)

        inline_tags_group = QGroupBox("Inline Formatting Tags to Skip  (Inline mode only)")
        inline_tags_group.setToolTip(
            "Select which inline formatting tags should be skipped during EPUB processing.\n"
            "Skipped tags will NOT be converted to placeholders – useful for broken EPUB files."
        )
        inline_tags_group.setStyleSheet("""
            QGroupBox {
                color: #777777; border: 1px dashed #3a4a2a; border-radius: 4px;
                margin-top: 4px; padding-top: 4px; background-color: #181818;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 8px; padding: 0 4px;
                font-size: 11px; color: #666666;
            }
        """)
        inline_tags_layout = QHBoxLayout(inline_tags_group)
        inline_tags_layout.setSpacing(12)

        self.skip_inline_checkboxes = {}
        skip_inline_tags = self.app_settings.get('skip_inline_tags', {})
        inline_tags_info = [
            ('span',   '<span>',   'Skip <span> tags'),
            ('i',      '<i>',      'Skip <i> (italic) tags'),
            ('b',      '<b>',      'Skip <b> (bold) tags'),
            ('em',     '<em>',     'Skip <em> (emphasis) tags'),
            ('strong', '<strong>', 'Skip <strong> (strong emphasis) tags'),
            ('u',      '<u>',      'Skip <u> (underline) tags'),
            ('sup',    '<sup>',    'Skip <sup> (superscript) tags'),
            ('sub',    '<sub>',    'Skip <sub> (subscript) tags'),
            ('small',  '<small>',  'Skip <small> (small text) tags'),
        ]
        for tag, label, tooltip in inline_tags_info:
            checkbox = QCheckBox(label)
            checkbox.setToolTip(tooltip)
            checkbox.setChecked(skip_inline_tags.get(tag, False))
            checkbox.setStyleSheet(CHECKBOX_SMALL_STYLE)
            self.skip_inline_checkboxes[tag] = checkbox
            inline_tags_layout.addWidget(checkbox)
        inline_tags_layout.addStretch()

        self.inline_tags_group = inline_tags_group
        self.inline_tags_group.setVisible(self.inline_formatting_checkbox.isChecked())
        self.inline_formatting_checkbox.stateChanged.connect(
            lambda: self.inline_tags_group.setVisible(self.inline_formatting_checkbox.isChecked())
        )
        epub_outer_layout.addWidget(inline_tags_group)

        self.alignment_section_widget = QWidget()
        alignment_section_layout = QVBoxLayout(self.alignment_section_widget)
        alignment_section_layout.setContentsMargins(0, 0, 0, 0)
        alignment_section_layout.setSpacing(4)

        sep_epub = QFrame()
        sep_epub.setFrameShape(QFrame.Shape.HLine)
        sep_epub.setStyleSheet("color: #2e2e2e; margin: 4px 0;")
        alignment_section_layout.addWidget(sep_epub)

        alignment_title_lbl = QLabel("🔡  Alignment Settings  (EPUB Legacy Mode only)")
        alignment_title_lbl.setStyleSheet(
            "color: #9977cc; font-size: 12px; font-weight: bold; margin-bottom: 2px;"
        )
        alignment_section_layout.addWidget(alignment_title_lbl)

        alignment_desc = QLabel(
            "Automatically transfers inline tags (&lt;i&gt;, &lt;b&gt;, &lt;span&gt;, etc.) "
            "from the original text to the translation using mBERT.<br>"
            "<span style='color: #666;'>Works only in LEGACY mode. "
            "The model selection prompt appears when saving EPUB.</span>"
        )
        alignment_desc.setWordWrap(True)
        alignment_desc.setStyleSheet("color: #8877aa; font-size: 11px;")
        alignment_section_layout.addWidget(alignment_desc)

        alignment_form = QFormLayout()
        alignment_form.setSpacing(8)
        alignment_form.setContentsMargins(0, 4, 0, 0)

        def align_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #888888; font-size: 11px;")
            return lbl

        self.alignment_device_combo = QComboBox()
        self.alignment_device_combo.addItems(["cpu", "cuda"])
        self.alignment_device_combo.setStyleSheet(COMBO_STYLE)
        self.alignment_device_combo.setToolTip(
            "cpu  - slower, always works\n"
            "cuda - requires NVIDIA GPU with PyTorch GPU installed (via installer)"
        )
        alignment_form.addRow(align_label("Device:"), self.alignment_device_combo)

        self.alignment_model_edit = QLineEdit()
        self.alignment_model_edit.setPlaceholderText("xlm-roberta-large")
        self.alignment_model_edit.setStyleSheet(INPUT_STYLE)
        self.alignment_model_edit.setToolTip(
            "Name of the model from HuggingFace Hub.\n"
            "These models are used ONLY for inline tag alignment in EPUB texts,\n"
            "i.e., to place <i>, <b>, <span> etc. in the correct position in the translated text.\n"
            "They do NOT perform translation or text generation.\n\n"
            "Lightweight / CPU-friendly:\n"
            "  bert-base-multilingual-cased        (~700 MB) – classic multilingual BERT\n"
            "  microsoft/mdeberta-v3-base          (~1.0 GB) – multilingual encoder, CPU OK\n\n"
            "Balanced / optional GPU:\n"
            "  xlm-roberta-base                     (~1.1 GB) – strong multilingual embeddings\n\n"
            "High-quality / GPU recommended:\n"
            "  xlm-roberta-large                    (~2.4 GB) – top-quality multilingual embeddings\n\n"
            "How it works:\n"
            "  The model computes semantic embeddings for each word in the original\n"
            "  and translated text, compares them to find word correspondences,\n"
            "  then inserts inline tags (<i>, <b>, etc.) in the correct positions.\n\n"
            "Each model is downloaded ONCE and stored in:\n"
            "  <app_directory>/models/<model_name>/\n\n"
            "Changing the name does NOT delete previous models – each model\n"
            "has its own subfolder. You can switch between downloaded models\n"
            "without re-downloading."
        )
        self.alignment_model_edit.textChanged.connect(self._refresh_alignment_status)
        alignment_form.addRow(align_label("HF Model:"), self.alignment_model_edit)
        alignment_section_layout.addLayout(alignment_form)

        models_dir = self._get_models_dir()
        self.alignment_path_label = QLabel(f"Models folder: {models_dir}")
        self.alignment_path_label.setWordWrap(True)
        self.alignment_path_label.setStyleSheet("color: #445544; font-size: 10px; margin-top: 2px;")
        alignment_section_layout.addWidget(self.alignment_path_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_download_model = QPushButton("  Download / Check Model (HuggingFace Hub)")
        btn_download_model.setStyleSheet("""
            QPushButton {
                background-color: #252030; color: #aaaacc;
                border: 1px solid #3a2e5a; border-radius: 3px;
                padding: 6px 14px; font-size: 11px;
            }
            QPushButton:hover {
                background-color: #302545; border-color: #6655aa; color: #ccbbee;
            }
            QPushButton:pressed { background-color: #1a1525; }
            QPushButton:disabled { color: #555; border-color: #2a2a2a; }
        """)
        btn_download_model.setToolTip(
            "Downloads the model from HuggingFace Hub and stores it in:\n"
            f"  {models_dir}/<model_name>/\n\n"
            "Requires internet connection and the 'transformers' package.\n"
            "After downloading, the model can be used offline."
        )
        btn_download_model.clicked.connect(self._on_download_alignment_model)
        btn_row.addWidget(btn_download_model)
        btn_row.addStretch()
        alignment_section_layout.addLayout(btn_row)

        self.alignment_status_label = QLabel("")
        self.alignment_status_label.setWordWrap(True)
        self.alignment_status_label.setStyleSheet("color: #556655; font-size: 10px;")
        alignment_section_layout.addWidget(self.alignment_status_label)

        epub_outer_layout.addWidget(self.alignment_section_widget)

        self.alignment_section_widget.setVisible(not self.inline_formatting_checkbox.isChecked())
        self.inline_formatting_checkbox.stateChanged.connect(
            lambda: self.alignment_section_widget.setVisible(not self.inline_formatting_checkbox.isChecked())
        )

        outer_layout.addWidget(
            make_collapsible("📖  EPUB Settings  (Inline / Legacy + Alignment)", epub_content,
                             expanded=False, color_scheme="purple")
        )

        _update_para_sub_visibility()
        self.inline_formatting_checkbox.stateChanged.connect(lambda _: _update_para_sub_visibility())

        def _sync_inline_dependent_mismatch_visibility():
            is_inline = self.inline_formatting_checkbox.isChecked()
            for key in ('nt_markers', 'inline_formatting'):
                cb = self.mismatch_check_checkboxes.get(key)
                if cb:
                    was_visible = cb.isVisible()
                    cb.setVisible(is_inline)
                    if not is_inline:
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)
                    elif not was_visible:
                        cb.blockSignals(True)
                        cb.setChecked(True)
                        cb.blockSignals(False)
            self._on_mismatch_check_toggled()

        _sync_inline_dependent_mismatch_visibility()
        self.inline_formatting_checkbox.stateChanged.connect(
            lambda _: _sync_inline_dependent_mismatch_visibility()
        )

        outer_layout.addStretch()

        btn_save_options = QPushButton("  Save Settings")
        btn_save_options.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e7a3a, stop:1 #125225);
                color: white; font-weight: bold; padding: 8px 24px;
                border-radius: 4px; border: 1px solid #2a9a4a; font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a9a4a, stop:1 #1a6a30);
                border-color: #4acc6a;
            }
            QPushButton:pressed { background: #0e4020; }
        """)
        btn_save_options.clicked.connect(self.save_app_settings)
        outer_layout.addWidget(btn_save_options, alignment=Qt.AlignmentFlag.AlignRight)

        current_llm = self.app_settings.get("llm_choice", "LM Studio")
        self.llm_choice_combo.setCurrentText(current_llm)
        self.server_url_edit.setText(self.app_settings.get("server_url", ""))
        self.ollama_model_edit.setText(self.app_settings.get("ollama_model_name", ""))
        self.openrouter_api_key_edit.setText(self.app_settings.get("openrouter_api_key", ""))
        self.openrouter_model_edit.setText(self.app_settings.get("openrouter_model_name", ""))
        self.deepl_free_api_key_edit.setText(self.app_settings.get("deepl_free_api_key", ""))
        self.deepl_pro_api_key_edit.setText(self.app_settings.get("deepl_pro_api_key", ""))

        alignment_cfg = self.app_settings.get("alignment_settings", {})
        self.alignment_device_combo.setCurrentText(alignment_cfg.get("device", "cpu"))
        model_name_cfg = alignment_cfg.get("model_name", "xlm-roberta-large")
        self.alignment_model_edit.setText(model_name_cfg)
        self._refresh_alignment_status(model_name_cfg)

        self.update_model_name_visibility(current_llm)
        return tab_root

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "Files (*.epub *.srt *.txt);;EPUB Files (*.epub);;SRT Files (*.srt);;Text Files (*.txt)"
        )
        if not path:
            return

        self.is_session_loaded = False

        if path.lower().endswith('.epub'):
            self.file_type = "epub"
        elif path.lower().endswith('.srt'):
            self.file_type = "srt"
        elif path.lower().endswith('.txt'):
            self.file_type = "txt"
        else:
            self.show_message("Unsupported Format", "Selected file has an unsupported format.", QMessageBox.Icon.Warning)
            return

        self.original_file_path = path

        settings_for_processor = self.app_settings.copy()

        if self.file_type == "epub":
            settings_for_processor['use_inline_formatting'] = self.inline_formatting_checkbox.isChecked()
            skip_inline_tags = {}
            for tag, checkbox in self.skip_inline_checkboxes.items():
                skip_inline_tags[tag] = checkbox.isChecked()
            settings_for_processor['skip_inline_tags'] = skip_inline_tags
        else:
            settings_for_processor['use_inline_formatting'] = False
            settings_for_processor['skip_inline_tags'] = {}

        try:
            self.file_processor = FileProcessorFactory.create_processor(
                self.file_type,
                settings_for_processor
            )

            result = self.file_processor.load(path)

            if isinstance(result, tuple):
                self.paragraphs = result[0]
            elif isinstance(result, list):
                self.paragraphs = result
            else:
                raise TypeError(f"Unexpected return type from file processor: {type(result)}")

            if not self.paragraphs:
                raise ValueError("No paragraphs loaded from file")

            if self.paragraphs and not isinstance(self.paragraphs[0], dict):
                raise TypeError(f"Invalid paragraph format: expected dict, got {type(self.paragraphs[0])}")

        except Exception as e:
            logging.error(f"Failed to load file: {e}")
            traceback.print_exc()
            self.show_message(
                "Load Error",
                f"Failed to load file:\n{e}",
                QMessageBox.Icon.Critical
            )
            self.original_file_path = None
            self.file_type = None
            self.paragraphs = []
            self.update_file_label()
            return

        if self.file_type == "epub":
            current_mode = self.inline_formatting_checkbox.isChecked()
            logger.info(f"EPUB loaded – processing mode: {'inline' if current_mode else 'legacy'}")
        else:
            logger.info(f"{self.file_type.upper()} loaded – inline formatting not applicable (EPUB only)")

        if hasattr(self, 'sentence_batch_checkbox'):
            is_epub = self.file_type == 'epub'
            self.sentence_batch_checkbox.setEnabled(is_epub)
            if not is_epub:
                self.sentence_batch_checkbox.setChecked(False)

        self._initialize_components()

        self.populate_list()
        self.update_file_label()

        self.update_llm_editor_content()

        self._update_status_after_file_load()

    def _get_dot_for_para(self, para):
        """
        Return the alignment dot emoji for a paragraph, or None if not applicable.
        Mirrors the logic in _update_item_visuals.
        """
        if not (self.file_type == 'epub' and para.get('processing_mode') == 'legacy'):
            return None
        if not para.get('is_translated'):
            return None
        original_html = para.get('original_html', '')
        has_inline = any(
            f'<{tag}' in original_html or f'<{tag}>' in original_html
            for tag in INLINE_TRANSFER_TAGS
        )
        if para.get('alignment_force_correct'):
            return '🟢'
        if para.get('alignment_force_bad'):
            return '🔴'
        if not has_inline:
            return '⚫'
        if para.get('aligned_translated_html'):
            return '🟢' if para.get('alignment_auto_wrap') else (
                '🟠' if para.get('alignment_corrections', 0) >= 2 else '🟡'
            )
        return '○'

    def _get_paragraphs_for_epub_save(self, output_path):
        """
        For Legacy-mode EPUBs that have aligned fragments, show the alignment
        filter dialog and return a (possibly filtered) copy of self.paragraphs.
        Returns None if the user cancelled.
        For non-legacy or un-aligned EPUBs returns self.paragraphs unchanged.
        """
        if not self._is_legacy_mode():
            return self.paragraphs

        counts = {}
        for para in self.paragraphs:
            dot = self._get_dot_for_para(para)
            if dot:
                counts[dot] = counts.get(dot, 0) + 1

        alignable = {'🟢', '🟡', '🟠'}
        if not any(d in counts for d in alignable):
            return self.paragraphs

        dlg = AlignmentSaveDialog(counts, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        include_dots = dlg.selected_dots()

        result = []
        for para in self.paragraphs:
            dot = self._get_dot_for_para(para)
            if dot == '🔴' or (dot in alignable and dot not in include_dots):
                para_copy = para.copy()
                para_copy.pop('aligned_translated_html', None)
                result.append(para_copy)
            else:
                result.append(para)
        return result

    def save_file(self):
        if not self.paragraphs:
            self.show_message("No Data", "Please open a file first.", QMessageBox.Icon.Warning)
            return

        if self.file_type == "epub":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save as New EPUB", "", "EPUB Files (*.epub)"
            )
        elif self.file_type == "srt":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save as New SRT", "", "SRT Files (*.srt)"
            )
        elif self.file_type == "txt":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save as New TXT", "", "Text Files (*.txt)"
            )
        else:
            return

        if not path:
            return

        try:
            if self.file_type == "epub":
                paragraphs_to_save = self._get_paragraphs_for_epub_save(path)
                if paragraphs_to_save is None:
                    return

                self.epub_creator = EPUBCreatorLxml(
                    self.file_processor.book,
                    paragraphs_to_save,
                    path,
                )

                self.epub_creator.finished.connect(self.on_file_saved)
                self.epub_creator.start()

            elif self.file_type == "srt":
                self.srt_creator = SRTCreator(self.paragraphs, path, self)
                self.srt_creator.finished.connect(self.on_file_saved)
                self.srt_creator.start()

            elif self.file_type == "txt":
                self.txt_creator = TXTCreator(self.paragraphs, path)
                self.txt_creator.finished.connect(self.on_file_saved)
                self.txt_creator.start()

            self.statusBar().showMessage("Saving file...", 0)

        except Exception as e:
            logging.error(f"Failed to save file: {e}")
            self.show_message(
                "Save Error",
                f"Failed to save file:\n{e}",
                QMessageBox.Icon.Critical,
            )

    def on_file_saved(self, path, is_error):
        if is_error:
            self.show_message("Save Error", f"Failed to save file:\n{path}", QMessageBox.Icon.Critical)
        else:
            self.show_message("Success", f"File saved:\n{path}")

    def unload_file(self):
        if not self.original_file_path:
            self.statusBar().showMessage("No file to unload", 2000)
            return

        msg = QMessageBox.question(
            self,
            "Unload File",
            "Are you sure you want to unload the current file?\n\nAll unsaved progress will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if msg != QMessageBox.StandardButton.Yes:
            return

        if hasattr(self, 'current_worker') and self.current_worker:
            self.cancel_translation()
            if self.current_worker.isRunning():
                self.current_worker.wait(2000)

        self.paragraphs = []
        self.original_file_path = None
        self.file_type = None
        self.file_processor = None
        self._preview_show_original_ids.clear()
        self.is_session_loaded = False
        self.last_checked_row = None

        self.list_widget.clear()
        self.original_text_view.clear()
        self.translated_text_view.textChanged.disconnect(self.update_translation_from_edit)
        self.translated_text_view.clear()
        self.translated_text_view.textChanged.connect(self.update_translation_from_edit)
        if hasattr(self, 'alignment_original_view'):
            self.alignment_original_view.clear()
        if hasattr(self, 'alignment_result_view'):
            self.alignment_result_view.textChanged.disconnect(self._on_alignment_result_edited)
            self.alignment_result_view.clear()
            self.alignment_result_view.textChanged.connect(self._on_alignment_result_edited)
        if hasattr(self, 'panel_tabs'):
            self.panel_tabs.setTabVisible(1, False)
            self.panel_tabs.setTabVisible(2, False)
            if self.panel_tabs.currentIndex() in (1, 2):
                self.panel_tabs.setCurrentIndex(0)
        if hasattr(self, 'epub_preview_view'):
            self.epub_preview_view.setHtml(
                '<html><body style="background:#1a1a1a;margin:0;padding:24px;">'
                '<p style="color:#666;font-family:sans-serif;font-size:13px;">'
                'Select an EPUB fragment from the list to see the visual preview.'
                '</p></body></html>'
            )
        self.progress_bar.setVisible(False)
        self.btn_translate.setVisible(True)
        self.btn_save_file.setVisible(True)
        if hasattr(self, 'btn_alignment'):
            self.btn_alignment.setVisible(False)

        self.update_file_label()

        if self.llm_editor_container.isVisible():
            self.update_llm_editor_content()

        self.statusBar().showMessage("File unloaded", 3000)
        logging.info("File unloaded - application state cleared")

    def save_session(self):
        if not self.paragraphs:
            self.show_message("No Data", "No progress to save.", QMessageBox.Icon.Warning)
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Session", "", "JSON Files (*.json)")
        if not path:
            return

        try:
            variant = self._get_current_variant()

            custom_prompts = {}
            if self.prompt_manager and variant:
                prompts = self.prompt_manager.load_prompts_for_variant(variant)
                custom_prompts = {
                    'ollama': prompts.get('ollama'),
                    'system': prompts.get('system'),
                    'assistant': prompts.get('assistant'),
                    'user': prompts.get('user')
                }

            processing_mode = 'inline' if self.app_settings.get('use_inline_formatting', True) else 'legacy'

            single_prompt_mode = self.single_prompt_checkbox.isChecked()

            json_payload_mode = self.json_payload_checkbox.isChecked()
            json_payload_template = ""
            json_response_field = ""

            if json_payload_mode:
                if hasattr(self, 'json_payload_edit'):
                    json_payload_template = self.json_payload_edit.toPlainText().strip()
                elif variant and variant in self.current_prompts_cache:
                    json_payload_template = self.current_prompts_cache[variant].get('json_payload', '')
                else:
                    json_payload_template = getattr(self, '_json_payload_content', '')
                json_response_field = self.json_response_field_edit.text().strip()

            SessionManager.save_session(
                path=path,
                paragraphs=self.paragraphs,
                original_file_path=self.original_file_path,
                file_type=self.file_type,
                app_settings=self.app_settings,
                context_before=self.context_before_spinbox.value(),
                context_after=self.context_after_spinbox.value(),
                temperature=self.temperature_spinbox.value(),
                custom_prompts=custom_prompts,
                single_prompt_mode=single_prompt_mode,
                processing_mode=processing_mode,
                prompt_variant=variant,
                json_payload_mode=json_payload_mode,
                json_payload_template=json_payload_template,
                json_response_field=json_response_field,
                sentence_batch_enabled=self.sentence_batch_checkbox.isChecked() if hasattr(self, 'sentence_batch_checkbox') else False,
                sentence_batch_size=self.batch_size_spinbox.value() if hasattr(self, 'batch_size_spinbox') else 5
            )

            self.show_message("Success", f"Session saved to file:\n{path}")

        except Exception as e:
            logging.error(f"Failed to save session: {e}")
            self.show_message("Session Save Error", f"Failed to save session:\n{e}", QMessageBox.Icon.Critical)

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Session", "", "JSON Files (*.json)")
        if not path:
            return

        try:
            session_data = SessionManager.load_session(path)

            original_path = session_data.get('original_file_path')
            if not original_path:
                self.show_message("Error", "No original file path in session.", QMessageBox.Icon.Critical)
                return

            confirmed_path, _ = QFileDialog.getOpenFileName(
                self, "Confirm original file location", original_path, "Files (*.epub *.srt *.txt)"
            )
            if not confirmed_path:
                self.show_message("Error", "No original file selected.", QMessageBox.Icon.Critical)
                return

            self.file_type = session_data['file_type']
            self.paragraphs = session_data['paragraphs']
            self.original_file_path = confirmed_path
            self.is_session_loaded = True

            metadata = session_data.get('metadata', {})
            processing_mode = metadata.get('processing_mode', 'inline')
            use_inline = (processing_mode == 'inline')

            skip_inline_tags = metadata.get('skip_inline_tags', {})

            self.app_settings['use_inline_formatting'] = use_inline
            self.app_settings['skip_inline_tags'] = skip_inline_tags

            self.context_before_spinbox.setValue(session_data.get('context_before', 3))
            self.context_after_spinbox.setValue(session_data.get('context_after', 2))
            self.temperature_spinbox.setValue(session_data.get('temperature', 0.8))

            if hasattr(self, 'sentence_batch_checkbox'):
                self.sentence_batch_checkbox.setChecked(session_data.get('sentence_batch_enabled', False))
            if hasattr(self, 'batch_size_spinbox'):
                self.batch_size_spinbox.setValue(session_data.get('sentence_batch_size', 5))

            self.inline_formatting_checkbox.blockSignals(True)
            self.inline_formatting_checkbox.setChecked(use_inline)
            self.inline_formatting_checkbox.blockSignals(False)

            for tag, checkbox in self.skip_inline_checkboxes.items():
                checkbox.blockSignals(True)
                checkbox.setChecked(skip_inline_tags.get(tag, False))
                checkbox.blockSignals(False)

            self.inline_tags_group.setVisible(use_inline)
            self.alignment_section_widget.setVisible(not use_inline)

            if self.file_type == "epub":
                logger.info(f"Session loaded – EPUB mode: {'inline' if use_inline else 'legacy'}, "
                            f"skip_inline_tags: {skip_inline_tags}")
            else:
                logger.info(f"Session loaded – {self.file_type.upper()} (inline not applicable)")

            variant = self._get_current_variant()
            custom_prompts = session_data.get('custom_prompts', {})

            if any(custom_prompts.values()):
                self.current_prompts_cache[variant] = {}

                if custom_prompts.get('ollama'):
                    self.current_prompts_cache[variant]['ollama'] = custom_prompts['ollama']
                else:
                    self.current_prompts_cache[variant]['ollama'] = self.prompt_manager.get_default_ollama_prompt(variant)

                if custom_prompts.get('system'):
                    self.current_prompts_cache[variant]['system'] = custom_prompts['system']
                else:
                    self.current_prompts_cache[variant]['system'] = self.prompt_manager.get_default_system_prompt(variant)

                if custom_prompts.get('assistant'):
                    self.current_prompts_cache[variant]['assistant'] = custom_prompts['assistant']
                else:
                    self.current_prompts_cache[variant]['assistant'] = self.prompt_manager.get_default_assistant_prompt(variant)

                if custom_prompts.get('user'):
                    self.current_prompts_cache[variant]['user'] = custom_prompts['user']
                else:
                    self.current_prompts_cache[variant]['user'] = self.prompt_manager.get_default_user_prompt(variant)

                logger.info(f"Loaded custom prompts from session into cache: {variant}")

            json_payload_mode = session_data.get('json_payload_mode', False)
            json_payload_template = session_data.get('json_payload_template', '')
            json_response_field = session_data.get('json_response_field', '')

            if json_payload_mode and json_payload_template:
                if variant not in self.current_prompts_cache:
                    self.current_prompts_cache[variant] = {}
                self.current_prompts_cache[variant]['json_payload'] = json_payload_template
                self._json_payload_content = json_payload_template

                self.app_settings['json_response_field'] = json_response_field

                self.json_payload_checkbox.blockSignals(True)
                self.json_payload_checkbox.setChecked(True)
                self.json_payload_checkbox.blockSignals(False)

                self.json_response_field_edit.setText(json_response_field)
                self.json_response_field_label.setVisible(True)
                self.json_response_field_edit.setVisible(True)

                logger.info(f"Loaded JSON payload mode from session: {variant}")
            else:
                self.json_payload_checkbox.blockSignals(True)
                self.json_payload_checkbox.setChecked(False)
                self.json_payload_checkbox.blockSignals(False)

                self.json_response_field_label.setVisible(False)
                self.json_response_field_edit.setVisible(False)

            self.file_processor = FileProcessorFactory.create_processor(
                self.file_type,
                self.app_settings
            )

            if self.file_type == "epub":
                try:
                    logger.info(f"Loading EPUB file to restore book object...")
                    file_paragraphs, loaded_book = self.file_processor.load(confirmed_path)

                    if loaded_book is None:
                        raise ValueError("Failed to load EPUB book object")

                    logger.info(f"✓ EPUB book object restored (needed for saving)")
                    self._remap_session_paragraph_ids(file_paragraphs)

                except Exception as e:
                    logger.error(f"Failed to load EPUB file: {e}")
                    traceback.print_exc()
                    self.show_message(
                        "Load Error",
                        f"Failed to load EPUB file:\n{e}\n\n"
                        f"Session paragraphs loaded, but you won't be able to save EPUB until you reload the original file.",
                        QMessageBox.Icon.Warning
                    )

            self._initialize_components()
            self.populate_list()
            self.update_file_label()
            self._update_status_after_file_load()

            if self.llm_editor_container.isVisible():
                self.update_llm_editor_content()

            mode_str = 'inline' if use_inline else 'legacy'
            self.show_message("Success", f"Session loaded successfully.\nMode: {mode_str}")

        except Exception as e:
            logging.error(f"Failed to load session: {e}")
            traceback.print_exc()
            self.show_message("Session Load Error", f"Failed to load session:\n{e}", QMessageBox.Icon.Critical)

    def _remap_session_paragraph_ids(self, file_paragraphs: list) -> None:
        def normalize(text: str) -> str:
            if not text:
                return ''
            text = unicodedata.normalize('NFKC', text)
            return ' '.join(text.split())

        if not file_paragraphs:
            logger.warning("[remap] file_paragraphs is empty – skipping ID remap")
            return

        fresh_map: dict = {}
        for fp in file_paragraphs:
            orig = normalize(fp.get('original_text', ''))
            if orig:
                fresh_map[orig] = fp

        logger.info(f"[remap] Built fresh_map with {len(fresh_map)} entries")

        remapped = 0
        not_found = 0

        for para in self.paragraphs:
            norm_orig = normalize(para.get('original_text', ''))
            if not norm_orig:
                continue

            fresh_para = fresh_map.get(norm_orig)

            if fresh_para is None:
                logger.warning(
                    f"[remap] No match for para id={para.get('id','?')}, "
                    f"text={norm_orig[:60]}..."
                )
                not_found += 1
                continue

            new_id = fresh_para.get('id', '')
            old_id = para.get('id', '')

            if new_id and new_id != old_id:
                para['id'] = new_id

                new_href = fresh_para.get('item_href', '')
                if new_href and new_href != para.get('item_href', ''):
                    para['item_href'] = new_href
                remapped += 1
                logger.debug(f"[remap] {old_id[:30]}... → {new_id[:30]}...")

        logger.info(
            f"[remap] ID remap complete: {remapped} remapped, "
            f"{not_found} not found (will use content fallback)"
        )

    def start_translation(self):
        selected_items = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                idx = item.data(Qt.ItemDataRole.UserRole)
                selected_items.append(idx)

        if not selected_items:
            self.show_message(
                "No Selection",
                "Select at least one fragment to translate.",
                QMessageBox.Icon.Warning
            )
            return

        self.translation_cancelled = False
        self.btn_translate.setVisible(False)
        self.btn_save_file.setVisible(False)
        if hasattr(self, "btn_alignment"):
            self.btn_alignment.setVisible(False)
        self.total_to_translate = len(selected_items)
        self.completed_translations = 0
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.translation_queue = selected_items.copy()
        self.current_translation_idx = None

        self._eta_session_start = time.time()
        self._eta_chars_done = 0
        self._eta_recent_times = []
        self._eta_chars_total = sum(
            len(self.paragraphs[i].get('original_text', ''))
            for i in selected_items
        )
        if hasattr(self, 'eta_label'):
            self.eta_label.setText("")
            self.eta_label.setVisible(True)

        try:
            llm_choice = self.app_settings.get('llm_choice', 'LM Studio')

            if llm_choice == "Ollama":
                llm_client = LLMClientFactory.create_client(
                    llm_choice="Ollama",
                    model_name=self.app_settings.get('ollama_model_name', 'llama3.2:3b')
                )
            elif llm_choice == "Openrouter":
                llm_client = LLMClientFactory.create_client(
                    llm_choice="Openrouter",
                    model_name=self.app_settings.get('openrouter_model_name', ''),
                    api_key=self.app_settings.get('openrouter_api_key', '')
                )
            else:
                custom_url = self.app_settings.get('server_url', '').strip()
                llm_client = LLMClientFactory.create_client(
                    llm_choice="LM Studio",
                    endpoint=custom_url if custom_url else None
                )

            variant = self._get_current_variant()
            prompts = self._get_current_prompts_from_cache(variant)

            logger.info(f"Using prompts from cache for variant: {variant}")

            is_json_payload = self.json_payload_checkbox.isChecked()
            batch_mode = hasattr(self, 'sentence_batch_checkbox') and self.sentence_batch_checkbox.isChecked()

            if is_json_payload:
                if batch_mode:
                    json_template = ''
                    if variant and variant in self.current_prompts_cache:
                        json_template = self.current_prompts_cache[variant].get('batch_json_payload', '')
                    if not json_template and hasattr(self, 'json_payload_edit'):
                        json_template = self.json_payload_edit.toPlainText().strip()
                elif hasattr(self, 'json_payload_edit'):
                    json_template = self.json_payload_edit.toPlainText().strip()
                else:
                    json_template = getattr(self, '_json_payload_content', None) or ''
                json_response_field = self.json_response_field_edit.text().strip()
                prompt_builder = PromptBuilder(
                    variant=variant,
                    json_payload_template=json_template,
                    json_response_field=json_response_field
                )
            elif llm_choice == "Ollama":
                prompt_builder = PromptBuilder(
                    variant=variant,
                    ollama_template=prompts['ollama'],
                    single_prompt_mode=self.single_prompt_checkbox.isChecked()
                )
            else:
                asst_template = prompts.get('batch_assistant', '') if batch_mode else prompts['assistant']
                prompt_builder = PromptBuilder(
                    variant=variant,
                    system_template=prompts['system'],
                    assistant_template=asst_template,
                    user_template=prompts['user'],
                    single_prompt_mode=self.single_prompt_checkbox.isChecked()
                )

            self.translation_orchestrator = TranslationOrchestrator(
                llm_client=llm_client,
                prompt_builder=prompt_builder,
                formatting_sync=self.formatting_sync,
                timeout_minutes=self.timeout_spinbox.value()
            )

            self.statusBar().showMessage("Translation started...", 0)
            self.translate_next_fragment()

        except Exception as e:
            logging.error(f"Failed to start translation: {e}")
            logging.error(traceback.format_exc())
            self.show_message("Translation Error", f"Failed to start translation:\n{e}", QMessageBox.Icon.Critical)
            self.finalize_translation()

    def translate_next_fragment(self):
        if self.translation_cancelled:
            logging.info("Translation cancelled by user - stopping queue")
            self.finalize_translation()
            return

        if hasattr(self, 'sentence_batch_checkbox') and self.sentence_batch_checkbox.isChecked() and self.file_type == 'epub':
            self._translate_next_batch()
            return

        if not self.translation_queue:
            self.finalize_translation()
            return

        idx = self.translation_queue.pop(0)
        self.current_translation_idx = idx

        self.current_fragment_index = idx
        self.translation_start_time = time.time()
        self.current_retry_attempt = 0
        self.current_max_attempts = 1
        self.translation_timer.start(1000)

        try:
            self.paragraphs[idx]['index'] = idx

            _restore_on = (
                hasattr(self, 'restore_paragraph_checkbox')
                and self.restore_paragraph_checkbox.isChecked()
            )
            if self.file_type == 'srt':
                _use_ps = False
            else:
                _use_ps = _restore_on

            self.paragraphs[idx]['use_ps_markers'] = _use_ps

            auto_fix_manager = None
            if self.auto_fix_checkbox.isChecked():
                auto_fix_manager = AutoFixManager(
                    max_attempts=self.auto_fix_spinbox.value(),
                    base_temperature=self.temperature_spinbox.value()
                )
                self.current_max_attempts = self.auto_fix_spinbox.value()

            context_before = self._get_context(idx, before=True, count=self.context_before_spinbox.value())
            context_after = self._get_context(idx, before=False, count=self.context_after_spinbox.value())

            self.current_worker = TranslationWorkerThread(
                orchestrator=self.translation_orchestrator,
                fragment=self.paragraphs[idx],
                context_before=context_before,
                context_after=context_after,
                temperature=self.temperature_spinbox.value(),
                auto_fix_manager=auto_fix_manager,
                mismatch_checker=self.mismatch_checker
            )

            self.current_worker.progress.connect(self.on_translation_progress)
            self.current_worker.retry_progress.connect(self.on_retry_progress)
            self.current_worker.finished.connect(self.on_translation_finished)
            self.current_worker.start()

        except Exception as e:
            logging.error(f"Failed to start translation worker: {e}")
            logging.error(traceback.format_exc())

            self.translation_timer.stop()
            if hasattr(self, '_cancellation_prefix'):
                delattr(self, '_cancellation_prefix')

            self.paragraphs[idx]['translated_text'] = f"Failed to start: {e}"
            self.paragraphs[idx]['is_translated'] = False

            self._update_progress()
            self._update_item_visuals(idx)
            self.translate_next_fragment()

    def _translate_next_batch(self):
        if not self.translation_queue:
            self.finalize_translation()
            return

        try:
            self._translate_next_batch_inner()
        except Exception as e:
            logging.error(f"[Batch] Unexpected error in _translate_next_batch: {e}")
            logging.error(traceback.format_exc())
            self.translate_next_fragment()

    def _translate_next_batch_inner(self):

        self._batch_mismatch_indices = []

        batch_size = self.batch_size_spinbox.value()
        first_idx = self.translation_queue[0]
        first_href = self.paragraphs[first_idx].get('item_href', '')

        batch_indices = []
        for i, idx in enumerate(self.translation_queue):
            if i >= batch_size:
                break
            para = self.paragraphs[idx]
            if i > 0 and para.get('item_href', '') != first_href:
                break
            if para.get('is_non_translatable', False):
                break
            batch_indices.append(idx)

        if not batch_indices:
            batch_indices = [self.translation_queue[0]]

        for idx in batch_indices:
            self.translation_queue.remove(idx)

        self._current_batch_indices = batch_indices

        fragments = [self.paragraphs[i] for i in batch_indices]

        context_before = self._get_context(batch_indices[0], before=True, count=self.context_before_spinbox.value())
        context_after = self._get_context(batch_indices[-1], before=False, count=self.context_after_spinbox.value())

        batcher = SentenceBatcher()

        variant = self._get_current_variant()
        batch_hint_template = ""
        if variant and variant in self.current_prompts_cache:
            batch_hint_template = self.current_prompts_cache[variant].get('batch_hint', '')
        if not batch_hint_template:
            from app_utils import PromptManager as _PM
            batch_hint_template = _PM.get_default_batch_hint_prompt()

        auto_fix_manager = None
        if self.auto_fix_checkbox.isChecked():
            auto_fix_manager = AutoFixManager(
                max_attempts=self.auto_fix_spinbox.value(),
                base_temperature=self.temperature_spinbox.value()
            )

        self.translation_start_time = time.time()
        self.translation_timer.start(1000)
        self.current_fragment_index = batch_indices[0]

        self.current_worker = TranslationBatchWorkerThread(
            orchestrator=self.translation_orchestrator,
            fragments=fragments,
            indices=batch_indices,
            context_before=context_before,
            context_after=context_after,
            temperature=self.temperature_spinbox.value(),
            auto_fix_manager=auto_fix_manager,
            mismatch_checker=self.mismatch_checker,
            batcher=batcher,
            batch_hint_template=batch_hint_template
        )

        self.current_worker.batch_progress.connect(self.on_batch_progress)
        self.current_worker.finished.connect(self.on_batch_finished)
        self.current_worker.start()

    def on_batch_progress(self, indices, translations, is_error):
        chars_translated = 0
        for idx, translation in zip(indices, translations):
            if is_error:
                self.paragraphs[idx]['translated_text'] = translation
                self.paragraphs[idx]['is_translated'] = False
            else:
                if self.formatting_sync:
                    synced = self.formatting_sync.sync_formatting(
                        original=self.paragraphs[idx]['original_text'],
                        translated=translation,
                        para=self.paragraphs[idx]
                    )
                else:
                    synced = translation

                is_now_translated = bool(synced.strip())
                self.paragraphs[idx]['translated_text'] = synced
                self.paragraphs[idx]['is_translated'] = is_now_translated

                self.paragraphs[idx].pop('aligned_translated_html', None)
                self.paragraphs[idx].pop('alignment_corrections', None)
                self.paragraphs[idx].pop('alignment_uncertain', None)

                if self.mismatch_checker:
                    has_mismatch, flags = self.mismatch_checker.check_mismatch(self.paragraphs[idx])
                    self.paragraphs[idx]['has_mismatch'] = has_mismatch
                    self.paragraphs[idx]['mismatch_flags'] = flags

                    if has_mismatch:
                        if not hasattr(self, '_batch_mismatch_indices'):
                            self._batch_mismatch_indices = []
                        self._batch_mismatch_indices.append(idx)
                        logging.info(f"[Batch] Fragment {idx} has mismatch — queued for auto-fix")

                chars_translated += len(self.paragraphs[idx].get('original_text', ''))

            self._update_item_visuals(idx)

        if not is_error and chars_translated > 0:
            self._update_eta(chars_translated)

        current_item = self.list_widget.currentItem()
        if current_item:
            current_idx = current_item.data(Qt.ItemDataRole.UserRole)
            if current_idx in indices:
                self.translated_text_view.textChanged.disconnect(self.update_translation_from_edit)
                self.translated_text_view.setPlainText(
                    self._format_text_for_display(self.paragraphs[current_idx]['translated_text'])
                )
                self.translated_text_view.textChanged.connect(self.update_translation_from_edit)

        if not is_error and self._reader_window is not None:
            book = getattr(self.file_processor, 'book', None)
            if book is not None:
                sel_para = {}
                current_item = self.list_widget.currentItem()
                if current_item:
                    ci = current_item.data(Qt.ItemDataRole.UserRole)
                    if ci is not None and ci < len(self.paragraphs):
                        sel_para = self.paragraphs[ci]
                self._refresh_reader_if_open(book, sel_para)

    def on_batch_finished(self):
        self.translation_timer.stop()

        if hasattr(self, '_cancellation_prefix'):
            delattr(self, '_cancellation_prefix')

        batch_indices = getattr(self, '_current_batch_indices', [self.current_fragment_index] if self.current_fragment_index is not None else [])

        for bidx in batch_indices:
            row = self.para_to_row_map.get(bidx)
            if row is not None and self.paragraphs[bidx].get('is_translated', False):
                item = self.list_widget.item(row)
                if item:
                    item.setCheckState(Qt.CheckState.Unchecked)

        self.update_file_label()

        if self.current_worker:
            try:
                self.current_worker.batch_progress.disconnect()
                self.current_worker.finished.disconnect()
            except Exception:
                pass
            if self.current_worker.isRunning():
                self.current_worker.wait(1000)
            self.current_worker.deleteLater()
            self.current_worker = None

        if self.translation_cancelled:
            logging.info("Translation cancelled - stopping after current batch")
            for _ in batch_indices:
                self._update_progress()
            self.finalize_translation()
            return

        for _ in batch_indices:
            self._update_progress()

        if (
            self.auto_fix_checkbox.isChecked()
            and getattr(self, '_batch_mismatch_indices', [])
        ):
            try:
                self._fix_batch_mismatches()
            except Exception as e:
                logging.error(f"[Batch Mismatch Fix] Setup failed: {e} — resuming batch normally")
                logging.error(traceback.format_exc())
                self._batch_mismatch_indices = []
                self.translate_next_fragment()
        else:
            self._batch_mismatch_indices = []
            self.translate_next_fragment()

    def _fix_batch_mismatches(self):
        self._mismatch_fix_queue = list(self._batch_mismatch_indices)
        self._batch_mismatch_indices = []
        logging.info(
            f"[Batch Mismatch Fix] Starting single-fragment auto-fix for "
            f"{len(self._mismatch_fix_queue)} fragment(s): {self._mismatch_fix_queue}"
        )
        self._fix_next_mismatch()

    def _fix_next_mismatch(self):
        if self.translation_cancelled:
            self.finalize_translation()
            return

        if not getattr(self, '_mismatch_fix_queue', []):
            logging.info("[Batch Mismatch Fix] All mismatches resolved — resuming batch mode")
            self.translate_next_fragment()
            return

        try:
            idx = self._mismatch_fix_queue.pop(0)
            self.current_translation_idx = idx
            self.current_fragment_index = idx

            logging.info(f"[Batch Mismatch Fix] Re-translating fragment {idx} with full auto-fix")

            self.paragraphs[idx]['index'] = idx

            _restore_on = (
                hasattr(self, 'restore_paragraph_checkbox')
                and self.restore_paragraph_checkbox.isChecked()
            )
            if self.file_type == 'srt':
                _use_ps = False
            else:
                _use_ps = _restore_on
            self.paragraphs[idx]['use_ps_markers'] = _use_ps

            auto_fix_manager = AutoFixManager(
                max_attempts=self.auto_fix_spinbox.value(),
                base_temperature=self.temperature_spinbox.value()
            )

            context_before = self._get_context(idx, before=True, count=self.context_before_spinbox.value())
            context_after = self._get_context(idx, before=False, count=self.context_after_spinbox.value())

            self.translation_start_time = time.time()
            self.current_retry_attempt = 0
            self.current_max_attempts = self.auto_fix_spinbox.value()
            self.translation_timer.start(1000)

            self.current_worker = TranslationWorkerThread(
                orchestrator=self.translation_orchestrator,
                fragment=self.paragraphs[idx],
                context_before=context_before,
                context_after=context_after,
                temperature=self.temperature_spinbox.value(),
                auto_fix_manager=auto_fix_manager,
                mismatch_checker=self.mismatch_checker
            )

            self.current_worker.progress.connect(self.on_translation_progress)
            self.current_worker.retry_progress.connect(self.on_retry_progress)
            self.current_worker.finished.connect(self.on_mismatch_fix_finished)
            self.current_worker.start()

        except Exception as e:
            logging.error(f"[Batch Mismatch Fix] Failed to start worker for fragment: {e}")
            logging.error(traceback.format_exc())
            # Don't get stuck — continue with next mismatch or resume batch
            self._fix_next_mismatch()

    def on_mismatch_fix_finished(self):
        self.translation_timer.stop()

        if hasattr(self, '_cancellation_prefix'):
            delattr(self, '_cancellation_prefix')
        if hasattr(self, 'current_retry_attempt'):
            delattr(self, 'current_retry_attempt')
        if hasattr(self, 'current_max_attempts'):
            delattr(self, 'current_max_attempts')

        idx = self.current_translation_idx

        row = self.para_to_row_map.get(idx)
        if row is not None:
            item = self.list_widget.item(row)
            if item and self.paragraphs[idx].get('is_translated', False):
                item.setCheckState(Qt.CheckState.Unchecked)

        self.update_file_label()

        if self.current_worker:
            try:
                self.current_worker.progress.disconnect()
                self.current_worker.retry_progress.disconnect()
                self.current_worker.finished.disconnect()
            except Exception:
                pass
            if self.current_worker.isRunning():
                self.current_worker.wait(1000)
            self.current_worker.deleteLater()
            self.current_worker = None

        if self.translation_cancelled:
            logging.info("[Batch Mismatch Fix] Cancelled — stopping")
            self.finalize_translation()
            return

        # Note: _update_progress was already called in on_batch_finished for this
        # fragment, so we do NOT call it again here to avoid double-counting.
        self._fix_next_mismatch()

    def _update_eta(self, chars_done: int):
        self._eta_chars_done += chars_done

        if self._eta_session_start is None:
            return

        elapsed_total = time.time() - self._eta_session_start
        if elapsed_total < 1.0 or self._eta_chars_done == 0:
            return

        chars_per_sec = self._eta_chars_done / elapsed_total
        chars_left = max(0, self._eta_chars_total - self._eta_chars_done)

        if chars_per_sec <= 0:
            return

        eta_seconds = int(chars_left / chars_per_sec)

        if eta_seconds < 60:
            eta_str = f"~{eta_seconds}s"
        elif eta_seconds < 3600:
            m, s = divmod(eta_seconds, 60)
            eta_str = f"~{m}m {s:02d}s"
        else:
            h, rem = divmod(eta_seconds, 3600)
            m = rem // 60
            eta_str = f"~{h}h {m}m"

        percent = int(self._eta_chars_done / max(self._eta_chars_total, 1) * 100)
        if hasattr(self, 'eta_label'):
            self.eta_label.setText(f"ETA: {eta_str}  ({percent}% chars)")

    def on_translation_progress(self, idx, translation, is_error):
        if is_error:
            self.paragraphs[idx]['translated_text'] = translation
            self.paragraphs[idx]['is_translated'] = False
        else:
            if self.file_type == "srt":
                lines = self.split_translated_text_into_lines(translation, self.paragraphs[idx])
                translation_with_newlines = '\n'.join(lines)
            else:
                translation_with_newlines = translation

            if self.formatting_sync:
                synced_translation = self.formatting_sync.sync_formatting(
                    original=self.paragraphs[idx]['original_text'],
                    translated=translation_with_newlines,
                    para=self.paragraphs[idx]
                )
            else:
                synced_translation = translation_with_newlines

            is_now_translated = bool(synced_translation.strip())

            self.paragraphs[idx]['translated_text'] = synced_translation
            self.paragraphs[idx]['is_translated'] = is_now_translated

            self.paragraphs[idx].pop('aligned_translated_html', None)
            self.paragraphs[idx].pop('alignment_corrections', None)
            self.paragraphs[idx].pop('alignment_uncertain', None)

            if self.mismatch_checker:
                has_mismatch, flags = self.mismatch_checker.check_mismatch(self.paragraphs[idx])
                self.paragraphs[idx]['has_mismatch'] = has_mismatch
                self.paragraphs[idx]['mismatch_flags'] = flags

        self._update_item_visuals(idx)

        if not is_error:
            self._update_eta(len(self.paragraphs[idx].get('original_text', '')))

        current_item = self.list_widget.currentItem()
        if current_item:
            current_idx = current_item.data(Qt.ItemDataRole.UserRole)
            if current_idx == idx:
                self.translated_text_view.textChanged.disconnect(self.update_translation_from_edit)
                self.translated_text_view.setPlainText(
                    self._format_text_for_display(self.paragraphs[idx]['translated_text'])
                )
                self.translated_text_view.textChanged.connect(self.update_translation_from_edit)

        if (
            hasattr(self, 'panel_tabs')
            and self.panel_tabs.currentIndex() == 2
            and self.paragraphs[idx].get('is_translated')
            and not self.paragraphs[idx].get('has_mismatch', False)
        ):
            current_item = self.list_widget.currentItem()
            if current_item:
                current_idx = current_item.data(Qt.ItemDataRole.UserRole)
                if (
                    current_idx is not None
                    and current_idx < len(self.paragraphs)
                    and self.paragraphs[current_idx].get('item_href')
                        == self.paragraphs[idx].get('item_href')
                ):
                    self.refresh_preview()

    def on_translation_finished(self):
        self.translation_timer.stop()

        if hasattr(self, '_cancellation_prefix'):
            delattr(self, '_cancellation_prefix')

        if hasattr(self, 'current_retry_attempt'):
            delattr(self, 'current_retry_attempt')
        if hasattr(self, 'current_max_attempts'):
            delattr(self, 'current_max_attempts')

        if self.current_translation_idx is None:
            return

        idx = self.current_translation_idx

        row = self.para_to_row_map.get(idx)
        if row is not None:
            item = self.list_widget.item(row)
            if item and self.paragraphs[idx].get('is_translated', False):
                item.setCheckState(Qt.CheckState.Unchecked)

        self.update_file_label()

        if self.current_worker:
            try:
                self.current_worker.progress.disconnect()
                self.current_worker.retry_progress.disconnect()
                self.current_worker.finished.disconnect()
            except:
                pass

            if self.current_worker.isRunning():
                self.current_worker.wait(1000)

            self.current_worker.deleteLater()
            self.current_worker = None

        if self.translation_cancelled:
            logging.info("Translation cancelled - stopping after current fragment")
            self._update_progress()
            self.finalize_translation()
            return

        self._update_progress()

        self.translate_next_fragment()

    def cancel_translation(self):
        self.translation_cancelled = True

        if hasattr(self, 'current_fragment_index') and self.current_fragment_index is not None:
            self._cancellation_prefix = "⚠️ CANCELLING - "
            logging.info("User requested translation cancellation - finishing current fragment...")
        else:
            self.statusBar().showMessage("Cancelling translation...", 0)

        if self.translation_orchestrator:
            self.translation_orchestrator.cancel()

        if hasattr(self, 'translation_queue'):
            remaining = len(self.translation_queue)
            self.translation_queue.clear()
            logging.info(f"Cleared translation queue ({remaining} fragments skipped)")

        logging.info("Translation cancel requested - waiting for current fragment to finish")

        self._hard_cancel_mode = True
        self.btn_cancel.setText("⛔ Hard Cancel")
        self.btn_cancel.setToolTip("Click: immediately stop LLM generation.")
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #6a0000; color: #ff4444;
                border: 2px solid #cc0000; border-radius: 4px;
                padding: 5px 12px; font-size: 11px; font-weight: bold;
            }
            QPushButton:hover { background-color: #990000; color: white; }
            QPushButton:pressed { background-color: #440000; }
        """)

    def _on_cancel_clicked(self):
        if hasattr(self, '_alignment_worker') and self._alignment_worker and self._alignment_worker.isRunning():
            if self._hard_cancel_mode:
                self._alignment_worker.hard_cancel()
                self.finalize_alignment()
            else:
                self._alignment_worker.request_cancel()
                self._alignment_cancelled = True
                self._hard_cancel_mode = True
                self.btn_cancel.setText("⛔ Hard Cancel")
                self.btn_cancel.setToolTip("Click: immediately stop alignment.")
                self.btn_cancel.setStyleSheet("""
                    QPushButton {
                        background-color: #6a0000; color: #ff4444;
                        border: 2px solid #cc0000; border-radius: 4px;
                        padding: 5px 12px; font-size: 11px; font-weight: bold;
                    }
                    QPushButton:hover { background-color: #990000; color: white; }
                    QPushButton:pressed { background-color: #440000; }
                """)
            return
        if self._hard_cancel_mode:
            self._hard_cancel_translation()
        else:
            self.cancel_translation()

    def _hard_cancel_translation(self):
        logging.info("HARD CANCEL requested by user")
        self.translation_cancelled = True

        if hasattr(self, '_cancellation_prefix'):
            delattr(self, '_cancellation_prefix')

        if hasattr(self, 'translation_queue'):
            self.translation_queue.clear()

        self.translation_timer.stop()
        self.current_fragment_index = None
        self.translation_start_time = None

        if self.translation_orchestrator:
            self.translation_orchestrator.hard_cancel()

        if self.current_worker:
            try:
                self.current_worker.progress.disconnect()
            except Exception:
                pass
            try:
                self.current_worker.retry_progress.disconnect()
            except Exception:
                pass
            try:
                self.current_worker.finished.disconnect()
            except Exception:
                pass
            try:
                self.current_worker.batch_progress.disconnect()
            except Exception:
                pass

            if isinstance(self.current_worker, (TranslationWorkerThread, TranslationBatchWorkerThread)):
                self.current_worker.hard_cancel()
            elif self.current_worker.isRunning():
                self.current_worker.terminate()
                self.current_worker.wait(2000)

            self.current_worker.deleteLater()
            self.current_worker = None

        logging.info("Hard cancel complete")
        self.finalize_translation()
    
    def _reset_cancel_button(self):
        if not hasattr(self, 'btn_cancel'):
            return
        self._hard_cancel_mode = False
        self.btn_cancel.setText("✖ Cancel")
        self.btn_cancel.setToolTip("Click: finish the current section and stop")
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #4a1a1a; color: #ff9999;
                border: 1px solid #772222; border-radius: 4px;
                padding: 5px 12px; font-size: 11px; font-weight: bold;
            }
            QPushButton:hover { background-color: #6a2222; color: white; }
        """)

    def finalize_translation(self):
        self.translation_timer.stop()
        self.current_fragment_index = None
        self.translation_start_time = None
        self.progress_bar.setVisible(False)

        if hasattr(self, 'eta_label'):
            self.eta_label.setVisible(False)
        self._eta_session_start = None
        self._eta_chars_done = 0
        self._eta_recent_times = []

        if self.translation_cancelled:
            self.statusBar().showMessage("⛔ Translation cancelled.", 5000)
            logging.info("Translation process cancelled by user")
        else:
            self.statusBar().showMessage("Translation completed.", 5000)
            logging.info("Translation process completed")

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            self._update_item_visuals(item.data(Qt.ItemDataRole.UserRole))

        self.update_file_label()

        self.translation_queue = []
        self.current_translation_idx = None
        self.translation_orchestrator = None

        self.btn_translate.setVisible(True)
        self.btn_save_file.setVisible(True)
        self._update_alignment_button_visibility()

        self._reset_cancel_button()



    def _is_legacy_mode(self) -> bool:
        """Returns True if loaded file is EPUB in legacy processing mode."""
        return (
            self.file_type == 'epub'
            and bool(self.paragraphs)
            and self.paragraphs[0].get('processing_mode') == 'legacy'
        )

    def _update_alignment_button_visibility(self):
        """Show/hide btn_alignment and tabs based on file type and mode."""
        is_legacy = self._is_legacy_mode()
        is_epub   = self.file_type == 'epub'
        if hasattr(self, 'btn_alignment'):
            self.btn_alignment.setVisible(is_legacy)
        if hasattr(self, 'panel_tabs'):
            self.panel_tabs.setTabVisible(1, is_legacy)
            self.panel_tabs.setTabVisible(2, is_epub)
        if hasattr(self, '_alignment_filter_row'):
            on_alignment_tab = (
                hasattr(self, 'panel_tabs')
                and self.panel_tabs.currentIndex() == 1
            )
            self._alignment_filter_row.setVisible(is_legacy and on_alignment_tab)

    def _update_alignment_tab(self, para_index):
        """Update the Alignment tab fields for the given paragraph index."""
        if not hasattr(self, 'alignment_original_view'):
            return

        para = self.paragraphs[para_index] if para_index < len(self.paragraphs) else None
        if para is None:
            self.alignment_original_view.setPlainText('')
            self.alignment_result_view.textChanged.disconnect(self._on_alignment_result_edited)
            self.alignment_result_view.setPlainText('')
            self.alignment_result_view.textChanged.connect(self._on_alignment_result_edited)
            return

        original_html = para.get('original_html', '')
        self.alignment_original_view.setPlainText(_html_strip_outer_tag(original_html))

        self.alignment_result_view.textChanged.disconnect(self._on_alignment_result_edited)
        aligned = para.get('aligned_translated_html', '')
        display_aligned = _html_strip_outer_tag(aligned) if aligned else ''
        self.alignment_result_view.setPlainText(display_aligned)
        self.alignment_result_view.textChanged.connect(self._on_alignment_result_edited)

    def _on_alignment_result_edited(self):
        """Save changes from alignment result editor back to the paragraph."""
        current_item = self.list_widget.currentItem()
        if not current_item:
            return
        idx = current_item.data(Qt.ItemDataRole.UserRole)
        if idx is None or idx >= len(self.paragraphs):
            return
        edited_inner = self.alignment_result_view.toPlainText()
        if edited_inner.strip():
            original_html = self.paragraphs[idx].get('original_html', '')
            full_html = _html_restore_outer_tag(edited_inner, original_html)
            self.paragraphs[idx]['aligned_translated_html'] = full_html
        else:
            self.paragraphs[idx].pop('aligned_translated_html', None)
        self._update_item_visuals(idx)

    def start_alignment(self):
        """Start alignment on selected, translated, non-mismatch paragraphs with inline tags."""
        models_dir = self._get_models_dir()
        alignment_model = self.alignment_model_edit.text().strip() or 'xlm-roberta-large'

        if not is_model_downloaded(alignment_model, models_dir):
            self.show_message(
                "Model Not Downloaded",
                f"The alignment model '{alignment_model}' is not available locally.\n\n"
                "Please download it first in the Options tab → EPUB Settings → "
                "Alignment Settings → Download / Check Model.",
                QMessageBox.Icon.Warning
            )
            return

        eligible = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            idx = item.data(Qt.ItemDataRole.UserRole)
            para = self.paragraphs[idx]
            if not para.get('is_translated'):
                continue
            if para.get('has_mismatch'):
                continue
            original_html = para.get('original_html', '')
            has_inline = any(
                f'<{tag}' in original_html for tag in INLINE_TRANSFER_TAGS
            )
            if not has_inline:
                continue
            eligible.append(para)

        if not eligible:
            self.show_message(
                "Nothing to Align",
                "No eligible fragments found.\n\n"
                "Fragments must be:\n"
                "• Checked (☑) in the list\n"
                "• Translated\n"
                "• Without mismatch\n"
                "• Original HTML must contain inline tags (<i>, <b>, <span>, etc.)",
                QMessageBox.Icon.Information
            )
            return

        device = self.alignment_device_combo.currentText()

        self._alignment_cancelled = False
        self._alignment_total = len(eligible)
        self._alignment_completed = 0

        self.btn_translate.setVisible(False)
        self.btn_save_file.setVisible(False)
        self.btn_alignment.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self._alignment_worker = AlignmentWorkerThread(
            paragraphs=eligible,
            model_name=alignment_model,
            device=device,
            models_dir=models_dir,
        )
        self._alignment_worker.progress.connect(self._on_alignment_progress)
        self._alignment_worker.finished.connect(self._on_alignment_finished)
        self._alignment_worker.start()

        self.statusBar().showMessage(
            f"Aligning fragment 0 / {self._alignment_total}...", 0
        )

    def _on_alignment_progress(self, completed, total):
        self._alignment_completed = completed
        self.statusBar().showMessage(
            f"Aligning fragment {completed} / {total}...", 0
        )
        percent = int(completed / max(total, 1) * 100)
        self.progress_bar.setValue(percent)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            self._update_item_visuals(item.data(Qt.ItemDataRole.UserRole))
        current_item = self.list_widget.currentItem()
        if current_item:
            idx = current_item.data(Qt.ItemDataRole.UserRole)
            self._update_alignment_tab(idx)

    def _on_alignment_finished(self):
        if self._alignment_worker:
            try:
                self._alignment_worker.progress.disconnect()
                self._alignment_worker.finished.disconnect()
            except Exception:
                pass
            if self._alignment_worker.isRunning():
                self._alignment_worker.wait(1000)
            self._alignment_worker.deleteLater()
            self._alignment_worker = None
        self.finalize_alignment()

    def finalize_alignment(self):
        self.progress_bar.setVisible(False)

        if getattr(self, '_alignment_cancelled', False):
            self.statusBar().showMessage("⛔ Alignment cancelled.", 5000)
        else:
            completed = getattr(self, '_alignment_completed', 0)
            total = getattr(self, '_alignment_total', 0)
            self.statusBar().showMessage(
                f"✓ Alignment complete: {completed}/{total} fragments processed.", 5000
            )

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            self._update_item_visuals(item.data(Qt.ItemDataRole.UserRole))

        current_item = self.list_widget.currentItem()
        if current_item:
            idx = current_item.data(Qt.ItemDataRole.UserRole)
            self._update_alignment_tab(idx)
            self._update_translation_field_state(self.paragraphs[idx])

        self.btn_translate.setVisible(True)
        self.btn_save_file.setVisible(True)
        self._update_alignment_button_visibility()
        self._reset_cancel_button()


    def _on_panel_tab_changed(self, index: int):
        """Handle switching between Translation (0), Alignment (1) and Preview (2) tabs."""
        is_alignment_tab = (index == 1)
        is_preview_tab   = (index == 2)
        hide_translation_controls = is_alignment_tab or is_preview_tab

        for widget in (
            self.btn_copy_original,
            self._qt_sep1,
            self.quick_translate_service_combo,
            self._qt_lbl_from,
            self.source_lang_combo,
            self._qt_lbl_arrow,
            self._qt_lbl_to,
            self.target_lang_combo,
            self.btn_quick_translate,
        ):
            if hasattr(self, widget.__class__.__name__) or widget is not None:
                widget.setVisible(not hide_translation_controls)

        if hasattr(self, '_alignment_filter_row'):
            show_filter = is_alignment_tab and self._is_legacy_mode()
            self._alignment_filter_row.setVisible(show_filter)
            if not is_alignment_tab:
                self._filter_alignment_dot(None)

        if is_alignment_tab:
            try:
                self.btn_mark_correct.clicked.disconnect()
            except Exception:
                pass
            try:
                self.btn_unmark_correct.clicked.disconnect()
            except Exception:
                pass
            self.btn_mark_correct.setText("🟢  Confirm Alignment")
            self.btn_mark_correct.setToolTip(
                "Mark alignment as manually confirmed (🟢 green circle)"
            )
            self.btn_unmark_correct.setText("🔴  Flag Alignment")
            self.btn_unmark_correct.setToolTip(
                "Flag alignment as incorrect / needs review (🔴 red)"
            )
            self.btn_mark_correct.clicked.connect(self._mark_alignment_confirmed)
            self.btn_unmark_correct.clicked.connect(self._flag_alignment_bad)
        else:
            try:
                self.btn_mark_correct.clicked.disconnect()
            except Exception:
                pass
            try:
                self.btn_unmark_correct.clicked.disconnect()
            except Exception:
                pass
            self.btn_mark_correct.setText("✓  Mark Correct")
            self.btn_mark_correct.setToolTip(
                "Mark selected fragment as correct (ignore mismatch)"
            )
            self.btn_unmark_correct.setText("✗  Unmark")
            self.btn_unmark_correct.setToolTip(
                "Remove 'correct' flag and recheck mismatch"
            )
            self.btn_mark_correct.clicked.connect(self.mark_fragment_as_correct)
            self.btn_unmark_correct.clicked.connect(self.unmark_fragment_as_correct)

        if is_preview_tab:
            self.refresh_preview()

        for _widget in (
            getattr(self, 'controls_bar', None),
            getattr(self, 'llm_options_widget', None),
            getattr(self, 'action_buttons_widget', None),
        ):
            if _widget is not None:
                _widget.setVisible(not is_preview_tab)

    def refresh_preview(self):
        """
        Regenerate the Preview tab content for the currently selected fragment.
        Called when the user switches to the Preview tab or selects a new fragment
        while already on the Preview tab.
        """
        if not hasattr(self, 'epub_preview_view'):
            return

        current_item = self.list_widget.currentItem()
        if not current_item:
            self.epub_preview_view.setHtml(
                '<html><body style="background:#1a1a1a;margin:0;padding:24px;">'
                '<p style="color:#666;font-family:sans-serif;font-size:13px;">'
                'No fragment selected.</p></body></html>'
            )
            return

        idx = current_item.data(Qt.ItemDataRole.UserRole)
        if idx is None or not self.paragraphs:
            return

        book = getattr(self.file_processor, 'book', None)
        if book is None:
            self.epub_preview_view.setHtml(
                '<html><body style="background:#1a1a1a;margin:0;padding:24px;">'
                '<p style="color:#666;font-family:sans-serif;font-size:13px;">'
                'No EPUB book object available.</p></body></html>'
            )
            return

        selected_para = self.paragraphs[idx]

        try:
            if hasattr(self, '_preview_toolbar') and self.paragraphs:
                chapter_list = EPUBPreviewEngine.get_chapter_list(self.paragraphs)
                current_href = selected_para.get('item_href', '')
                self._preview_toolbar.set_chapters(chapter_list, current_href)

            html_str = self._preview_engine.generate_preview_html(
                book, self.paragraphs, selected_para,
                show_original_ids=self._preview_show_original_ids,
                dark_mode=self._preview_dark_mode,
            )
            base_url = QUrl.fromLocalFile(
                os.path.abspath(book.content_dir) + os.sep
            )
            self.epub_preview_view.setContent(
                html_str.encode('utf-8'),
                'application/xhtml+xml',
                base_url
            )

            self._refresh_reader_if_open(book, selected_para)

        except Exception as exc:
            logger.error(f'Preview generation failed: {exc}', exc_info=True)
            self.epub_preview_view.setHtml(
                f'<html><body style="background:#1a1a1a;margin:0;padding:24px;">'
                f'<p style="color:#ff6b6b;font-family:monospace;font-size:13px;">'
                f'Preview error: {exc}</p></body></html>'
            )

    def _on_preview_fragment_clicked(self, elem_id: str):
        """
        Called when the user clicks a fragment element inside the Preview.
        Finds the matching paragraph, selects it in the list, and re-renders
        the preview so the blue highlight moves to the clicked element.
        """
        if not self.paragraphs:
            return

        para_index = None
        for i, para in enumerate(self.paragraphs):
            if para.get('id') == elem_id:
                para_index = i
                break

        if para_index is None:
            return

        row = self.para_to_row_map.get(para_index)
        if row is None:
            return

        item = self.list_widget.item(row)
        if item is None:
            return

        self.list_widget.blockSignals(True)
        self.list_widget.setCurrentItem(item)
        self.list_widget.scrollToItem(item, QListWidget.ScrollHint.EnsureVisible)
        self.list_widget.blockSignals(False)

        idx = para_index
        self.original_text_view.setPlainText(
            self._format_text_for_display(self.paragraphs[idx]['original_text'])
        )
        self.translated_text_view.textChanged.disconnect(self.update_translation_from_edit)
        self.translated_text_view.setPlainText(
            self._format_text_for_display(self.paragraphs[idx]['translated_text'])
        )
        self.translated_text_view.textChanged.connect(self.update_translation_from_edit)
        self._update_alignment_tab(idx)
        self._update_translation_field_state(self.paragraphs[idx])

        self.refresh_preview()

    def _on_preview_contextmenu(self, elem_id: str):
        """
        Show a native context menu when the user right-clicks an element
        inside the Preview tab.  Offers 'Show Original' / 'Show Translation'
        depending on the current display state of the element.
        """
        if not self.paragraphs:
            return

        para = next((p for p in self.paragraphs if p.get('id') == elem_id), None)
        is_translated = (
            para is not None
            and para.get('is_translated')
            and bool(para.get('translated_text', '').strip())
        )

        menu = QMenu(self)

        if elem_id in self._preview_show_original_ids:
            act = menu.addAction('Show Translation')
            act.triggered.connect(lambda: self._preview_toggle_original(elem_id, False))
        else:
            act = menu.addAction('Show Original')
            act.setEnabled(is_translated)
            act.triggered.connect(lambda: self._preview_toggle_original(elem_id, True))

        menu.exec(QCursor.pos())

    def _preview_toggle_original(self, elem_id: str, show_original: bool):
        """Add or remove *elem_id* from the 'show original' set and refresh."""
        if show_original:
            self._preview_show_original_ids.add(elem_id)
        else:
            self._preview_show_original_ids.discard(elem_id)
        self.refresh_preview()

    def _on_preview_chapter_changed(self, item_href: str):
        """
        Navigate to the first paragraph of *item_href* when the user clicks
        the Prev / Next chapter buttons in the preview toolbar.
        """
        if not self.paragraphs:
            return
        para_index = EPUBPreviewEngine.first_para_index_for_chapter(
            self.paragraphs, item_href
        )
        if para_index < 0:
            return
        row = self.para_to_row_map.get(para_index)
        if row is None:
            return
        item = self.list_widget.item(row)
        if item is None:
            return
        self.list_widget.blockSignals(True)
        self.list_widget.setCurrentItem(item)
        self.list_widget.scrollToItem(item, QListWidget.ScrollHint.EnsureVisible)
        self.list_widget.blockSignals(False)

        idx = para_index
        self.original_text_view.setPlainText(
            self._format_text_for_display(self.paragraphs[idx]['original_text'])
        )
        self.translated_text_view.textChanged.disconnect(self.update_translation_from_edit)
        self.translated_text_view.setPlainText(
            self._format_text_for_display(self.paragraphs[idx]['translated_text'])
        )
        self.translated_text_view.textChanged.connect(self.update_translation_from_edit)
        self._update_alignment_tab(idx)
        self._update_translation_field_state(self.paragraphs[idx])
        self._preview_show_original_ids.clear()
        self.refresh_preview()

    def _on_preview_dark_mode_toggled(self, dark: bool):
        """Store dark-mode preference and re-render the preview."""
        self._preview_dark_mode = dark
        self.refresh_preview()

    def _on_open_reader(self):
        book = getattr(self.file_processor, 'book', None)
        if book is None or not self.paragraphs:
            return

        current_item = self.list_widget.currentItem()
        idx = current_item.data(Qt.ItemDataRole.UserRole) if current_item else 0
        if idx is None:
            idx = 0
        selected_para = self.paragraphs[idx]

        if self._reader_window is not None:
            try:
                self._reader_window.navigate_to_para(book, self.paragraphs, idx)
                self._reader_window.raise_()
                self._reader_window.activateWindow()
                return
            except RuntimeError:
                self._reader_window = None

        self._reader_window = EPUBReaderWindow(
            book, self.paragraphs, selected_para,
            dark_mode=self._preview_dark_mode,
            parent=None,
        )
        self._reader_window.closed.connect(self._on_reader_closed)
        self._reader_window.show()

    def _on_reader_closed(self):
        self._reader_window = None

    def _refresh_reader_if_open(self, book, selected_para):
        if self._reader_window is None:
            return
        try:
            self._reader_window.refresh_chapter(book, self.paragraphs, selected_para)
        except RuntimeError:
            self._reader_window = None

    def _mark_alignment_confirmed(self):
        """Set alignment_force_correct on the current paragraph (🟢)."""
        current_item = self.list_widget.currentItem()
        if not current_item:
            return
        idx = current_item.data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        self.paragraphs[idx]['alignment_force_correct'] = True
        self.paragraphs[idx].pop('alignment_force_bad', None)
        self._update_item_visuals(idx)

    def _flag_alignment_bad(self):
        """Set alignment_force_bad on the current paragraph (🔴)."""
        current_item = self.list_widget.currentItem()
        if not current_item:
            return
        idx = current_item.data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        self.paragraphs[idx]['alignment_force_bad'] = True
        self.paragraphs[idx].pop('alignment_force_correct', None)
        self._update_item_visuals(idx)

    def on_retry_progress(self, idx: int, attempt: int, max_attempts: int, temperature: float):
        self.current_retry_attempt = attempt
        self.current_max_attempts = max_attempts

        self.translation_start_time = time.time()

        timeout_seconds = self.timeout_spinbox.value() * 60
        prefix = getattr(self, '_cancellation_prefix', '')

        display_row = self.para_to_row_map.get(idx, idx)
        display_num = display_row + 1

        if max_attempts > 1:
            status_msg = (
                f"{prefix}Translating fragment {display_num} "
                f"({self.completed_translations + 1}/{self.total_to_translate}) • "
                f"Auto-fix attempt {attempt}/{max_attempts} • "
                f"Temp: {temperature:.1f} • "
                f"⏱ 0s / {timeout_seconds}s"
            )
        else:
            status_msg = (
                f"{prefix}Translating fragment {display_num} "
                f"({self.completed_translations + 1}/{self.total_to_translate}) • "
                f"⏱ 0s / {timeout_seconds}s"
            )

        self.statusBar().showMessage(status_msg, 0)

        logging.info(
            f"Fragment {display_num} (para_index={idx}): Auto-fix attempt {attempt}/{max_attempts} started "
            f"(temp={temperature:.1f})"
        )

    def update_translation_time(self):
        if self.translation_start_time and self.current_fragment_index is not None:
            elapsed = int(time.time() - self.translation_start_time)
            idx = self.current_fragment_index

            timeout_seconds = self.timeout_spinbox.value() * 60

            prefix = getattr(self, '_cancellation_prefix', '')

            display_row = self.para_to_row_map.get(idx, idx)
            display_num = display_row + 1

            attempt = getattr(self, 'current_retry_attempt', 0)
            max_attempts = getattr(self, 'current_max_attempts', 1)

            if max_attempts > 1 and attempt > 0:
                temp = self.temperature_spinbox.value()
                if hasattr(self, 'current_worker') and self.current_worker:
                    if hasattr(self.current_worker, 'auto_fix_manager') and self.current_worker.auto_fix_manager:
                        temp = self.current_worker.auto_fix_manager.current_temperature

                status_msg = (
                    f"{prefix}Translating fragment {display_num} "
                    f"({self.completed_translations + 1}/{self.total_to_translate}) • "
                    f"Auto-fix attempt {attempt}/{max_attempts} • "
                    f"Temp: {temp:.1f} • "
                    f"⏱ {elapsed}s / {timeout_seconds}s"
                )
            else:
                status_msg = (
                    f"{prefix}Translating fragment {display_num} "
                    f"({self.completed_translations + 1}/{self.total_to_translate}) • "
                    f"⏱ {elapsed}s / {timeout_seconds}s"
                )

            self.statusBar().showMessage(status_msg, 0)

    def populate_list(self):
        self.list_widget.clear()
        self.last_checked_row = None
        self.para_to_row_map = {}
        self.row_to_para_map = {}
        display_index = 0
        for para_index, para in enumerate(self.paragraphs):
            if para.get('is_non_translatable', False):
                continue
            original_text = para.get('original_text', '')
            _check = original_text
            _check = re.sub(r'</?(?:id|p)_\d{2}>|<nt_\d{2}/>', '', _check)
            _check = re.sub(
                r'[\ufffc\ufffd\u200b\u200c\u200d\u200e\u200f'
                r'\u2028\u2029\ufeff\u00ad\u00a0]',
                '', _check
            )
            _check_alpha = ''.join(c for c in _check if c.isalpha())
            if not _check_alpha:
                para['is_non_translatable'] = True
                continue
            self.para_to_row_map[para_index] = display_index
            self.row_to_para_map[display_index] = para_index

            number = para.get('subtitle_block', display_index + 1)
            preview_text = original_text.replace('\n', ' | ')
            if len(preview_text) > 70:
                preview_text = preview_text[:67] + '...'
            item = QListWidgetItem(f"[{number}] {preview_text}")

            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, para_index)
            self.list_widget.addItem(item)
            self._update_item_visuals(para_index)
            display_index += 1

    def _get_list_row_for_para_index(self, para_index):
        return self.para_to_row_map.get(para_index, None)

    def _update_translation_field_state(self, para: dict) -> None:
        """Update the Translation tab label and editability based on alignment state."""
        is_aligned = bool(
            self.file_type == 'epub'
            and para.get('processing_mode') == 'legacy'
            and para.get('aligned_translated_html')
        )
        if is_aligned:
            self.trans_lbl.setText("TRANSLATION  (EDITABLE IN ALIGNMENT TAB)")
            self.translated_text_view.setReadOnly(True)
        else:
            self.trans_lbl.setText("TRANSLATION  (EDITABLE)")
            self.translated_text_view.setReadOnly(False)

    def display_selected_fragment(self, current_item, previous_item):
        if not current_item:
            return

        idx = current_item.data(Qt.ItemDataRole.UserRole)

        if getattr(self, '_mouse_click', False):
            self._mouse_click = False
        else:
            self.list_widget.scrollToItem(current_item, QListWidget.ScrollHint.EnsureVisible)

        self.translated_text_view.textChanged.disconnect(self.update_translation_from_edit)

        self.original_text_view.setPlainText(
            self._format_text_for_display(self.paragraphs[idx]['original_text'])
        )
        self.translated_text_view.setPlainText(
            self._format_text_for_display(self.paragraphs[idx]['translated_text'])
        )

        self.translated_text_view.textChanged.connect(self.update_translation_from_edit)

        self._update_alignment_tab(idx)
        self._update_translation_field_state(self.paragraphs[idx])

        if hasattr(self, 'panel_tabs') and self.panel_tabs.currentIndex() == 2:
            self.refresh_preview()

    def _refresh_current_fragment_display(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            return
        idx = current_item.data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return

        self.original_text_view.setPlainText(
            self._format_text_for_display(self.paragraphs[idx]['original_text'])
        )
        self.translated_text_view.textChanged.disconnect(self.update_translation_from_edit)
        self.translated_text_view.setPlainText(
            self._format_text_for_display(self.paragraphs[idx]['translated_text'])
        )
        self.translated_text_view.textChanged.connect(self.update_translation_from_edit)
        self._update_alignment_tab(idx)
        self._update_translation_field_state(self.paragraphs[idx])

        if hasattr(self, 'panel_tabs') and self.panel_tabs.currentIndex() == 2:
            self.refresh_preview()

    def update_translation_from_edit(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            return

        idx = current_item.data(Qt.ItemDataRole.UserRole)

        edited_text = self._parse_text_from_display(
            self.translated_text_view.toPlainText()
        )

        was_translated = self.paragraphs[idx]['is_translated']
        is_now_translated = bool(edited_text.strip())

        self.paragraphs[idx]['translated_text'] = edited_text
        self.paragraphs[idx]['is_translated'] = is_now_translated

        if was_translated and is_now_translated:
            if self.paragraphs[idx].get('aligned_translated_html'):
                self.paragraphs[idx].pop('aligned_translated_html', None)
                self.paragraphs[idx].pop('alignment_corrections', None)
                self.paragraphs[idx].pop('alignment_uncertain', None)

        self._update_item_visuals(idx)

        if was_translated != is_now_translated:
            self.update_file_label()

    def _update_item_visuals(self, para_index_or_row):
        if para_index_or_row in self.para_to_row_map:
            row = self.para_to_row_map[para_index_or_row]

        elif para_index_or_row < self.list_widget.count():
            row = para_index_or_row
        else:
            return

        item = self.list_widget.item(row)
        if not item:
            return

        para_index = item.data(Qt.ItemDataRole.UserRole)
        if para_index is None or para_index >= len(self.paragraphs):
            return

        para = self.paragraphs[para_index]

        display_index = row
        number = para.get('subtitle_block', display_index + 1)

        if para.get('is_translated') and para.get('translated_text', '').strip():
            preview_text = para['translated_text'].replace('\n', ' | ')
        else:
            preview_text = para.get('original_text', '').replace('\n', ' | ')

        if len(preview_text) > 70:
            preview_text = preview_text[:67] + '...'

        alignment_dot = ''
        if (self.file_type == 'epub'
                and para.get('processing_mode') == 'legacy'
                and para.get('is_translated')):
            original_html = para.get('original_html', '')
            has_inline_tags = any(
                f'<{tag}' in original_html or f'<{tag}>' in original_html
                for tag in INLINE_TRANSFER_TAGS
            )
            if para.get('alignment_force_correct'):
                alignment_dot = '🟢 '
            elif para.get('alignment_force_bad'):
                alignment_dot = '🔴 '
            elif not has_inline_tags:
                alignment_dot = '⚫ '
            elif para.get('aligned_translated_html'):
                if para.get('alignment_auto_wrap'):
                    alignment_dot = '🟢 '
                else:
                    corrections = para.get('alignment_corrections', 0)
                    if corrections >= 2:
                        alignment_dot = '🟠 '
                    else:
                        alignment_dot = '🟡 '

        item.setText(f"[{number}] {alignment_dot}{preview_text}")

        if self.mismatch_checker:
            has_mismatch, flags = self.mismatch_checker.check_mismatch(para)
            para['has_mismatch'] = has_mismatch
            para['mismatch_flags'] = flags
        else:
            has_mismatch = para.get('has_mismatch', False)
            flags = para.get('mismatch_flags', {})

        is_translated = para.get("is_translated", False)
        is_forced = para.get("force_mismatch", False)

        font = item.font()
        font.setUnderline(bool(flags.get("paragraphs", False)))
        font.setItalic(flags.get("first_char", False))
        font.setStrikeOut(flags.get("last_char", False))
        font.setBold(is_forced)
        item.setFont(font)

        if has_mismatch:
            item.setForeground(QColor("#ff3333"))
        elif is_translated:
            item.setForeground(QColor("#228B22"))
        else:
            item.setForeground(QColor("#ffffff"))

        if has_mismatch or is_forced:
            tooltip = self._build_mismatch_tooltip(flags, is_forced, para)
            item.setToolTip(tooltip)
        else:
            item.setToolTip("")

    def _build_mismatch_tooltip(self, flags, is_forced, para_data):
        lines = []

        if is_forced:
            lines.append("<b>⚠ FLAGGED FOR REVIEW</b>")
            lines.append("<i>Manually marked - requires attention</i>")
            lines.append("")
        else:
            lines.append("<b>❌ MISMATCH DETECTED</b>")
            lines.append("")

        error_count = sum(1 for v in flags.values() if v)
        lines.append(f"<b>{error_count} issue(s) found:</b>")
        lines.append("")

        structure_issues = []

        if flags.get("untranslated"):
            structure_issues.append(
                "• <b>Not translated:</b> Translation is identical to original – "
                "LLM may have skipped translation"
            )

        if flags.get("paragraphs"):
            para_flag = flags["paragraphs"]
            orig_count = para_flag["orig"]
            trans_count = para_flag["trans"]
            structure_issues.append(
                f"• <b>Paragraph count:</b> Original has {orig_count}, "
                f"translation has {trans_count}"
            )

        if flags.get("length"):
            orig_text = para_data.get("original_text", "")
            trans_text = para_data.get("translated_text", "")
            orig_clean = re.sub(r'</?p_\d{2}>|<id_\d{2}>|</id_\d{2}>|</?ps>|<nt_\d{2}/>', '', orig_text)
            trans_clean = re.sub(r'</?p_\d{2}>|<id_\d{2}>|</id_\d{2}>|</?ps>|<nt_\d{2}/>', '', trans_text)
            orig_len = len(orig_clean)
            trans_len = len(trans_clean)
            ratio = trans_len / max(orig_len, 1)
            if ratio < 1.0:
                direction = "too short"
            else:
                direction = "too long"
            structure_issues.append(
                f"• <b>Length mismatch ({direction}):</b> {orig_len} → {trans_len} chars "
                f"(ratio: {ratio:.2f}x)"
            )

        if structure_issues:
            lines.append("<u>Text Structure:</u>")
            lines.extend(structure_issues)
            lines.append("")

        formatting_issues = []

        if flags.get("first_char"):
            orig_text = para_data.get("original_text", "")
            trans_text = para_data.get("translated_text", "")
            orig_clean = re.sub(r'</?p_\d{2}>|<id_\d{2}>|</id_\d{2}>|</?ps>|<nt_\d{2}/>', '', orig_text)
            trans_clean = re.sub(r'</?p_\d{2}>|<id_\d{2}>|</id_\d{2}>|</?ps>|<nt_\d{2}/>', '', trans_text)
            orig_desc = self._describe_first_char(orig_clean)
            trans_desc = self._describe_first_char(trans_clean)
            formatting_issues.append(
                f"• <b>First character:</b> {orig_desc} → {trans_desc}"
            )

        if flags.get("last_char"):
            orig_text = para_data.get("original_text", "")
            trans_text = para_data.get("translated_text", "")
            orig_clean = re.sub(r'</?p_\d{2}>|<id_\d{2}>|</id_\d{2}>|</?ps>|<nt_\d{2}/>', '', orig_text)
            trans_clean = re.sub(r'</?p_\d{2}>|<id_\d{2}>|</id_\d{2}>|</?ps>|<nt_\d{2}/>', '', trans_text)
            orig_desc = self._describe_last_char(orig_clean)
            trans_desc = self._describe_last_char(trans_clean)
            formatting_issues.append(
                f"• <b>Last character:</b> {orig_desc['type']} → {trans_desc['type']}"
            )

        if flags.get("quote_parity"):
            trans_text = para_data.get("translated_text", "")
            quote_count = sum(1 for ch in trans_text if ch in '"\u201C\u201D\u201E\u201F\u00AB\u00BB\u301D\u301E\u301F\uFF02')
            formatting_issues.append(
                f"• <b>Quote parity:</b> Odd number of quotes ({quote_count}) - likely unpaired"
            )

        if formatting_issues:
            lines.append("<u>Formatting:</u>")
            lines.extend(formatting_issues)
            lines.append("")

        reserve_errors = flags.get("reserve_elements")
        if reserve_errors and isinstance(reserve_errors, dict):
            missing          = reserve_errors.get("missing", [])
            extra            = reserve_errors.get("extra", [])
            spurious         = reserve_errors.get("spurious_closing", [])
            positioning_res  = reserve_errors.get("positioning", [])

            if missing or extra or spurious or positioning_res:
                lines.append("<u>Reserve Elements (&lt;id_XX&gt;):</u>")
                if missing:
                    missing_str = ", ".join(missing[:5])
                    if len(missing) > 5:
                        missing_str += f", ... (+{len(missing) - 5} more)"
                    lines.append(f"• <b>Missing:</b> {missing_str}")
                if extra:
                    extra_str = ", ".join(extra[:5])
                    if len(extra) > 5:
                        extra_str += f", ... (+{len(extra) - 5} more)"
                    lines.append(f"• <b>Extra:</b> {extra_str}")
                if spurious:
                    spurious_str = ", ".join(spurious[:5])
                    lines.append(f"• <b>Spurious closing tags (LLM bug):</b> {spurious_str}")
                    lines.append(f"  &nbsp;&nbsp;<i>&lt;id_XX&gt; are self-closing — no closing tag needed!</i>")
                if positioning_res:
                    lines.append(f"• <b>Position shifted ({len(positioning_res)} tag(s)):</b>")
                    for pos_err in positioning_res[:3]:
                        tag_id = pos_err.get('tag_id', '??')
                        desc   = pos_err.get('description', '')
                        if desc:
                            lines.append(f"  &nbsp;&nbsp;<i>{desc}</i>")
                        else:
                            orig_pos  = pos_err.get('orig_rel_pos', 0)
                            trans_pos = pos_err.get('trans_rel_pos', 0)
                            lines.append(
                                f"  &nbsp;&nbsp;id_{tag_id}: ~{orig_pos:.0%} in original "
                                f"→ ~{trans_pos:.0%} in translation"
                            )
                    if len(positioning_res) > 3:
                        lines.append(f"  &nbsp;&nbsp;... (+{len(positioning_res) - 3} more)")
                lines.append("")

        inline_errors = flags.get("inline_formatting")
        if inline_errors and isinstance(inline_errors, dict):
            lines.append("<u>Inline Formatting (&lt;p_XX&gt;):</u>")

            unexpected_tags = inline_errors.get("unexpected_tags", [])
            if unexpected_tags:
                unique_tags = list(dict.fromkeys(unexpected_tags))
                tags_str = ", ".join(f"&lt;{t}&gt;" for t in unique_tags[:5])
                if len(unique_tags) > 5:
                    tags_str += f", ... (+{len(unique_tags) - 5} more)"
                lines.append(f"• <b>Unexpected HTML tags in translation (LLM bug):</b> {tags_str}")
                lines.append(f"  &nbsp;&nbsp;<i>Translation contains raw HTML not present in original</i>")

            opening_errors = inline_errors.get("opening_tags", {})
            if opening_errors:
                missing_opens = opening_errors.get("missing", [])
                extra_opens   = opening_errors.get("extra", [])
                if missing_opens:
                    opens_str = ", ".join(missing_opens[:5])
                    if len(missing_opens) > 5:
                        opens_str += f", ... (+{len(missing_opens) - 5} more)"
                    lines.append(f"• <b>Missing opening tags:</b> {opens_str}")
                if extra_opens:
                    opens_str = ", ".join(extra_opens[:5])
                    if len(extra_opens) > 5:
                        opens_str += f", ... (+{len(extra_opens) - 5} more)"
                    lines.append(f"• <b>Extra opening tags:</b> {opens_str}")

            closing_errors = inline_errors.get("closing_tags", {})
            if closing_errors:
                missing_closes = closing_errors.get("missing", [])
                extra_closes   = closing_errors.get("extra", [])
                if missing_closes:
                    closes_str = ", ".join(missing_closes[:5])
                    if len(missing_closes) > 5:
                        closes_str += f", ... (+{len(missing_closes) - 5} more)"
                    lines.append(f"• <b>Missing closing tags:</b> {closes_str}")
                if extra_closes:
                    closes_str = ", ".join(extra_closes[:5])
                    if len(extra_closes) > 5:
                        closes_str += f", ... (+{len(extra_closes) - 5} more)"
                    lines.append(f"• <b>Extra closing tags:</b> {closes_str}")

            unpaired = inline_errors.get("unpaired_tags", [])
            if unpaired:
                lines.append(f"• <b>Unpaired tags:</b>")
                for tag_info in unpaired[:3]:
                    tag_id      = tag_info.get('tag_id', '??')
                    open_count  = tag_info.get('open_count', 0)
                    close_count = tag_info.get('close_count', 0)
                    lines.append(
                        f"  &nbsp;&nbsp;p_{tag_id}: {open_count} opening, {close_count} closing"
                    )
                if len(unpaired) > 3:
                    lines.append(f"  &nbsp;&nbsp;... (+{len(unpaired) - 3} more)")

            positioning = inline_errors.get("positioning", [])
            if positioning:
                lines.append(f"• <b>Position shifted ({len(positioning)} tag(s)):</b>")
                for pos_err in positioning[:3]:
                    tag_id = pos_err.get('tag_id', '??')
                    issue  = pos_err.get('issue', '')
                    desc   = pos_err.get('description', '')
                    if desc:
                        lines.append(f"  &nbsp;&nbsp;<i>{desc}</i>")
                    elif issue == 'coverage_mismatch':
                        orig_cov      = pos_err.get('orig_coverage', 0)
                        trans_cov     = pos_err.get('trans_coverage', 0)
                        orig_content  = pos_err.get('orig_content', '')
                        trans_content = pos_err.get('trans_content', '')
                        lines.append(
                            f"  &nbsp;&nbsp;p_{tag_id}: covers {orig_cov:.0%} in original "
                            f"→ {trans_cov:.0%} in translation"
                        )
                        if orig_content or trans_content:
                            lines.append(
                                f"  &nbsp;&nbsp;&nbsp;&nbsp;"
                                f"orig: <i>\"{orig_content[:40]}{'…' if len(orig_content) > 40 else ''}\"</i>"
                            )
                            lines.append(
                                f"  &nbsp;&nbsp;&nbsp;&nbsp;"
                                f"trans: <i>\"{trans_content[:40]}{'…' if len(trans_content) > 40 else ''}\"</i>"
                            )
                    elif issue == 'position_shift':
                        orig_s        = pos_err.get('orig_rel_start', 0)
                        trans_s       = pos_err.get('trans_rel_start', 0)
                        orig_content  = pos_err.get('orig_content', '')
                        trans_content = pos_err.get('trans_content', '')
                        lines.append(
                            f"  &nbsp;&nbsp;p_{tag_id}: starts at {orig_s:.0%} in original "
                            f"→ {trans_s:.0%} in translation"
                        )
                        if orig_content or trans_content:
                            lines.append(
                                f"  &nbsp;&nbsp;&nbsp;&nbsp;"
                                f"orig: <i>\"{orig_content[:40]}{'…' if len(orig_content) > 40 else ''}\"</i>"
                            )
                            lines.append(
                                f"  &nbsp;&nbsp;&nbsp;&nbsp;"
                                f"trans: <i>\"{trans_content[:40]}{'…' if len(trans_content) > 40 else ''}\"</i>"
                            )
                    elif issue == 'nesting_mismatch':
                        orig_parent  = pos_err.get('orig_parent')
                        trans_parent = pos_err.get('trans_parent')
                        lines.append(
                            f"  &nbsp;&nbsp;p_{tag_id}: nested inside p_{orig_parent} in original "
                            f"→ p_{trans_parent} in translation"
                        )
                if len(positioning) > 3:
                    lines.append(f"  &nbsp;&nbsp;... (+{len(positioning) - 3} more)")

            nt_errors = inline_errors.get("nt_markers", {})
            if nt_errors:
                missing_nt = nt_errors.get("missing", [])
                extra_nt   = nt_errors.get("extra", [])
                if missing_nt:
                    nt_str = ", ".join(missing_nt[:5])
                    lines.append(f"• <b>Missing NT markers (padding/anchors):</b> {nt_str}")
                    lines.append(f"  &nbsp;&nbsp;<i>Keep these markers in the same position as original</i>")
                if extra_nt:
                    nt_str = ", ".join(extra_nt[:5])
                    lines.append(f"• <b>Extra NT markers:</b> {nt_str}")

            lines.append("")

        if not is_forced:
            lines.append("<i style='color: #888;'>Tip: Use 'Mark as Correct' to ignore, or 'Unmark' to flag manually</i>")

        align_flag = flags.get("alignment_uncertain")
        if align_flag and isinstance(align_flag, dict):
            lines.append("")
            lines.append("<u>⚠ Alignment Uncertainty (Legacy mode):</u>")
            count   = align_flag.get("count", 0)
            detail  = align_flag.get("detail", [])
            lines.append(
                f"• Heuristics had to correct tag positions <b>{count}×</b> "
                f"in this paragraph — inline tags may be misplaced."
            )
            if detail:
                lines.append("  &nbsp;&nbsp;<i>Corrections applied:</i>")
                for d in detail[:4]:
                    lines.append(f"  &nbsp;&nbsp;– {d}")
                if len(detail) > 4:
                    lines.append(f"  &nbsp;&nbsp;... (+{len(detail) - 4} more)")
            lines.append(
                "  &nbsp;&nbsp;<i>Verify bold/italic/span placement in the output.</i>"
            )

        return "<br>".join(lines)

    def _update_status_after_file_load(self):
        self._update_alignment_button_visibility()

        if self.file_type == "txt":
            self.statusBar().showMessage("📄 TXT file loaded - ready to translate", 5000)
            return

        if self.file_type == "srt":
            self.statusBar().showMessage("🎬 SRT file loaded - ready to translate", 5000)
            return

        if self.file_type == "epub":
            if self.paragraphs:
                loaded_mode = self.paragraphs[0].get('processing_mode', 'inline')
            else:
                loaded_mode = 'inline'

            checkbox_mode = 'inline' if self.inline_formatting_checkbox.isChecked() else 'legacy'

            modes_match = (loaded_mode == checkbox_mode)

            if modes_match:
                mode_emoji = "🔧" if checkbox_mode == "inline" else "📦"
                mode_name = "INLINE formatting" if checkbox_mode == "inline" else "LEGACY mode"

                self.statusBar().showMessage(
                    f"📖 EPUB loaded with {mode_emoji} {mode_name} - ready to translate",
                    5000
                )
            else:
                self.statusBar().showMessage(
                    f"⚠️ EPUB loaded as {loaded_mode.upper()}, but will translate with {checkbox_mode.upper()} prompts - "
                    f"Consider reloading file for consistency",
                    0
                )

                logger.warning(
                    f"Mode mismatch: file loaded as {loaded_mode}, "
                    f"but checkbox set to {checkbox_mode}"
                )

    def update_file_label(self):
        has_file = bool(self.original_file_path and os.path.exists(self.original_file_path))

        self.btn_unload_file.setVisible(has_file)
        self.btn_hard_reset.setVisible(has_file)
        self.btn_reset_llm.setVisible(has_file)
        self.btn_save_llm_instruction.setVisible(has_file)

        if self.llm_editor_container.isVisible():
            self.single_prompt_checkbox.setVisible(has_file)

        if has_file:
            filename = os.path.basename(self.original_file_path)

            if self.file_type == "epub":
                icon = "📖"
                color = "#88ccff"
            elif self.file_type == "srt":
                icon = "🎬"
                color = "#ffaa88"
            elif self.file_type == "txt":
                icon = "📄"
                color = "#aaffaa"
            else:
                icon = "📁"
                color = "#cccccc"

            if len(filename) > 65:
                name, ext = os.path.splitext(filename)
                filename = name[:60] + "…" + ext

            translatable = [p for p in self.paragraphs if not p.get('is_non_translatable', False)]
            translated_count = sum(1 for p in translatable if p.get('is_translated', False))
            total_count = len(translatable)

            self.file_label.setText(f"{icon} {filename}  •  {translated_count}/{total_count}")

            tooltip_parts = [
                f"File: {self.original_file_path}",
                f"Type: {self.file_type.upper()}",
                f"Fragments: {total_count}",
                f"Translated: {translated_count}"
            ]

            if self.file_type == "epub" and self.paragraphs:
                loaded_mode = self.paragraphs[0].get('processing_mode', 'inline')
                checkbox_mode = 'inline' if self.inline_formatting_checkbox.isChecked() else 'legacy'

                tooltip_parts.append("")
                tooltip_parts.append(f"Loaded as: {loaded_mode.upper()}")
                tooltip_parts.append(f"Will translate with: {checkbox_mode.upper()} prompts")

                if loaded_mode != checkbox_mode:
                    tooltip_parts.append("")
                    tooltip_parts.append("⚠️ Mode mismatch - consider reloading file")

            self.file_label.setToolTip("\n".join(tooltip_parts))

            self.file_label.setStyleSheet(f"""
                QLabel {{
                    color: {color};
                    background-color: #1a1a1a;
                    padding: 6px 10px;
                    border-radius: 3px;
                    border: 1px solid #555;
                    font-size: 11px;
                    font-weight: bold;
                }}
            """)
        else:
            self.file_label.setText("📄 No file loaded")
            self.file_label.setToolTip("No file loaded")
            self.file_label.setStyleSheet("""
                QLabel {
                    color: #888888;
                    background-color: #1a1a1a;
                    padding: 6px 10px;
                    border-radius: 3px;
                    border: 1px dashed #444;
                    font-size: 11px;
                    font-style: italic;
                }
            """)

    def _update_progress(self):
        self.completed_translations += 1
        percent = int(self.completed_translations / self.total_to_translate * 100)
        self.progress_bar.setValue(percent)

    def _update_context_visibility(self):
        batch_on = hasattr(self, 'sentence_batch_checkbox') and self.sentence_batch_checkbox.isChecked()
        for widget in [
            getattr(self, '_lbl_context_before', None),
            getattr(self, 'context_before_spinbox', None),
            getattr(self, '_lbl_context_after', None),
            getattr(self, 'context_after_spinbox', None),
        ]:
            if widget is not None:
                widget.setVisible(not batch_on)

    def _show_ps_in_ui(self) -> bool:
        if not hasattr(self, 'inline_formatting_checkbox'):
            return False
        if not hasattr(self, 'show_ps_in_ui_checkbox'):
            return False
        return (
            self.inline_formatting_checkbox.isChecked()
            and self.show_ps_in_ui_checkbox.isChecked()
        )

    def _format_text_for_display(self, text: str) -> str:
        if not text or not self._show_ps_in_ui():
            return text

        return text.replace('\n', '<ps>')

    def _parse_text_from_display(self, text: str) -> str:
        if not text or not self._show_ps_in_ui():
            return text

        return text.replace('<ps>', '\n')

    def _filter_alignment_dot(self, dot: Optional[str]):
        """Filter list to show only fragments with the given alignment dot colour.
        dot values: '🟢','🟡','🟠','⚫','🔴','none', or None (show all).
        Note: alignment_force_correct also maps to '🟢'.
        """
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            idx = item.data(Qt.ItemDataRole.UserRole)
            para = self.paragraphs[idx]

            if dot is None:
                item.setHidden(False)
                continue

            original_html = para.get('original_html', '')
            has_inline_tags = any(
                f'<{tag}' in original_html or f'<{tag}>' in original_html
                for tag in INLINE_TRANSFER_TAGS
            )
            is_translated = para.get('is_translated', False)
            is_legacy_epub = (
                self.file_type == 'epub'
                and para.get('processing_mode') == 'legacy'
            )

            if not is_legacy_epub or not is_translated:
                item.setHidden(True)
                continue

            if para.get('alignment_force_correct'):
                current_dot = '🟢'
            elif para.get('alignment_force_bad'):
                current_dot = '🔴'
            elif not has_inline_tags:
                current_dot = '⚫'
            elif para.get('aligned_translated_html'):
                if para.get('alignment_auto_wrap'):
                    current_dot = '🟢'
                else:
                    corrections = para.get('alignment_corrections', 0)
                    current_dot = '🟠' if corrections >= 2 else '🟡'
            else:
                current_dot = 'none'

            item.setHidden(current_dot != dot)

    def filter_list(self, show_translated):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            idx = item.data(Qt.ItemDataRole.UserRole)
            is_translated = self.paragraphs[idx]['is_translated']
            if show_translated is None:
                item.setHidden(False)
            else:
                item.setHidden(is_translated != show_translated)

    def filter_mismatch(self, show_mismatch: bool):
        if not self.mismatch_checker:
            return

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            idx = item.data(Qt.ItemDataRole.UserRole)
            has_mismatch, _ = self.mismatch_checker.check_mismatch(self.paragraphs[idx])
            item.setHidden(has_mismatch != show_mismatch)

    def filter_search(self):
        phrase = self.search_edit.text().lower().strip()
        mode = self.search_mode_combo.currentText()

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            idx = item.data(Qt.ItemDataRole.UserRole)
            para = self.paragraphs[idx]
            text = para['original_text'] if mode == "Original" else para['translated_text']
            visible = True if not phrase else (phrase in text.lower())
            item.setHidden(not visible)

        if phrase:
            self.btn_select_all.setText("Select Filtered")
            self.btn_deselect_all.setText("Deselect Filtered")
        else:
            self.btn_select_all.setText("Select All")
            self.btn_deselect_all.setText("Deselect All")

    def toggle_all_selection(self, check):
        state = Qt.CheckState.Checked if check else Qt.CheckState.Unchecked
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item.isHidden():
                item.setCheckState(state)

    def toggle_selection_by_translated(self, translated: bool):
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            para_index = item.data(Qt.ItemDataRole.UserRole)

            if para_index is not None and para_index < len(self.paragraphs):
                if self.paragraphs[para_index]['is_translated'] == translated:
                    item.setCheckState(Qt.CheckState.Checked)
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)

    def toggle_selection_mismatch(self, select: bool):
        if not self.mismatch_checker:
            return

        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            para_index = item.data(Qt.ItemDataRole.UserRole)

            if para_index is not None and para_index < len(self.paragraphs):
                has_mismatch, _ = self.mismatch_checker.check_mismatch(self.paragraphs[para_index])
                if has_mismatch == select:
                    item.setCheckState(Qt.CheckState.Checked)
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)

    def on_list_item_clicked(self, item):
        self._mouse_click = True

        current_row = self.list_widget.row(item)

        modifiers = QApplication.keyboardModifiers()
        is_shift = modifiers == Qt.KeyboardModifier.ShiftModifier

        if is_shift and self.last_checked_row is not None:
            start_row = min(self.last_checked_row, current_row)
            end_row = max(self.last_checked_row, current_row)

            target_state = item.checkState()

            for row in range(start_row, end_row + 1):
                list_item = self.list_widget.item(row)
                if list_item and not list_item.isHidden():
                    list_item.setCheckState(target_state)

        self.last_checked_row = current_row

    def eventFilter(self, source, event):
        if source is self.list_widget.viewport():
            if event.type() == event.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_start_y = event.position().y()
                self._drag_last_y = event.position().y()
                self._drag_scrolling = False
                self._drag_threshold_passed = False

            elif event.type() == event.Type.MouseMove and event.buttons() == Qt.MouseButton.LeftButton:
                current_y = event.position().y()
                delta_from_start = current_y - self._drag_start_y

                if not self._drag_threshold_passed and abs(delta_from_start) > 5:
                    self._drag_threshold_passed = True
                    self._drag_scrolling = True
                    self._drag_last_y = current_y

                if self._drag_scrolling:
                    step = int(self._drag_last_y - current_y)
                    self._drag_last_y = current_y
                    scrollbar = self.list_widget.verticalScrollBar()
                    scrollbar.setValue(scrollbar.value() + step)
                    return True

            elif event.type() == event.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                if self._drag_scrolling:
                    self._drag_scrolling = False
                    self._drag_threshold_passed = False
                    return True

        return super().eventFilter(source, event)

    def copy_original_to_translation(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            self.show_message("No Selection", "Please select a fragment first.", QMessageBox.Icon.Warning)
            return

        checked_indices = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                idx = item.data(Qt.ItemDataRole.UserRole)
                checked_indices.append(idx)

        current_idx = current_item.data(Qt.ItemDataRole.UserRole)

        if checked_indices:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Question)
            msg_box.setWindowTitle("Copy Original to Translation")
            msg_box.setText(f"What would you like to copy?\n\n"
                           f"• Current fragment only (#{current_idx + 1})\n"
                           f"• All {len(checked_indices)} checked fragments")

            btn_current = msg_box.addButton("Current Only", QMessageBox.ButtonRole.YesRole)
            btn_all_checked = msg_box.addButton(f"All Checked ({len(checked_indices)})", QMessageBox.ButtonRole.NoRole)
            btn_cancel = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

            msg_box.exec()
            clicked_button = msg_box.clickedButton()

            if clicked_button == btn_cancel:
                return
            elif clicked_button == btn_current:
                self._copy_single_fragment(current_idx)
                self.statusBar().showMessage(f"Fragment #{current_idx + 1} copied.", 2000)
            elif clicked_button == btn_all_checked:
                self._copy_multiple_fragments(checked_indices)
                self.statusBar().showMessage(f"{len(checked_indices)} fragments copied.", 3000)
        else:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Question)
            msg_box.setWindowTitle("Copy Original to Translation")
            msg_box.setText(f"Copy original text to translation for fragment #{current_idx + 1}?")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)

            result = msg_box.exec()

            if result == QMessageBox.StandardButton.Yes:
                self._copy_single_fragment(current_idx)
                self.statusBar().showMessage(f"Fragment #{current_idx + 1} copied.", 2000)

    def _copy_single_fragment(self, idx):
        original_text = self.paragraphs[idx]['original_text']

        self.paragraphs[idx]['translated_text'] = original_text
        self.paragraphs[idx]['is_translated'] = True

        self.paragraphs[idx]['has_mismatch'] = False
        self.paragraphs[idx]['mismatch_flags'] = {}
        if 'ignore_mismatch' in self.paragraphs[idx]:
            del self.paragraphs[idx]['ignore_mismatch']
        if 'force_mismatch' in self.paragraphs[idx]:
            del self.paragraphs[idx]['force_mismatch']

        current_item = self.list_widget.currentItem()
        if current_item and current_item.data(Qt.ItemDataRole.UserRole) == idx:
            self.translated_text_view.textChanged.disconnect(self.update_translation_from_edit)
            self.translated_text_view.setText(original_text)
            self.translated_text_view.textChanged.connect(self.update_translation_from_edit)

        self._update_item_visuals(idx)

        row = self.para_to_row_map.get(idx)
        if row is not None:
            item = self.list_widget.item(row)
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _copy_multiple_fragments(self, indices):
        for idx in indices:
            self._copy_single_fragment(idx)

        self.update_file_label()

    def mark_fragment_as_correct(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            self.statusBar().showMessage("Please select a fragment first.", 2000)
            return

        idx = current_item.data(Qt.ItemDataRole.UserRole)
        para = self.paragraphs[idx]

        if not para.get('is_translated', False):
            self.statusBar().showMessage("Translate this fragment first.", 2000)
            return

        para['ignore_mismatch'] = True
        para['force_mismatch'] = False
        para['has_mismatch'] = False
        para['mismatch_flags'] = {}

        self._update_item_visuals(idx)
        self.statusBar().showMessage("Fragment marked as CORRECT (mismatch ignored).", 2000)

    def unmark_fragment_as_correct(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            self.statusBar().showMessage("Please select a fragment first.", 2000)
            return

        idx = current_item.data(Qt.ItemDataRole.UserRole)
        para = self.paragraphs[idx]

        if 'ignore_mismatch' in para:
            del para['ignore_mismatch']

        para['force_mismatch'] = True

        self.mismatch_checker.check_mismatch(para)

        self._update_item_visuals(idx)
        self.statusBar().showMessage("Fragment FLAGGED for review.", 2000)

    def translate_with_quick_service(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            self.show_message("No Selection", "Please select a fragment to translate.", QMessageBox.Icon.Warning)
            return

        current_idx = current_item.data(Qt.ItemDataRole.UserRole)

        checked_indices = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                idx = item.data(Qt.ItemDataRole.UserRole)
                checked_indices.append(idx)

        service = self.quick_translate_service_combo.currentText()
        source_lang = (self.source_lang_combo.currentData(Qt.ItemDataRole.UserRole) or "").strip().lower()
        target_lang = (self.target_lang_combo.currentData(Qt.ItemDataRole.UserRole) or "").strip().lower()

        if not target_lang:
            self.show_message("Missing Target Language", "Please select a target language.", QMessageBox.Icon.Warning)
            return

        if checked_indices:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Question)
            msg_box.setWindowTitle("Quick Translate")
            msg_box.setText(
                f"What would you like to translate using [{service}]?\n\n"
                f"• Current fragment only (#{current_idx + 1})\n"
                f"• All {len(checked_indices)} checked fragments"
            )

            btn_current = msg_box.addButton("Current Only", QMessageBox.ButtonRole.YesRole)
            btn_all_checked = msg_box.addButton(f"All Checked ({len(checked_indices)})", QMessageBox.ButtonRole.NoRole)
            btn_cancel = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

            msg_box.exec()
            clicked = msg_box.clickedButton()

            if clicked == btn_cancel:
                return
            elif clicked == btn_current:
                self._quick_translate_single(current_idx, service, source_lang, target_lang)
            elif clicked == btn_all_checked:
                total_chars = sum(
                    len(self.paragraphs[idx].get('original_text', ''))
                    for idx in checked_indices
                )
                confirm_box = QMessageBox(self)
                confirm_box.setIcon(QMessageBox.Icon.Warning)
                confirm_box.setWindowTitle("Bulk Translation Warning")
                confirm_box.setText(
                    f"You are about to translate {len(checked_indices)} fragments "
                    f"(~{total_chars:,} characters total) using [{service}].\n\n"
                    f"⚠️  Warning: Bulk translation may quickly exceed free API character limits!\n\n"
                    f"Are you sure you want to continue?"
                )
                confirm_box.setStandardButtons(
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                confirm_box.setDefaultButton(QMessageBox.StandardButton.No)

                if confirm_box.exec() != QMessageBox.StandardButton.Yes:
                    return

                self._quick_translate_multiple(checked_indices, service, source_lang, target_lang)
        else:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Question)
            msg_box.setWindowTitle("Quick Translate")
            msg_box.setText(
                f"Translate fragment #{current_idx + 1} using [{service}]?"
            )
            msg_box.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
            )

            if msg_box.exec() == QMessageBox.StandardButton.Yes:
                self._quick_translate_single(current_idx, service, source_lang, target_lang)

    def _quick_translate_single(self, idx: int, service: str, source_lang: str, target_lang: str):
        original_text = self.paragraphs[idx]['original_text']

        text_to_translate, had_newlines, original_parts, original_separators = \
            self._prepare_text_for_quick_translate(original_text)

        try:
            translated_text = self._do_quick_translate(
                text=text_to_translate,
                service=service,
                source_lang=source_lang,
                target_lang=target_lang
            )
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Quick translate error [{service}] idx={idx}: {error_msg}")
            self.show_message("Translation Error", f"[{service}] {error_msg}", QMessageBox.Icon.Critical)
            self.paragraphs[idx]['translated_text'] = "Translation failed"
            self.paragraphs[idx]['is_translated'] = False
            self._update_item_visuals(idx)
            return

        translated_text = self._restore_quick_translate_structure(
            translated_text, had_newlines, original_parts, original_separators
        )

        self.paragraphs[idx]['translated_text'] = translated_text
        self.paragraphs[idx]['is_translated'] = True
        self.paragraphs[idx].pop('aligned_translated_html', None)
        self.paragraphs[idx].pop('alignment_corrections', None)
        self.paragraphs[idx].pop('alignment_uncertain', None)
        self._update_item_visuals(idx)

        row = self.para_to_row_map.get(idx)
        if row is not None:
            item = self.list_widget.item(row)
            if item and self.paragraphs[idx].get('is_translated', False):
                item.setCheckState(Qt.CheckState.Unchecked)

        self.update_file_label()

        current_item = self.list_widget.currentItem()
        if current_item and current_item.data(Qt.ItemDataRole.UserRole) == idx:
            self.translated_text_view.textChanged.disconnect(self.update_translation_from_edit)
            self.translated_text_view.setPlainText(
                self._format_text_for_display(translated_text)
            )
            self.translated_text_view.textChanged.connect(self.update_translation_from_edit)
            self._update_alignment_tab(idx)

        self.statusBar().showMessage(f"[{service}] Fragment #{idx + 1} translated.", 3000)

    def _quick_translate_multiple(self, indices: list, service: str, source_lang: str, target_lang: str):
        success_count = 0
        error_count = 0
        first_error_msg = None

        for i, idx in enumerate(indices):
            self.statusBar().showMessage(
                f"[{service}] Translating fragment {i + 1}/{len(indices)} (#{idx + 1})...", 0
            )

            QApplication.processEvents()

            original_text = self.paragraphs[idx].get('original_text', '')
            if not original_text.strip():
                row = self.para_to_row_map.get(idx)
                if row is not None:
                    item = self.list_widget.item(row)
                    if item:
                        item.setCheckState(Qt.CheckState.Unchecked)
                continue

            text_to_translate, had_newlines, original_parts, original_separators = \
                self._prepare_text_for_quick_translate(original_text)

            try:
                translated_text = self._do_quick_translate(
                    text=text_to_translate,
                    service=service,
                    source_lang=source_lang,
                    target_lang=target_lang
                )

                translated_text = self._restore_quick_translate_structure(
                    translated_text, had_newlines, original_parts, original_separators
                )

                self.paragraphs[idx]['translated_text'] = translated_text
                self.paragraphs[idx]['is_translated'] = True
                self._update_item_visuals(idx)

                row = self.para_to_row_map.get(idx)
                if row is not None:
                    item = self.list_widget.item(row)
                    if item and self.paragraphs[idx].get('is_translated', False):
                        item.setCheckState(Qt.CheckState.Unchecked)

                current_item = self.list_widget.currentItem()
                if current_item and current_item.data(Qt.ItemDataRole.UserRole) == idx:
                    self.translated_text_view.textChanged.disconnect(self.update_translation_from_edit)
                    self.translated_text_view.setPlainText(
                        self._format_text_for_display(translated_text)
                    )
                    self.translated_text_view.textChanged.connect(self.update_translation_from_edit)

                success_count += 1

            except Exception as e:
                error_msg = str(e)
                logging.error(f"Quick translate error [{service}] idx={idx}: {error_msg}")
                self.paragraphs[idx]['translated_text'] = "Translation failed"
                self.paragraphs[idx]['is_translated'] = False
                self._update_item_visuals(idx)
                error_count += 1
                if first_error_msg is None:
                    first_error_msg = error_msg

                break

        self.update_file_label()

        if error_count == 0:
            self.statusBar().showMessage(
                f"[{service}] All {success_count} fragments translated successfully.", 5000
            )
        else:
            remaining = len(indices) - success_count - error_count
            self.show_message(
                "Translation Stopped",
                f"[{service}] Translated {success_count} fragment(s) successfully.\n"
                f"Stopped due to error:\n\n{first_error_msg}\n\n"
                f"({remaining + error_count} fragment(s) were not translated.)",
                QMessageBox.Icon.Warning
            )

    def _do_quick_translate(self, text: str, service: str, source_lang: str, target_lang: str) -> str:
        if source_lang in ("", "auto", "detect"):
            source_lang = "auto"

        if service == "Google (Free)":
            try:
                pass
            except ImportError:
                raise Exception(
                    "deep-translator is not installed.\n"
                    "Run: pip install deep-translator"
                )
            try:
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                result = translator.translate(text)
                if not result:
                    raise Exception("Google Translate returned an empty response.")
                return result
            except Exception as e:
                err = str(e)
                if "invalid" in err.lower() or "language" in err.lower():
                    raise Exception(
                        f"Invalid language code for Google Translate.\n"
                        f"Source: '{source_lang}', Target: '{target_lang}'\n"
                        f"Details: {err}"
                    )
                raise Exception(f"Google Translate error: {err}")

        elif service in ("DeepL Free", "DeepL Pro"):
            try:
                pass
            except ImportError:
                raise Exception(
                    "Biblioteka deepl nie jest zainstalowana.\n"
                    "Uruchom: pip install deepl"
                )

            use_free = (service == "DeepL Free")
            api_key = (
                self.app_settings.get("deepl_free_api_key", "")
                if use_free
                else self.app_settings.get("deepl_pro_api_key", "")
            )

            if not api_key:
                raise Exception(
                    f"Missing API key for {service}.\n"
                    f"Please enter your key in the Options tab and save settings."
                )

            try:
                translator = deepl.Translator(api_key)
                source = None if source_lang == "auto" else source_lang.upper()
                result = translator.translate_text(
                    text,
                    source_lang=source,
                    target_lang=target_lang.upper()
                )
                return str(result)
            except deepl.AuthorizationException:
                raise Exception(
                    f"Invalid API key for {service}.\n"
                    f"Please check your key in the Options tab."
                )
            except deepl.QuotaExceededException:
                raise Exception(f"{service} quota exceeded. Check your account.")
            except deepl.TooManyRequestsException:
                raise Exception(f"{service}: Too many requests. Please wait a moment.")
            except deepl.DeepLException as e:
                raise Exception(f"{service} error: {e}")
            except Exception as e:
                raise Exception(f"{service} unexpected error: {e}")

        else:
            raise Exception(f"Unknown service: {service}")

    def _prepare_text_for_quick_translate(self, original_text: str):
        _restore_on = (
            hasattr(self, 'restore_paragraph_checkbox')
            and self.restore_paragraph_checkbox.isChecked()
        )

        if '\n' not in original_text or not _restore_on or self.file_type == 'srt':
            return original_text, False, [], []

        tokens = re.split(r'(\n\n|\n)', original_text)
        original_parts = []
        original_separators = []
        for i, token in enumerate(tokens):
            if i % 2 == 0:
                original_parts.append(token)
            else:
                original_separators.append(token)

        non_empty_parts = [p for p in original_parts if p.strip()]
        if len(non_empty_parts) <= 1:
            return original_text, False, [], []

        text_to_send = '<ps>'.join(p.strip() for p in original_parts if p.strip())
        return text_to_send, True, original_parts, original_separators

    def _restore_quick_translate_structure(
        self,
        translated_text: str,
        had_newlines: bool,
        original_parts: list,
        original_separators: list
    ) -> str:
        if not had_newlines:
            return translated_text

        n_parts = len([p for p in original_parts if p.strip()])

        if '<ps>' in translated_text:
            trans_parts = [p.strip() for p in translated_text.split('<ps>')]
            trans_parts = [p for p in trans_parts if p]

            if len(trans_parts) == n_parts:
                result = ''
                for i, part in enumerate(trans_parts):
                    result += part
                    if i < len(original_separators):
                        result += original_separators[i]
                logger.debug(
                    f"Quick translate: restored {n_parts} paragraphs via <ps> markers"
                )
                return result

            elif len(trans_parts) > 1:
                dominant_sep = original_separators[0] if original_separators else '\n'
                logger.warning(
                    f"Quick translate <ps> count mismatch: expected {n_parts}, "
                    f"got {len(trans_parts)} — joining with {repr(dominant_sep)}"
                )
                return dominant_sep.join(trans_parts)

        logger.warning(
            f"Quick translate: <ps> markers lost in translation "
            f"(expected {n_parts} parts) — returning flat text"
        )
        return translated_text

    def _get_current_prompts_from_cache(self, variant: str) -> Dict[str, str]:
        if variant not in self.current_prompts_cache:
            self.current_prompts_cache[variant] = self.prompt_manager.load_prompts_for_variant(variant)
            logger.debug(f"Loaded prompts into cache for variant: {variant}")

        return self.current_prompts_cache[variant]

    def toggle_llm_editor(self):
        is_visible = self.llm_editor_container.isVisible()
        self.llm_editor_container.setVisible(not is_visible)

        if not is_visible:
            self.update_llm_editor_content()
            has_file = bool(self.original_file_path)
            self.single_prompt_checkbox.setVisible(has_file)
            self.json_payload_checkbox.setVisible(has_file)
            is_json = self.json_payload_checkbox.isChecked()
            self.json_response_field_label.setVisible(has_file and is_json)
            self.json_response_field_edit.setVisible(has_file and is_json)

    def update_llm_editor_content(self):
        for i in reversed(range(self.llm_editor_layout.count())):
            item = self.llm_editor_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()

        has_file = bool(self.original_file_path)

        self.single_prompt_checkbox.setVisible(has_file and not self.json_payload_checkbox.isChecked())
        self.json_payload_checkbox.setVisible(has_file)
        is_json = self.json_payload_checkbox.isChecked()
        self.json_response_field_label.setVisible(has_file and is_json)
        self.json_response_field_edit.setVisible(has_file and is_json)

        if not has_file:
            label = QLabel("Load a file first to edit LLM prompts")
            label.setStyleSheet("color: #888; padding: 20px;")
            self.llm_editor_layout.addWidget(label)
            for attr in ('ollama_prompt_edit', 'system_prompt_edit',
                         'assistant_prompt_edit', 'user_prompt_edit',
                         'json_payload_edit', 'batch_hint_edit'):
                if hasattr(self, attr):
                    delattr(self, attr)
            return

        batch_mode = hasattr(self, 'sentence_batch_checkbox') and self.sentence_batch_checkbox.isChecked()

        outer_splitter = QSplitter(Qt.Orientation.Vertical)

        if self.json_payload_checkbox.isChecked():
            main_widget = QWidget()
            main_layout = QVBoxLayout(main_widget)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(4)

            if batch_mode:
                info_label = QLabel("JSON Payload Template (Batch mode)  —  available variable: {core_text}  (context disabled in batch)")
            else:
                info_label = QLabel("JSON Payload Template  —  available variables: {core_text}  {context_before}  {context_after}")
            info_label.setStyleSheet("font-weight: bold; color: #cc8800;")
            main_layout.addWidget(info_label)

            self.json_payload_edit = QTextEdit()
            variant = self._get_current_variant()
            cached_template = ""
            if variant and variant in self.current_prompts_cache:
                cache_key = 'batch_json_payload' if batch_mode else 'json_payload'
                cached_template = self.current_prompts_cache[variant].get(cache_key, '')
            elif variant:
                loaded = self.prompt_manager.load_prompts_for_variant(variant)
                cache_key = 'batch_json_payload' if batch_mode else 'json_payload'
                cached_template = loaded.get(cache_key, '')
            if not cached_template:
                if batch_mode:
                    cached_template = self.prompt_manager.get_default_batch_json_payload_prompt(variant or '')
                else:
                    cached_template = getattr(self, '_json_payload_content', '')
            if cached_template:
                self.json_payload_edit.setPlainText(cached_template)
            else:
                self.json_payload_edit.setPlainText(
                    self.prompt_manager.get_default_json_payload_prompt(variant or '')
                )
            cached_response_field = self.app_settings.get('json_response_field', '')
            if cached_response_field:
                self.json_response_field_edit.setText(cached_response_field)
            try:
                self.json_payload_edit.textChanged.disconnect()
            except:
                pass
            self.json_payload_edit.textChanged.connect(self.on_json_payload_content_changed)
            main_layout.addWidget(self.json_payload_edit)

            for attr in ('ollama_prompt_edit', 'system_prompt_edit',
                         'assistant_prompt_edit', 'user_prompt_edit'):
                if hasattr(self, attr):
                    delattr(self, attr)

            outer_splitter.addWidget(main_widget)

        else:
            variant = self._get_current_variant()
            prompts = self._get_current_prompts_from_cache(variant)
            llm_choice = self.app_settings.get("llm_choice", "LM Studio")

            if llm_choice == "Ollama":
                main_widget = QWidget()
                main_layout = QVBoxLayout(main_widget)
                main_layout.setContentsMargins(0, 0, 0, 0)
                main_layout.setSpacing(4)

                label = QLabel("Full Ollama Prompt (system + context + user):")
                label.setStyleSheet("font-weight: bold; color: #0066cc;")
                main_layout.addWidget(label)

                self.ollama_prompt_edit = QTextEdit()
                self.ollama_prompt_edit.setPlainText(prompts['ollama'])
                try:
                    self.ollama_prompt_edit.textChanged.disconnect()
                except:
                    pass
                self.ollama_prompt_edit.textChanged.connect(self.on_ollama_prompt_changed)
                main_layout.addWidget(self.ollama_prompt_edit)

                for attr in ('system_prompt_edit', 'assistant_prompt_edit',
                             'user_prompt_edit', 'json_payload_edit'):
                    if hasattr(self, attr):
                        delattr(self, attr)

                outer_splitter.addWidget(main_widget)

            else:
                horiz_splitter = QSplitter(Qt.Orientation.Horizontal)

                system_container = QWidget()
                system_layout = QVBoxLayout(system_container)
                system_layout.setContentsMargins(0, 0, 0, 0)
                system_label = QLabel("System Prompt:")
                system_label.setStyleSheet("font-weight: bold; color: #0066cc;")
                system_layout.addWidget(system_label)
                self.system_prompt_edit = QTextEdit()
                self.system_prompt_edit.setPlainText(prompts['system'])
                try:
                    self.system_prompt_edit.textChanged.disconnect()
                except:
                    pass
                self.system_prompt_edit.textChanged.connect(self.on_system_prompt_changed)
                system_layout.addWidget(self.system_prompt_edit)
                horiz_splitter.addWidget(system_container)

                assistant_container = QWidget()
                assistant_layout = QVBoxLayout(assistant_container)
                assistant_layout.setContentsMargins(0, 0, 0, 0)
                assistant_label = QLabel("Assistant Instruction / Context:")
                assistant_label.setStyleSheet("font-weight: bold; color: #009900;")
                assistant_layout.addWidget(assistant_label)
                self.assistant_prompt_edit = QTextEdit()
                asst_key = 'batch_assistant' if batch_mode else 'assistant'
                self.assistant_prompt_edit.setPlainText(prompts.get(asst_key, ''))
                try:
                    self.assistant_prompt_edit.textChanged.disconnect()
                except:
                    pass
                self.assistant_prompt_edit.textChanged.connect(self.on_assistant_prompt_changed)
                assistant_layout.addWidget(self.assistant_prompt_edit)
                horiz_splitter.addWidget(assistant_container)

                user_container = QWidget()
                user_layout = QVBoxLayout(user_container)
                user_layout.setContentsMargins(0, 0, 0, 0)
                user_label = QLabel("User Prompt:")
                user_label.setStyleSheet("font-weight: bold; color: #cc6600;")
                user_layout.addWidget(user_label)
                self.user_prompt_edit = QTextEdit()
                self.user_prompt_edit.setPlainText(prompts['user'])
                try:
                    self.user_prompt_edit.textChanged.disconnect()
                except:
                    pass
                self.user_prompt_edit.textChanged.connect(self.on_user_prompt_changed)
                user_layout.addWidget(self.user_prompt_edit)
                horiz_splitter.addWidget(user_container)

                for attr in ('ollama_prompt_edit', 'json_payload_edit'):
                    if hasattr(self, attr):
                        delattr(self, attr)

                outer_splitter.addWidget(horiz_splitter)

        if batch_mode:
            batch_widget = QWidget()
            batch_layout = QVBoxLayout(batch_widget)
            batch_layout.setContentsMargins(0, 4, 0, 0)
            batch_layout.setSpacing(4)

            lbl = QLabel(
                "Batch Hint  —  appended to prompt when Sentence batching is ON  "
                "(variables: {n} = segment count, {marker_count} = n−1):"
            )
            lbl.setStyleSheet("font-weight: bold; color: #aa44cc;")
            batch_layout.addWidget(lbl)

            self.batch_hint_edit = QTextEdit()
            variant = self._get_current_variant()
            cached = ""
            if variant and variant in self.current_prompts_cache:
                cached = self.current_prompts_cache[variant].get('batch_hint', '')
            if not cached:
                cached = self.prompt_manager.get_default_batch_hint_prompt()
            self.batch_hint_edit.setPlainText(cached)
            try:
                self.batch_hint_edit.textChanged.disconnect()
            except:
                pass
            self.batch_hint_edit.textChanged.connect(self.on_batch_hint_changed)
            batch_layout.addWidget(self.batch_hint_edit)

            outer_splitter.addWidget(batch_widget)
            outer_splitter.setSizes([300, 160])
        elif hasattr(self, 'batch_hint_edit'):
            delattr(self, 'batch_hint_edit')

        self.llm_editor_layout.addWidget(outer_splitter)

    def on_batch_hint_changed(self):
        if not hasattr(self, 'batch_hint_edit'):
            return
        variant = self._get_current_variant()
        if not variant:
            return
        if variant not in self.current_prompts_cache:
            self.current_prompts_cache[variant] = {}
        self.current_prompts_cache[variant]['batch_hint'] = self.batch_hint_edit.toPlainText()

    def on_ollama_prompt_changed(self):
        if not hasattr(self, 'ollama_prompt_edit'):
            return

        variant = self._get_current_variant()
        if not variant:
            return

        if variant not in self.current_prompts_cache:
            self.current_prompts_cache[variant] = {}

        self.current_prompts_cache[variant]['ollama'] = self.ollama_prompt_edit.toPlainText()

        logger.debug(f"Ollama prompt updated in cache (variant: {variant})")

    def on_system_prompt_changed(self):
        if not hasattr(self, 'system_prompt_edit'):
            return

        variant = self._get_current_variant()
        if not variant:
            return

        if variant not in self.current_prompts_cache:
            self.current_prompts_cache[variant] = {}

        self.current_prompts_cache[variant]['system'] = self.system_prompt_edit.toPlainText()

        logger.debug(f"System prompt updated in cache (variant: {variant})")

    def on_assistant_prompt_changed(self):
        if not hasattr(self, 'assistant_prompt_edit'):
            return

        variant = self._get_current_variant()
        if not variant:
            return

        if variant not in self.current_prompts_cache:
            self.current_prompts_cache[variant] = {}

        batch_mode = hasattr(self, 'sentence_batch_checkbox') and self.sentence_batch_checkbox.isChecked()
        cache_key = 'batch_assistant' if batch_mode else 'assistant'
        self.current_prompts_cache[variant][cache_key] = self.assistant_prompt_edit.toPlainText()

        logger.debug(f"Assistant prompt updated in cache (variant: {variant}, key: {cache_key})")

    def on_user_prompt_changed(self):
        if not hasattr(self, 'user_prompt_edit'):
            return

        variant = self._get_current_variant()
        if not variant:
            return

        if variant not in self.current_prompts_cache:
            self.current_prompts_cache[variant] = {}

        self.current_prompts_cache[variant]['user'] = self.user_prompt_edit.toPlainText()

        logger.debug(f"User prompt updated in cache (variant: {variant})")

    def _on_json_payload_toggled(self, state):
        is_json = bool(state)
        self.json_response_field_label.setVisible(is_json)
        self.json_response_field_edit.setVisible(is_json)
        if is_json:
            self.single_prompt_checkbox.setVisible(False)
        else:
            has_file = bool(self.original_file_path)
            self.single_prompt_checkbox.setVisible(has_file)
        if self.llm_editor_container.isVisible():
            self.update_llm_editor_content()
        self._update_context_visibility()

    def on_json_payload_content_changed(self):
        if not hasattr(self, 'json_payload_edit'):
            return
        batch_mode = hasattr(self, 'sentence_batch_checkbox') and self.sentence_batch_checkbox.isChecked()
        variant = self._get_current_variant()
        if batch_mode:
            if variant:
                if variant not in self.current_prompts_cache:
                    self.current_prompts_cache[variant] = {}
                self.current_prompts_cache[variant]['batch_json_payload'] = self.json_payload_edit.toPlainText()
        else:
            self._json_payload_content = self.json_payload_edit.toPlainText()
            if variant:
                if variant not in self.current_prompts_cache:
                    self.current_prompts_cache[variant] = {}
                self.current_prompts_cache[variant]['json_payload'] = self.json_payload_edit.toPlainText()
        logger.debug(f"JSON payload template updated in memory (batch={batch_mode})")

    def save_llm_instruction(self):
        variant = self._get_current_variant()
        if not variant:
            self.show_message("No File Loaded", "Please load a file first.", QMessageBox.Icon.Warning)
            return

        try:
            if self.json_payload_checkbox.isChecked():
                if not hasattr(self, 'json_payload_edit'):
                    self.show_message("No Changes", "No JSON payload template to save.", QMessageBox.Icon.Warning)
                    return

                template = self.json_payload_edit.toPlainText().strip()
                response_field = self.json_response_field_edit.text().strip()
                batch_mode = hasattr(self, 'sentence_batch_checkbox') and self.sentence_batch_checkbox.isChecked()

                if batch_mode:
                    self.prompt_manager.save_prompt(variant, 'batch_json_payload', template)
                    if variant not in self.current_prompts_cache:
                        self.current_prompts_cache[variant] = {}
                    self.current_prompts_cache[variant]['batch_json_payload'] = template
                else:
                    self.prompt_manager.save_prompt(variant, 'json_payload', template)
                    if variant not in self.current_prompts_cache:
                        self.current_prompts_cache[variant] = {}
                    self.current_prompts_cache[variant]['json_payload'] = template
                    self._json_payload_content = template

                self.app_settings['json_response_field'] = response_field
                AppSettingsManager.save_settings(self.app_settings)

                if hasattr(self, 'batch_hint_edit'):
                    bh = self.batch_hint_edit.toPlainText().strip()
                    self.prompt_manager.save_prompt(variant, 'batch_hint', bh)
                    self.current_prompts_cache[variant]['batch_hint'] = bh

                logger.info(f"Saved JSON payload template to file and response_field to settings: {variant}")
                self.show_message("Success", f"JSON payload template saved for: {variant}")
                return

            if variant not in self.current_prompts_cache:
                self.show_message("No Changes", "No prompts in cache to save.", QMessageBox.Icon.Warning)
                return

            prompts = self.current_prompts_cache[variant]
            llm_choice = self.app_settings.get("llm_choice", "LM Studio")
            batch_mode = hasattr(self, 'sentence_batch_checkbox') and self.sentence_batch_checkbox.isChecked()

            if llm_choice == "Ollama":
                if 'ollama' in prompts:
                    self.prompt_manager.save_prompt(variant, 'ollama', prompts['ollama'])
                    logger.info(f"Saved Ollama prompt to file: {variant}")
            else:
                if 'system' in prompts:
                    self.prompt_manager.save_prompt(variant, 'system', prompts['system'])
                if batch_mode:
                    if 'batch_assistant' in prompts:
                        self.prompt_manager.save_prompt(variant, 'batch_assistant', prompts['batch_assistant'])
                else:
                    if 'assistant' in prompts:
                        self.prompt_manager.save_prompt(variant, 'assistant', prompts['assistant'])
                if 'user' in prompts:
                    self.prompt_manager.save_prompt(variant, 'user', prompts['user'])

                logger.info(f"Saved all prompts to files: {variant}")

            if hasattr(self, 'batch_hint_edit'):
                bh = self.batch_hint_edit.toPlainText().strip()
                self.prompt_manager.save_prompt(variant, 'batch_hint', bh)
                self.current_prompts_cache[variant]['batch_hint'] = bh

            self.show_message("Success", f"LLM instructions saved to disk for: {variant}")

        except Exception as e:
            logging.error(f"Failed to save LLM instructions: {e}")
            logging.error(traceback.format_exc())
            self.show_message("Save Error", f"Failed to save: {e}", QMessageBox.Icon.Critical)

    def reset_llm_instruction(self):
        variant = self._get_current_variant()
        if not variant:
            self.show_message("No File Loaded", "Please load a file first.", QMessageBox.Icon.Warning)
            return

        msg = QMessageBox.question(
            self,
            "Reset to Factory Defaults",
            f"Reset LLM prompts to factory defaults for: {variant}?\n\n"
            f"⚠️ This will discard your current changes in the editor.\n"
            f"Saved .txt files will NOT be deleted.\n\n"
            f"To make this permanent, click 'Save LLM Instruction' after reset.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if msg == QMessageBox.StandardButton.Yes:
            if variant in self.current_prompts_cache:
                del self.current_prompts_cache[variant]

            self.current_prompts_cache[variant] = {
                'system': self.prompt_manager.get_default_system_prompt(variant),
                'assistant': self.prompt_manager.get_default_assistant_prompt(variant),
                'batch_assistant': '',
                'user': self.prompt_manager.get_default_user_prompt(variant),
                'ollama': self.prompt_manager.get_default_ollama_prompt(variant),
                'json_payload': '',
                'batch_json_payload': self.prompt_manager.get_default_batch_json_payload_prompt(variant),
                'batch_hint': self.prompt_manager.get_default_batch_hint_prompt(),
            }

            self._json_payload_content = ''
            self.json_response_field_edit.setText('')
            self.app_settings['json_response_field'] = ''
            self.json_payload_checkbox.setChecked(False)

            logger.info(f"Reset prompts to factory defaults in cache: {variant}")
            self.update_llm_editor_content()

            self.show_message(
                "Reset Complete",
                f"Prompts reset to factory defaults: {variant}\n\n"
                f"💡 Changes are active now, but NOT saved to disk.\n"
                f"Click 'Save LLM Instruction' to make permanent."
            )

    def hard_reset_llm_instruction(self):
        variant = self._get_current_variant()
        if not variant:
            self.show_message("No File Loaded", "Please load a file first.", QMessageBox.Icon.Warning)
            return

        msg = QMessageBox.question(
            self,
            "Hard Reset",
            f"PERMANENTLY DELETE prompt files for: {variant}?\n\n"
            f"This will:\n"
            f"• Delete all .txt files for this variant\n"
            f"• Reset editor to factory defaults\n\n"
            f"⚠️ This action CANNOT be undone!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if msg == QMessageBox.StandardButton.Yes:
            deleted_count = self.prompt_manager.hard_reset(variant)

            if variant in self.current_prompts_cache:
                del self.current_prompts_cache[variant]

            self.current_prompts_cache[variant] = {
                'system': self.prompt_manager.get_default_system_prompt(variant),
                'assistant': self.prompt_manager.get_default_assistant_prompt(variant),
                'batch_assistant': '',
                'user': self.prompt_manager.get_default_user_prompt(variant),
                'ollama': self.prompt_manager.get_default_ollama_prompt(variant),
                'json_payload': '',
                'batch_json_payload': self.prompt_manager.get_default_batch_json_payload_prompt(variant),
                'batch_hint': self.prompt_manager.get_default_batch_hint_prompt(),
            }

            self._json_payload_content = ''
            self.json_response_field_edit.setText('')
            self.app_settings['json_response_field'] = ''
            self.json_payload_checkbox.setChecked(False)

            self.update_llm_editor_content()

            self.show_message(
                "Hard Reset Complete",
                f"Deleted {deleted_count} file(s).\n"
                f"Factory defaults restored: {variant}"
            )

    def save_app_settings(self):
        try:
            settings = self.app_settings.copy()
            settings["llm_choice"] = self.llm_choice_combo.currentText()
            settings["server_url"] = self.server_url_edit.text().strip()
            if settings["llm_choice"] == "Ollama":
                settings["ollama_model_name"] = self.ollama_model_edit.text()
            elif settings["llm_choice"] == "Openrouter":
                settings["openrouter_api_key"] = self.openrouter_api_key_edit.text()
                settings["openrouter_model_name"] = self.openrouter_model_edit.text()
            settings["deepl_free_api_key"] = self.deepl_free_api_key_edit.text()
            settings["deepl_pro_api_key"] = self.deepl_pro_api_key_edit.text()
            settings["use_inline_formatting"] = self.inline_formatting_checkbox.isChecked()
            settings["restore_paragraph_structure"] = self.restore_paragraph_checkbox.isChecked()
            settings["show_ps_in_ui"] = self.show_ps_in_ui_checkbox.isChecked()
            for obsolete in ("use_ps_markers", "restore_paragraph_epub", "restore_paragraph_txt"):
                settings.pop(obsolete, None)
            skip_inline_tags = {}
            for tag, checkbox in self.skip_inline_checkboxes.items():
                skip_inline_tags[tag] = checkbox.isChecked()
            settings["skip_inline_tags"] = skip_inline_tags
            mismatch_checks = {}
            for check_name, checkbox in self.mismatch_check_checkboxes.items():
                mismatch_checks[check_name] = checkbox.isChecked()
            settings["mismatch_checks"] = mismatch_checks
            settings["mismatch_thresholds"] = {
                "length_ratio_short":               round(self.length_ratio_short_spinbox.value(), 2),
                "length_ratio_medium":              round(self.length_ratio_medium_spinbox.value(), 2),
                "length_ratio_long":                round(self.length_ratio_long_spinbox.value(), 2),
                "length_ratio_too_short":           round(self.length_ratio_too_short_spinbox.value(), 2),
                "untranslated_ratio":               round(self.untranslated_ratio_spinbox.value(), 2),
                "position_shift_threshold":         round(self.position_shift_threshold_spinbox.value(), 2),
                "inline_position_shift_threshold":  round(self.inline_position_shift_threshold_spinbox.value(), 2),
            }
            model_name_input = self.alignment_model_edit.text().strip()
            if not model_name_input:
                model_name_input = "xlm-roberta-large"
            settings["alignment_settings"] = {
                "device":     self.alignment_device_combo.currentText(),
                "model_name": model_name_input,
            }
            logger.info(
                f"Saving settings - use_inline_formatting: {settings['use_inline_formatting']}, "
                f"restore_paragraph_structure: {settings['restore_paragraph_structure']}, "
                f"alignment_settings: {settings['alignment_settings']}"
            )
            AppSettingsManager.save_settings(settings)
            self.app_settings = settings
            self._initialize_components()
            self._refresh_alignment_status(model_name_input)
            QMessageBox.information(
                self, "Settings Saved", "Settings have been saved successfully."
            )
        except Exception as e:
            logging.error(f"Failed to save settings: {e}")
            QMessageBox.critical(self, "Save Error", f"Failed to save settings: {e}")

    def _on_download_alignment_model(self) -> None:
        model_name = self.alignment_model_edit.text().strip()
        if not model_name:
            model_name = "xlm-roberta-large"

        models_dir = self._get_models_dir()

        try:
            already = is_model_downloaded(model_name, models_dir)
        except Exception:
            already = False

        if already:
            local_path = get_local_model_path(model_name, models_dir)
            reply = QMessageBox.question(
                self,
                "Model already available locally",
                f"The model '{model_name}' is already downloaded at:\n"
                f"{local_path}\n\n"
                f"You don't need to download it again – it can be used immediately.\n\n"
                f"Do you still want to download it AGAIN (overwrite existing files)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.alignment_status_label.setText(
                    f"✓ Model '{model_name}' is available locally. Ready to use."
                )
                self.alignment_status_label.setStyleSheet(
                    "color: #44aa44; font-size: 10px;"
                )
                return

        self.alignment_status_label.setText(
            f"Downloading model '{model_name}'... (this may take a few minutes)"
        )
        self.alignment_status_label.setStyleSheet("color: #aaaa44; font-size: 10px;")
        QApplication.processEvents()

        def _download_worker():
            try:
                saved_path = _fa_download_model(
                    model_name=model_name,
                    models_dir=models_dir,
                )

                self._download_done_signal.emit(model_name, saved_path, '')
            except Exception as exc:
                self._download_done_signal.emit(model_name, '', str(exc))

        t = threading.Thread(target=_download_worker, daemon=True)
        t.start()
        logger.info(f"[app] Download of model '{model_name}' started in a thread.")

    def _on_download_done_slot(self, model_name: str, saved_path: str, error: str) -> None:
        self._on_download_done(model_name, saved_path, error if error else None)

    def _on_download_done(
        self,
        model_name: str,
        saved_path: str,
        error,
    ) -> None:
        if not hasattr(self, 'alignment_status_label'):
            return

        if error:
            self.alignment_status_label.setText(
                f"Error downloading model '{model_name}': {error}"
            )
            self.alignment_status_label.setStyleSheet(
                "color: #cc4444; font-size: 10px;"
            )
            logger.error(f"[app] Error downloading model '{model_name}': {error}")
            QMessageBox.critical(
                self,
                "Model download error",
                f"Failed to download model '{model_name}':\n\n{error}\n\n"
                "Check your internet connection and make sure the "
                "'transformers' package is installed.",
            )
        else:
            logger.info(f"[app] Model '{model_name}' downloaded to: {saved_path}")

            self._refresh_alignment_status(model_name)

    def update_model_name_visibility(self, llm_choice):
        is_ollama = llm_choice == "Ollama"
        is_openrouter = llm_choice == "Openrouter"
        is_local = llm_choice == "LM Studio"

        self.server_url_label.setVisible(is_local)
        self.server_url_edit.setVisible(is_local)
        self.ollama_model_label.setVisible(is_ollama)
        self.ollama_model_edit.setVisible(is_ollama)
        self.openrouter_api_key_label.setVisible(is_openrouter)
        self.openrouter_api_key_edit.setVisible(is_openrouter)
        self.openrouter_model_label.setVisible(is_openrouter)
        self.openrouter_model_edit.setVisible(is_openrouter)
        self.openrouter_free_warning.setVisible(is_openrouter)

    def _toggle_processing_mode(self, checked):
        new_mode = 'inline' if checked else 'legacy'

        self.app_settings['use_inline_formatting'] = checked

        if self.file_type != "epub" or not self.original_file_path or not self.paragraphs:
            mode_emoji = "🔧" if new_mode == "inline" else "📦"
            self.statusBar().showMessage(
                f"{mode_emoji} EPUB processing mode set to {new_mode.upper()} "
                f"– will apply when an EPUB file is loaded",
                4000
            )
            logger.info(f"Processing mode changed to {new_mode} (no EPUB loaded – saved to settings only)")
            self._initialize_components()
            if self.llm_editor_container.isVisible():
                self.update_llm_editor_content()
            return

        translated_count = sum(1 for p in self.paragraphs if p.get('is_translated', False))

        if self.paragraphs:
            loaded_mode = self.paragraphs[0].get('processing_mode', 'inline')
        else:
            loaded_mode = 'inline'

        if new_mode == loaded_mode:
            logger.debug(f"Checkbox changed but matches loaded mode: {new_mode}")
            self._update_status_after_file_load()
            if self.llm_editor_container.isVisible():
                self.update_llm_editor_content()
            return

        mode_emoji_old = "📦" if loaded_mode == "legacy" else "🔧"
        mode_emoji_new = "📦" if new_mode == "legacy" else "🔧"

        message_parts = [
            f"Change processing mode?\n",
            f"\n{mode_emoji_old} Current: {loaded_mode.upper()}",
            f"\n{mode_emoji_new} New: {new_mode.upper()}",
            f"\n\nFile: {os.path.basename(self.original_file_path)}",
            f"\nFragments: {len(self.paragraphs)} total"
        ]

        if translated_count > 0:
            message_parts.append(f", {translated_count} translated")
            message_parts.append("\n\n⚠️ WARNING: Reloading will reset all translations!")

        message_parts.append("\n\nTo use the new mode, the file must be reloaded.")
        message_parts.append("\nWhat would you like to do?")

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Question)
        msg_box.setWindowTitle("Change Processing Mode")
        msg_box.setText("".join(message_parts))

        btn_reload = msg_box.addButton("Reload Now", QMessageBox.ButtonRole.AcceptRole)
        btn_later = msg_box.addButton("Change Later", QMessageBox.ButtonRole.RejectRole)
        btn_cancel = msg_box.addButton("Cancel", QMessageBox.ButtonRole.DestructiveRole)

        msg_box.setDefaultButton(btn_reload)
        msg_box.exec()

        clicked = msg_box.clickedButton()

        if clicked == btn_cancel:
            logger.info("Processing mode change cancelled by user")
            self.inline_formatting_checkbox.blockSignals(True)
            self.inline_formatting_checkbox.setChecked(loaded_mode == 'inline')
            self.inline_formatting_checkbox.blockSignals(False)
            self.app_settings['use_inline_formatting'] = (loaded_mode == 'inline')
            self.alignment_section_widget.setVisible(loaded_mode != 'inline')
            if self.llm_editor_container.isVisible():
                self.update_llm_editor_content()
            return

        elif clicked == btn_reload:
            logger.info(f"Changing processing mode and reloading: {loaded_mode} → {new_mode}")
            self._reload_current_file(self.original_file_path)
            mode_emoji = "🔧" if new_mode == "inline" else "📦"
            self.statusBar().showMessage(
                f"✓ File reloaded with {mode_emoji} {new_mode.upper()} mode",
                5000
            )
            return

        else:
            logger.info(f"Processing mode changed to {new_mode} – file reload pending")
            self.statusBar().showMessage(
                f"⚠️ File still loaded as {loaded_mode.upper()}, will translate with "
                f"{new_mode.upper()} prompts – Reload file for full consistency",
                0
            )
            self.update_file_label()
            if self.llm_editor_container.isVisible():
                self.update_llm_editor_content()

    def _reload_current_file(self, file_path):
        if not file_path or not os.path.exists(file_path):
            self.show_message(
                "Reload Error",
                f"Cannot reload file: {file_path}",
                QMessageBox.Icon.Critical
            )
            return

        logger.info(f"Reloading file: {file_path}")

        self.paragraphs = []
        self.translation_queue = []
        self.current_translation_idx = None
        self._preview_show_original_ids.clear()

        self.list_widget.clear()
        self.original_text_view.clear()
        self.translated_text_view.textChanged.disconnect(self.update_translation_from_edit)
        self.translated_text_view.clear()
        self.translated_text_view.textChanged.connect(self.update_translation_from_edit)

        settings_for_processor = self.app_settings.copy()
        settings_for_processor['use_inline_formatting'] = self.inline_formatting_checkbox.isChecked()

        try:
            self.file_processor = FileProcessorFactory.create_processor(
                self.file_type,
                settings_for_processor
            )

            result = self.file_processor.load(file_path)

            if isinstance(result, tuple):
                self.paragraphs = result[0]
            elif isinstance(result, list):
                self.paragraphs = result
            else:
                raise TypeError(f"Unexpected return type: {type(result)}")

            if not self.paragraphs:
                raise ValueError("No paragraphs loaded")

            if self.paragraphs and not isinstance(self.paragraphs[0], dict):
                raise TypeError(f"Invalid paragraph format: {type(self.paragraphs[0])}")

            self._initialize_components()

            self.populate_list()
            self.update_file_label()

            self.update_llm_editor_content()

            mode_str = 'inline' if self.inline_formatting_checkbox.isChecked() else 'legacy'
            logger.info(f"File reloaded successfully with mode: {mode_str}")

        except Exception as e:
            logging.error(f"Failed to reload file: {e}")
            traceback.print_exc()
            self.show_message(
                "Reload Error",
                f"Failed to reload file:\n{e}",
                QMessageBox.Icon.Critical
            )

    def _refresh_alignment_status(self, model_name=None) -> None:
        if not hasattr(self, 'alignment_status_label'):
            return

        if not isinstance(model_name, str) or not model_name.strip():
            model_name = ''

        if not model_name:
            if hasattr(self, 'alignment_model_edit'):
                model_name = self.alignment_model_edit.text().strip()
        if not model_name:
            model_name = "xlm-roberta-large"

        models_dir = self._get_models_dir()

        try:
            downloaded = is_model_downloaded(model_name, models_dir)
        except Exception:
            downloaded = False

        if downloaded:
            local_path = get_local_model_path(model_name, models_dir)
            self.alignment_status_label.setText(
                f"✓ Model available locally: {local_path}"
            )
            self.alignment_status_label.setStyleSheet(
                "color: #44aa44; font-size: 10px;"
            )
        else:
            self.alignment_status_label.setText(
                f"Model '{model_name}' not found locally. "
                "Click 'Download' to fetch it (one-time)."
            )
            self.alignment_status_label.setStyleSheet(
                "color: #aa7722; font-size: 10px;"
            )

    def _on_mismatch_check_toggled(self):
        if not self.mismatch_checker:
            return
        for check_name, cb in self.mismatch_check_checkboxes.items():
            self.mismatch_checker.checks[check_name] = cb.isChecked()
        self.mismatch_checker.thresholds["length_ratio_short"]               = self.length_ratio_short_spinbox.value()
        self.mismatch_checker.thresholds["length_ratio_medium"]              = self.length_ratio_medium_spinbox.value()
        self.mismatch_checker.thresholds["length_ratio_long"]                = self.length_ratio_long_spinbox.value()
        self.mismatch_checker.thresholds["length_ratio_too_short"]           = self.length_ratio_too_short_spinbox.value()
        self.mismatch_checker.thresholds["untranslated_ratio"]               = self.untranslated_ratio_spinbox.value()
        self.mismatch_checker.thresholds["position_shift_threshold"]         = self.position_shift_threshold_spinbox.value()
        self.mismatch_checker.thresholds["inline_position_shift_threshold"]  = self.inline_position_shift_threshold_spinbox.value()
        if not hasattr(self, 'paragraphs') or not self.paragraphs:
            return
        changed = False
        for para in self.paragraphs:
            if not para.get('is_translated'):
                continue
            if para.get('ignore_mismatch') or para.get('force_mismatch'):
                continue
            has_mismatch, mismatch_flags = self.mismatch_checker.check_mismatch(para)
            old_mismatch  = para.get('has_mismatch', False)
            old_flags     = para.get('mismatch_flags', {})
            para['has_mismatch']   = has_mismatch
            para['mismatch_flags'] = mismatch_flags
            if old_mismatch != has_mismatch or old_flags != mismatch_flags:
                changed = True
        if changed:
            self.populate_list()

    def _get_mode_description(self, mode: str) -> str:
        if mode == 'inline':
            return "INLINE (full formatting with <p_XX> placeholders)"
        else:
            return "LEGACY (simple reserve elements only)"

    def show_message(self, title, message, icon=QMessageBox.Icon.Information):
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setText(message)
        msg_box.setWindowTitle(title)
        msg_box.exec()

    def _get_current_variant(self):
        if not self.file_type:
            return None

        if self.file_type == "txt":
            return "txt"
        elif self.file_type == "srt":
            return "srt"
        elif self.file_type == "epub":
            use_inline = self.inline_formatting_checkbox.isChecked()
            variant = "epub_inline" if use_inline else "epub_legacy"
            logger.debug(f"Current variant: {variant} (checkbox: {use_inline})")
            return variant

        return None

    def _get_context(self, idx, before=True, count=3):
        context = []

        if before:
            start = max(0, idx - count)
            for i in range(start, idx):
                context.append(self.paragraphs[i])
        else:
            end = min(len(self.paragraphs), idx + count + 1)
            for i in range(idx + 1, end):
                context.append(self.paragraphs[i])

        return context

    def _describe_first_char(self, text):
        text = re.sub(r'</?(?:id|p)_\d{2}>|<nt_\d{2}/>', '', text)
        text = text.lstrip()
        if not text:
            return "empty text"

        has_quote = any(ch in ALL_QUOTES_CHARS for ch in text[:1])

        text_no_quotes = text.lstrip()
        while text_no_quotes and text_no_quotes[0] in ALL_QUOTES_CHARS:
            text_no_quotes = text_no_quotes[1:].lstrip()

        if not text_no_quotes:
            return "quote only"

        c = text_no_quotes[0]

        if c == '<':
            char_type = "HTML/placeholder tag"
        elif c.isdigit():
            char_type = f"digit ({c})"
        elif c.isupper():
            char_type = f"uppercase letter ({c})"
        elif c.islower():
            char_type = f"lowercase letter ({c})"
        else:
            char_type = f"special char ({c})"

        return f"quote + {char_type}" if has_quote else char_type

    def _describe_last_char(self, text):
        text = re.sub(r'</?(?:id|p)_\d{2}>|<nt_\d{2}/>', '', text)
        text = text.rstrip()
        if not text:
            return {"type": "empty text"}

        has_double_quote = any(ch in DOUBLE_QUOTES_CHARS for ch in text[-1:])

        if has_double_quote:
            text_no_double = text.rstrip()
            while text_no_double and text_no_double[-1] in DOUBLE_QUOTES_CHARS:
                text_no_double = text_no_double[:-1].rstrip()
        else:
            text_no_double = text

        if not text_no_double:
            return {"type": "quote only"}

        has_apostrophe = any(ch in SINGLE_QUOTES_CHARS for ch in text_no_double[-1:])

        if has_apostrophe:
            text_no_quotes = text_no_double.rstrip()
            while text_no_quotes and text_no_quotes[-1] in SINGLE_QUOTES_CHARS:
                text_no_quotes = text_no_quotes[:-1].rstrip()
        else:
            text_no_quotes = text_no_double

        if not text_no_quotes:
            return {"type": "quote only"}

        last_char = text_no_quotes[-1]

        char_map = {
            '.': "period (.)", ',': "comma (,)", '!': "exclamation (!)",
            '?': "question (?)", '…': "ellipsis (…)", ';': "semicolon (;)",
            ':': "colon (:)", '-': "dash (-)", '–': "en-dash (–)", '—': "em-dash (—)",
        }

        if last_char in char_map:
            char_type = char_map[last_char]
        elif last_char == '>':
            char_type = "placeholder tag"
        elif last_char.isdigit():
            char_type = f"digit ({last_char})"
        elif last_char.isalpha():
            char_type = f"letter ({last_char})"
        else:
            char_type = f"other ({last_char})"

        suffix = ""
        if has_apostrophe:
            suffix = " + apostrophe"
        if has_double_quote:
            suffix += " + closing quote"

        return {"type": char_type + suffix if suffix else char_type}

    def split_translated_text_into_lines(self, translated_text, original_para):
        ALL_QUOTES_CLASS = r'["\'\u2018\u2019\u201C\u201D\u201E\u201F\u00AB\u00BB\u2039\u203A\u201A\u201B\u2032\u2033\u301D\u301E\u301F\uFF02]'
        text = translated_text.strip()
        orig_full = original_para.get('original_text', '').strip()

        if re.match(f"^{ALL_QUOTES_CLASS}", text) and not re.match(f"^{ALL_QUOTES_CLASS}", orig_full):
            text = text[1:].lstrip()

        if re.search(f"{ALL_QUOTES_CLASS}$", text) and not re.search(f"{ALL_QUOTES_CLASS}$", orig_full):
            text = text[:-1].rstrip()

        orig_clean_lines = original_para.get('original_clean_lines', [])
        if not orig_clean_lines:
            return [text]

        orig_line_count = len(orig_clean_lines)

        if '\n' in text:
            trans_lines = [l.strip() for l in text.split('\n') if l.strip()]
            if len(trans_lines) == orig_line_count:
                return trans_lines

        if orig_line_count == 2:
            return self.split_into_two_lines_balanced(text, original_para)

        elif orig_line_count >= 3:
            return self.split_into_multiple_lines_proportionally(text, orig_line_count, original_para)

        else:
            return [text]

    def split_into_two_lines_balanced(self, text, original_paragraph):
        text = text.strip()

        if len(text) < 20:
            return [text]

        original_lines = original_paragraph.get('original_clean_lines', [])
        if len(original_lines) == 2:
            orig_len1 = len(original_lines[0])
            orig_len2 = len(original_lines[1])

            total_orig = orig_len1 + orig_len2
            if total_orig > 0:
                ratio1 = orig_len1 / total_orig
                target_len1 = int(len(text) * ratio1)

                split_point = self.find_good_split_point(text, target_len1)

                line1 = text[:split_point].strip()
                line2 = text[split_point:].strip()

                if not line2:
                    return [text]

                if len(line1) > len(line2):
                    words1 = line1.split()
                    if len(words1) > 1:
                        last_word = words1[-1]
                        line1 = ' '.join(words1[:-1])
                        line2 = last_word + ' ' + line2

                return [line1, line2]

        return self.split_into_two_lines_by_natural_break(text)

    def find_good_split_point(self, text, target_position):
        search_range = 15
        start = max(0, target_position - search_range)
        end = min(len(text), target_position + search_range)

        split_chars = ['. ', '! ', '? ', '; ', ', ', ' - ', ' — ', ' ']

        for char in split_chars:
            pos = text.rfind(char, start, target_position + len(char))
            if pos != -1:
                return pos + len(char) - 1 if char.endswith(' ') else pos + len(char)

            pos = text.find(char, target_position, end)
            if pos != -1:
                return pos + len(char) - 1 if char.endswith(' ') else pos + len(char)

        return target_position

    def split_into_two_lines_by_natural_break(self, text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        if len(sentences) >= 2:
            line1 = sentences[0]
            line2 = ' '.join(sentences[1:])

            if len(line1) < 15 and len(sentences) > 1:
                line1 = sentences[0] + ' ' + sentences[1]
                line2 = ' '.join(sentences[2:]) if len(sentences) > 2 else ""

            if line2 and len(line1) > len(line2):
                words1 = line1.split()
                if len(words1) > 1:
                    last_word = words1[-1]
                    line1_new = ' '.join(words1[:-1])
                    line2_new = last_word + ' ' + line2

                    if len(line1_new) <= len(line2_new):
                        line1 = line1_new
                        line2 = line2_new

            if line2:
                return [line1, line2]

        for separator in ['; ', ' - ', ' — ', ', ']:
            if separator in text:
                parts = text.split(separator, 1)
                if len(parts) == 2:
                    line1 = parts[0] + (separator.strip() if separator.strip() else '')
                    line2 = parts[1]

                    if len(line1) >= 10 and len(line2) >= 10:
                        if len(line1) > len(line2):
                            words1 = line1.split()
                            if len(words1) > 2:
                                last_word = words1[-1]
                                line1 = ' '.join(words1[:-1])
                                line2 = last_word + ' ' + line2

                        return [line1, line2]

        middle = len(text) // 2
        split_point = None
        for i in range(middle - 5, max(0, middle - 30), -1):
            if text[i] == ' ':
                split_point = i + 1
                break

        if split_point is None:
            for i in range(middle, min(len(text), middle + 20)):
                if text[i] == ' ':
                    split_point = i + 1
                    break

        if split_point is None:
            split_point = middle

        line1 = text[:split_point].strip()
        line2 = text[split_point:].strip()

        if not line2 or len(line1) < 10:
            return [text]

        return [line1, line2]

    def split_into_multiple_lines_proportionally(self, text, line_count, original_paragraph):
        original_lines = original_paragraph.get('original_clean_lines', [])
        if len(original_lines) == line_count:
            original_lengths = [len(line) for line in original_lines]
            total_original_length = sum(original_lengths)

            if total_original_length > 0:
                target_lengths = [int(len(text) * (ol / total_original_length)) for ol in original_lengths]

                diff = len(text) - sum(target_lengths)
                if diff != 0:
                    max_idx = target_lengths.index(max(target_lengths))
                    target_lengths[max_idx] += diff

                result_lines = []
                current_pos = 0

                for i, target_len in enumerate(target_lengths):
                    if i == len(target_lengths) - 1:
                        line = text[current_pos:].strip()
                    else:
                        end_pos = current_pos + target_len

                        if end_pos < len(text):
                            for j in range(end_pos, min(len(text), end_pos + 20)):
                                if text[j] in ' ,.;:!?-–—':
                                    end_pos = j + 1 if text[j] in ' ,.;:!?' else j
                                    break
                            else:
                                for j in range(end_pos, max(current_pos, end_pos - 20), -1):
                                    if text[j] in ' ,.;:!?-–—':
                                        end_pos = j + 1 if text[j] in ' ,.;:!?' else j
                                        break

                        line = text[current_pos:end_pos].strip()
                        current_pos = end_pos

                    result_lines.append(line)

                if len(result_lines) == line_count:
                    if len(result_lines) >= 2:
                        first_len = len(result_lines[0])
                        second_len = len(result_lines[1])
                        if first_len > second_len:
                            words = result_lines[0].split()
                            if len(words) > 1:
                                last_word = words[-1]
                                result_lines[0] = ' '.join(words[:-1])
                                result_lines[1] = last_word + ' ' + result_lines[1]

                    return result_lines

        return self.split_text_evenly(text, line_count)

    def split_text_evenly(self, text, line_count):
        words = text.split()
        if len(words) < line_count:
            return [' '.join(words[i:i+1]) for i in range(0, len(words), 1)]

        words_per_line = len(words) // line_count
        remainder = len(words) % line_count

        lines = []
        current_index = 0

        for i in range(line_count):
            current_words = words_per_line + (1 if i < remainder else 0)
            line_words = words[current_index:current_index + current_words]
            lines.append(' '.join(line_words))
            current_index += current_words

        if len(lines) >= 2 and len(lines[0]) > len(lines[1]):
            words1 = lines[0].split()
            words2 = lines[1].split()

            if len(words1) > 1:
                last_word = words1[-1]
                lines[0] = ' '.join(words1[:-1])
                lines[1] = last_word + ' ' + ' '.join(words2)

        return lines

    def closeEvent(self, event):
        self.cancel_translation()

        if hasattr(self, 'epub_creator') and self.epub_creator:
            if self.epub_creator.isRunning():
                self.epub_creator.terminate()
                self.epub_creator.wait(5000)

        if hasattr(self, 'srt_creator') and self.srt_creator:
            if self.srt_creator.isRunning():
                self.srt_creator.terminate()
                self.srt_creator.wait(5000)

        if hasattr(self, 'txt_creator') and self.txt_creator:
            if self.txt_creator.isRunning():
                self.txt_creator.terminate()
                self.txt_creator.wait(5000)

        super().closeEvent(event)

if __name__ == '__main__':
    if sys.platform == 'win32':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('Mubumbutu.EPUBTranslator')

    app = QApplication(sys.argv)
    ex = TranslatorApp()
    ex.show()
    sys.exit(app.exec())
