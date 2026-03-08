# epub_preview.py
import copy
import logging
import os
import re

from lxml import etree
from PyQt6.QtCore import pyqtSignal, Qt, QTimer, QUrl
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

XHTML_NS = 'http://www.w3.org/1999/xhtml'

_DARK_MODE_CSS = """
html, body {
    background-color: #1a1a1a !important;
    color: #cccccc !important;
}
* {
    border-color: #444 !important;
}
a { color: #7ab4f5 !important; }
"""

_HIGHLIGHT_CSS = """
[data-epub-preview-show-original] {
    outline: 2px dashed #e09c2a !important;
    outline-offset: 2px !important;
    background-color: rgba(224, 156, 42, 0.10) !important;
    border-radius: 2px !important;
}
"""

_SELECTED_HIGHLIGHT_CSS = """
[data-epub-preview-selected] {
    outline: 2px solid #4a90d9 !important;
    outline-offset: 2px !important;
    background-color: rgba(74, 144, 217, 0.12) !important;
    border-radius: 2px !important;
}
"""

_NO_SCROLLBAR_CSS = """
::-webkit-scrollbar { display: none !important; }
body { -ms-overflow-style: none; scrollbar-width: none; }
"""

_PREVIEW_SCROLLBAR_CSS = """
::-webkit-scrollbar { width: 8px; background: #1e1e1e; }
::-webkit-scrollbar-track { background: #1e1e1e; }
::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 4px; min-height: 20px; }
::-webkit-scrollbar-thumb:hover { background: #4a4a4a; }
::-webkit-scrollbar-button { display: none; }
"""

_SCROLL_TO_SEL_JS = """
(function(){
    var el = document.querySelector('[data-epub-preview-selected]');
    if (el) { el.scrollIntoView({behavior:'smooth', block:'center'}); }
})();
"""

_INTERACTION_JS = """
(function(){
    document.addEventListener('click', function(e){
        var node = e.target;
        while (node && node !== document.body) {
            if (node.id) {
                window.location.href = 'epub-preview://select/' + encodeURIComponent(node.id);
                e.preventDefault();
                e.stopPropagation();
                return;
            }
            node = node.parentElement;
        }
    }, true);

    document.addEventListener('contextmenu', function(e){
        var node = e.target;
        while (node && node !== document.body) {
            if (node.id) {
                window.location.href = 'epub-preview://contextmenu/' + encodeURIComponent(node.id);
                e.preventDefault();
                e.stopPropagation();
                return;
            }
            node = node.parentElement;
        }
    }, true);
})();
"""

_DRAG_SCROLL_JS = """
(function(){
    var isDragging = false;
    var startY = 0;
    var startScrollY = 0;
    var dragMoved = false;
    var DRAG_THRESHOLD = 5;

    document.documentElement.style.userSelect = 'none';
    document.documentElement.style.webkitUserSelect = 'none';

    document.addEventListener('mousedown', function(e){
        if (e.button === 0) {
            isDragging = true;
            dragMoved = false;
            startY = e.clientY;
            startScrollY = window.scrollY;
        }
    }, true);

    document.addEventListener('mousemove', function(e){
        if (!isDragging) return;
        var delta = e.clientY - startY;
        if (!dragMoved && Math.abs(delta) > DRAG_THRESHOLD) {
            dragMoved = true;
        }
        if (dragMoved) {
            window.scrollTo(0, startScrollY - delta);
            e.preventDefault();
            e.stopPropagation();
        }
    }, true);

    document.addEventListener('mouseup', function(){
        isDragging = false;
    }, true);

    document.addEventListener('mouseleave', function(){
        isDragging = false;
    });

    document.addEventListener('click', function(e){
        if (dragMoved) {
            e.preventDefault();
            e.stopPropagation();
            dragMoved = false;
            return;
        }
        var node = e.target;
        while (node && node.tagName) {
            if (node.tagName.toLowerCase() === 'a') {
                var href = node.getAttribute('href') || '';
                e.preventDefault();
                e.stopPropagation();
                if (href.startsWith('#')) {
                    var target = document.getElementById(href.slice(1));
                    if (target) target.scrollIntoView({behavior: 'smooth', block: 'start'});
                }
                return;
            }
            node = node.parentElement;
        }
    }, true);

    document.addEventListener('contextmenu', function(e){
        e.preventDefault();
        e.stopPropagation();
    }, true);
})();
"""


class EPUBPreviewToolbar(QWidget):
    """
    Toolbar displayed above the EPUB preview web view.

    Signals
    -------
    refresh_requested()
        Emitted when the user clicks the Refresh button.
    chapter_changed(str)
        Emitted when the user navigates to a different chapter; carries the
        target item_href string.
    dark_mode_toggled(bool)
        Emitted when the user toggles the dark/light mode button; carries
        the new dark-mode state (True = dark).
    """

    refresh_requested = pyqtSignal()
    chapter_changed   = pyqtSignal(str)
    dark_mode_toggled = pyqtSignal(bool)
    reader_requested  = pyqtSignal()

    _TOOLBAR_STYLE = """
        QWidget#EPUBPreviewToolbar {
            background-color: #1e1e1e;
            border-bottom: 1px solid #2e2e2e;
        }
        QPushButton {
            background-color: #2a2a2a;
            color: #cccccc;
            border: 1px solid #3a3a3a;
            border-radius: 3px;
            padding: 2px 8px;
            font-size: 12px;
            min-width: 24px;
        }
        QPushButton:hover  { background-color: #353535; color: #ffffff; }
        QPushButton:pressed { background-color: #1a1a1a; }
        QPushButton:disabled { color: #555555; border-color: #2a2a2a; }
        QLabel {
            color: #888888;
            font-size: 12px;
            padding: 0 6px;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('EPUBPreviewToolbar')
        self.setStyleSheet(self._TOOLBAR_STYLE)
        self.setFixedHeight(32)

        self._chapters: list = []
        self._current_index: int = -1
        self._dark_mode: bool = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(4)

        self._btn_refresh = QPushButton("⟳ Refresh")
        self._btn_refresh.setToolTip("Regenerate preview for current fragment")
        self._btn_refresh.clicked.connect(self.refresh_requested)
        layout.addWidget(self._btn_refresh)

        self._sep1 = QLabel("|")
        layout.addWidget(self._sep1)

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setToolTip("Previous chapter")
        self._btn_prev.setFixedWidth(28)
        self._btn_prev.clicked.connect(self._go_prev)
        layout.addWidget(self._btn_prev)

        self._chapter_label = QLabel("—")
        self._chapter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chapter_label.setMinimumWidth(160)
        layout.addWidget(self._chapter_label)

        self._btn_next = QPushButton("▶")
        self._btn_next.setToolTip("Next chapter")
        self._btn_next.setFixedWidth(28)
        self._btn_next.clicked.connect(self._go_next)
        layout.addWidget(self._btn_next)

        self._sep2 = QLabel("|")
        layout.addWidget(self._sep2)

        self._btn_dark = QPushButton("🌙 Dark")
        self._btn_dark.setToolTip("Toggle dark / light background")
        self._btn_dark.setCheckable(True)
        self._btn_dark.clicked.connect(self._toggle_dark)
        layout.addWidget(self._btn_dark)

        self._sep3 = QLabel("|")
        layout.addWidget(self._sep3)

        self._btn_reader = QPushButton("📖 Reader")
        self._btn_reader.setToolTip("Open full-screen reading mode  (live translations)")
        self._btn_reader.clicked.connect(self.reader_requested)
        layout.addWidget(self._btn_reader)

        layout.addStretch()

        self._update_nav_buttons()

    def set_chapters(self, chapters: list, current_href: str):
        """
        Populate chapter list and mark the active chapter.

        Parameters
        ----------
        chapters : list[str]
            Ordered list of item_href strings representing all chapters.
        current_href : str
            The item_href of the currently displayed chapter.
        """
        self._chapters = list(chapters)
        try:
            self._current_index = self._chapters.index(current_href)
        except ValueError:
            self._current_index = -1
        self._update_chapter_label()
        self._update_nav_buttons()

    def set_dark_mode(self, dark: bool):
        """Sync button state with external dark-mode flag."""
        self._dark_mode = dark
        self._btn_dark.setChecked(dark)
        self._btn_dark.setText("☀ Light" if dark else "🌙 Dark")

    def current_href(self) -> str:
        """Return item_href for the currently displayed chapter, or ''."""
        if 0 <= self._current_index < len(self._chapters):
            return self._chapters[self._current_index]
        return ''

    def _go_prev(self):
        if self._current_index > 0:
            self._current_index -= 1
            self._update_chapter_label()
            self._update_nav_buttons()
            self.chapter_changed.emit(self._chapters[self._current_index])

    def _go_next(self):
        if self._current_index < len(self._chapters) - 1:
            self._current_index += 1
            self._update_chapter_label()
            self._update_nav_buttons()
            self.chapter_changed.emit(self._chapters[self._current_index])

    def _toggle_dark(self, checked: bool):
        self._dark_mode = checked
        self._btn_dark.setText("☀ Light" if checked else "🌙 Dark")
        self.dark_mode_toggled.emit(checked)

    def _update_chapter_label(self):
        if not self._chapters or self._current_index < 0:
            self._chapter_label.setText("—")
            return
        total = len(self._chapters)
        idx   = self._current_index + 1
        href  = self._chapters[self._current_index]
        short = os.path.basename(href)
        self._chapter_label.setText(f"{short}  ({idx}/{total})")
        self._chapter_label.setToolTip(href)

    def _update_nav_buttons(self):
        has_list = bool(self._chapters) and self._current_index >= 0
        self._btn_prev.setEnabled(has_list and self._current_index > 0)
        self._btn_next.setEnabled(has_list and self._current_index < len(self._chapters) - 1)


_READER_PAGE_CSS = """
html, body {
    margin: 0 auto;
    padding: 32px 48px;
    max-width: 820px;
    font-size: 1.15em;
    line-height: 1.75;
    font-family: Georgia, "Times New Roman", serif;
    word-spacing: 0.05em;
}
img { max-width: 100%; height: auto; }
"""

_READER_PAGE_DARK_CSS = """
html, body {
    background-color: #1c1c1c !important;
    color: #d4d0c8 !important;
}
a { color: #7ab4f5 !important; }
"""

_READER_PAGE_SEPIA_CSS = """
html, body {
    background-color: #e8dcc8 !important;
    color: #2c1a0e !important;
}
div, article, section, main, p, span, li, td, th, blockquote, header, footer, nav {
    background-color: #e8dcc8 !important;
}
* { background-color: inherit; }
a { color: #7a4f1f !important; }
"""

_SEPIA_FORCE_JS = """
(function(){
    var BG = '#e8dcc8';
    var FG = '#2c1a0e';
    function isNearWhite(c){
        var m = c.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
        if(!m) return false;
        var r=+m[1],g=+m[2],b=+m[3];
        return r>210 && g>200 && b>185;
    }
    document.documentElement.style.setProperty('background-color', BG, 'important');
    document.body.style.setProperty('background-color', BG, 'important');
    document.body.style.setProperty('color', FG, 'important');
    var all = document.querySelectorAll('*');
    for(var i=0;i<all.length;i++){
        var cs = window.getComputedStyle(all[i]);
        if(isNearWhite(cs.backgroundColor)){
            all[i].style.setProperty('background-color', BG, 'important');
        }
        if(isNearWhite(cs.color)){
            all[i].style.setProperty('color', FG, 'important');
        }
    }
})();
"""

_READER_ARROW_STYLE_LIGHT = """
    QPushButton {
        background-color: rgba(30, 30, 30, 0.55);
        color: #cccccc;
        border: none;
        border-radius: 6px;
        font-size: 28px;
        min-width: 48px;
        max-width: 48px;
        padding: 0;
    }
    QPushButton:hover  { background-color: rgba(74, 144, 217, 0.75); color: #ffffff; }
    QPushButton:pressed { background-color: rgba(40, 100, 180, 0.90); }
    QPushButton:disabled { color: rgba(100, 100, 100, 0.4);
                           background-color: rgba(20, 20, 20, 0.25); }
"""

_READER_ARROW_STYLE_SEPIA = """
    QPushButton {
        background-color: #e8dcc8;
        color: #7a4f1f;
        border: none;
        border-radius: 6px;
        font-size: 28px;
        min-width: 48px;
        max-width: 48px;
        padding: 0;
    }
    QPushButton:hover  { background-color: #f2ead9; color: #2c1a0e; }
    QPushButton:pressed { background-color: #ddd0b8; }
    QPushButton:disabled { color: rgba(44, 26, 14, 0.25);
                           background-color: #e8dcc8; }
"""

_READER_ARROW_STYLE_DARK = """
    QPushButton {
        background-color: rgba(30, 30, 30, 0.55);
        color: #aaaaaa;
        border: none;
        border-radius: 6px;
        font-size: 28px;
        min-width: 48px;
        max-width: 48px;
        padding: 0;
    }
    QPushButton:hover  { background-color: rgba(80, 80, 80, 0.70); color: #d4d0c8; }
    QPushButton:pressed { background-color: rgba(55, 55, 55, 0.90); }
    QPushButton:disabled { color: rgba(100, 100, 100, 0.4);
                           background-color: rgba(20, 20, 20, 0.25); }
"""

_READER_TOPBAR_STYLE = """
    QWidget#ReaderTopBar {
        background-color: #141414;
        border-bottom: 1px solid #2a2a2a;
    }
    QPushButton {
        background-color: #252525;
        color: #cccccc;
        border: 1px solid #383838;
        border-radius: 3px;
        padding: 2px 10px;
        font-size: 12px;
        min-width: 28px;
    }
    QPushButton:hover  { background-color: #303030; color: #ffffff; }
    QPushButton:pressed { background-color: #181818; }
    QPushButton:disabled { color: #4a4a4a; border-color: #252525; }
    QLabel {
        color: #888888;
        font-size: 12px;
        padding: 0 8px;
    }
    QLabel#ChapterLabel {
        color: #aaaaaa;
        font-size: 13px;
        font-weight: bold;
    }
"""

_READER_STATUS_STYLE = """
    QStatusBar {
        background-color: #141414;
        color: #666666;
        font-size: 11px;
        border-top: 1px solid #2a2a2a;
    }
"""


class _ReaderWebPage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        pass


class EPUBReaderWindow(QMainWindow):
    """
    Full-screen EPUB reader window.

    Opens modeless alongside the main application so live translation
    updates can be pushed in without interrupting the reading experience.

    Public API
    ----------
    refresh_chapter(book, paragraphs, selected_para, dark_mode)
        Re-render the currently displayed chapter with fresh translation
        data.  Call this from app.py whenever a fragment is translated
        while the reader is open.
    navigate_to_para(book, paragraphs, para_index, dark_mode)
        Jump to the chapter that contains paragraph *para_index* and
        scroll to that element.
    """

    closed = pyqtSignal()

    def __init__(self, book, paragraphs, selected_para,
                 dark_mode: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EPUB Reader")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._engine      = EPUBPreviewEngine()
        self._book        = book
        self._paragraphs  = paragraphs
        self._dark_mode   = dark_mode
        self._sepia_mode  = False
        self._chapters    = EPUBPreviewEngine.get_chapter_list(paragraphs)
        self._current_href = selected_para.get('item_href', '')
        try:
            self._chapter_idx = self._chapters.index(self._current_href)
        except ValueError:
            self._chapter_idx = 0

        self._build_ui()
        self._apply_shortcuts()

        self.showFullScreen()
        QTimer.singleShot(80, lambda: self._load_chapter(self._current_href, selected_para))

    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet("background-color: #1c1c1c;")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_topbar())

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self._btn_left = QPushButton("‹")
        self._btn_left.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._btn_left.setToolTip("Previous chapter  [←]")
        self._btn_left.clicked.connect(self._go_prev)
        content_layout.addWidget(self._btn_left)

        self._web = QWebEngineView()
        web_page = _ReaderWebPage(self._web)
        self._web.setPage(web_page)
        self._web.page().setBackgroundColor(QColor('#1c1c1c'))
        self._web.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self._web.setStyleSheet("QWebEngineView { background-color: #1c1c1c; border: none; }")
        content_layout.addWidget(self._web, stretch=1)

        self._btn_right = QPushButton("›")
        self._btn_right.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._btn_right.setToolTip("Next chapter  [→]")
        self._btn_right.clicked.connect(self._go_next)
        content_layout.addWidget(self._btn_right)

        outer.addWidget(content, stretch=1)

        status = QStatusBar()
        status.setStyleSheet(_READER_STATUS_STYLE)
        status.setSizeGripEnabled(False)
        self.setStatusBar(status)
        self._update_status()
        self._apply_arrow_styles()

    def _build_topbar(self):
        bar = QWidget()
        bar.setObjectName("ReaderTopBar")
        bar.setStyleSheet(_READER_TOPBAR_STYLE)
        bar.setFixedHeight(36)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(6)

        btn_close = QPushButton("✕  Close")
        btn_close.setToolTip("Close reader  [Esc]")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

        layout.addWidget(self._make_sep())

        self._chapter_label = QLabel("—")
        self._chapter_label.setObjectName("ChapterLabel")
        self._chapter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chapter_label.setMinimumWidth(200)
        layout.addWidget(self._chapter_label)

        layout.addWidget(self._make_sep())

        self._btn_dark = QPushButton("☀ Light" if self._dark_mode else "🌙 Dark")
        self._btn_dark.setCheckable(True)
        self._btn_dark.setChecked(self._dark_mode)
        self._btn_dark.setToolTip("Toggle dark / light mode")
        self._btn_dark.clicked.connect(self._toggle_dark)
        layout.addWidget(self._btn_dark)

        layout.addWidget(self._make_sep())

        self._btn_sepia = QPushButton("📜 Sepia")
        self._btn_sepia.setCheckable(True)
        self._btn_sepia.setChecked(False)
        self._btn_sepia.setToolTip("Toggle sepia reading mode")
        self._btn_sepia.clicked.connect(self._toggle_sepia)
        layout.addWidget(self._btn_sepia)

        layout.addStretch()

        hint = QLabel("← →  navigate chapters     Esc  close")
        hint.setStyleSheet("color: #444; font-size: 11px;")
        layout.addWidget(hint)

        return bar

    @staticmethod
    def _make_sep():
        sep = QLabel("|")
        sep.setStyleSheet("color: #333; padding: 0 2px;")
        return sep

    def _apply_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Escape),    self, self.close)
        QShortcut(QKeySequence(Qt.Key.Key_Left),      self, self._go_prev)
        QShortcut(QKeySequence(Qt.Key.Key_Right),     self, self._go_next)
        QShortcut(QKeySequence("Ctrl+R"),             self, self._reload_current)

    def _toggle_dark(self, checked: bool):
        self._dark_mode = checked
        self._btn_dark.setText("☀ Light" if checked else "🌙 Dark")
        if checked and self._sepia_mode:
            self._sepia_mode = False
            self._btn_sepia.setChecked(False)
        self._apply_arrow_styles()
        self._reload_current()

    def _toggle_sepia(self, checked: bool):
        self._sepia_mode = checked
        if checked and self._dark_mode:
            self._dark_mode = False
            self._btn_dark.setChecked(False)
            self._btn_dark.setText("🌙 Dark")
        self._apply_arrow_styles()
        self._reload_current()

    def _apply_arrow_styles(self):
        if self._sepia_mode:
            style = _READER_ARROW_STYLE_SEPIA
        elif self._dark_mode:
            style = _READER_ARROW_STYLE_DARK
        else:
            style = _READER_ARROW_STYLE_LIGHT
        self._btn_left.setStyleSheet(style)
        self._btn_right.setStyleSheet(style)
    def _go_prev(self):
        if self._chapter_idx > 0:
            self._chapter_idx -= 1
            href = self._chapters[self._chapter_idx]
            first_idx = EPUBPreviewEngine.first_para_index_for_chapter(self._paragraphs, href)
            para = self._paragraphs[first_idx] if first_idx >= 0 else {'item_href': href}
            self._load_chapter(href, para)

    def _go_next(self):
        if self._chapter_idx < len(self._chapters) - 1:
            self._chapter_idx += 1
            href = self._chapters[self._chapter_idx]
            first_idx = EPUBPreviewEngine.first_para_index_for_chapter(self._paragraphs, href)
            para = self._paragraphs[first_idx] if first_idx >= 0 else {'item_href': href}
            self._load_chapter(href, para)

    def _reload_current(self):
        href = self._chapters[self._chapter_idx] if self._chapters else ''
        if not href:
            return
        first_idx = EPUBPreviewEngine.first_para_index_for_chapter(self._paragraphs, href)
        para = self._paragraphs[first_idx] if first_idx >= 0 else {'item_href': href}
        self._load_chapter(href, para)

    def _load_chapter(self, item_href: str, representative_para: dict,
                      scroll_to_selected: bool = True):
        self._current_href = item_href
        try:
            self._chapter_idx = self._chapters.index(item_href)
        except ValueError:
            pass

        css_extra = _READER_PAGE_CSS
        if self._sepia_mode:
            css_extra += _READER_PAGE_SEPIA_CSS
        elif self._dark_mode:
            css_extra += _READER_PAGE_DARK_CSS

        try:
            html_str = self._engine.generate_preview_html(
                self._book,
                self._paragraphs,
                representative_para,
                show_original_ids=None,
                dark_mode=self._dark_mode,
                reader_css=css_extra,
                scroll_to_selected=scroll_to_selected,
                reader_mode=True,
                skip_mismatch=True,
            )
            base_url = QUrl.fromLocalFile(
                os.path.abspath(self._book.content_dir) + os.sep
            )
            if self._sepia_mode:
                self._web.page().setBackgroundColor(QColor('#e8dcc8'))
            elif self._dark_mode:
                self._web.page().setBackgroundColor(QColor('#1c1c1c'))
            else:
                self._web.page().setBackgroundColor(QColor('#ffffff'))
            if self._sepia_mode:
                self._web.loadFinished.connect(self._apply_sepia_js)
            self._web.setContent(html_str.encode('utf-8'), 'application/xhtml+xml', base_url)
        except Exception as exc:
            logger.error(f'[EPUBReader] load failed: {exc}', exc_info=True)

        self._update_chapter_label()
        self._update_nav_buttons()
        self._update_status()

    def _apply_sepia_js(self, _ok=None):
        self._web.loadFinished.disconnect(self._apply_sepia_js)
        self._web.page().runJavaScript(_SEPIA_FORCE_JS)

    def _update_chapter_label(self):
        if not self._chapters:
            self._chapter_label.setText("—")
            return
        total = len(self._chapters)
        idx   = self._chapter_idx + 1
        short = os.path.basename(self._current_href)
        self._chapter_label.setText(f"{short}  ({idx} / {total})")
        self._chapter_label.setToolTip(self._current_href)

    def _update_nav_buttons(self):
        can_prev = bool(self._chapters) and self._chapter_idx > 0
        can_next = bool(self._chapters) and self._chapter_idx < len(self._chapters) - 1
        self._btn_left.setEnabled(can_prev)
        self._btn_right.setEnabled(can_next)

    def _update_status(self):
        if not self._paragraphs or not self._current_href:
            self.statusBar().showMessage("")
            return
        chapter_paras = [p for p in self._paragraphs if p.get('item_href') == self._current_href]
        total         = len(chapter_paras)
        translated    = sum(1 for p in chapter_paras if p.get('is_translated'))
        chapter_num   = self._chapter_idx + 1
        chapters_total = len(self._chapters)
        self.statusBar().showMessage(
            f"Chapter {chapter_num} of {chapters_total}  —  "
            f"{translated} / {total} fragments translated"
        )

    def refresh_chapter(self, book, paragraphs, selected_para, dark_mode: bool = None):
        self._book       = book
        self._paragraphs = paragraphs
        if dark_mode is not None:
            if dark_mode != self._dark_mode:
                self._dark_mode = dark_mode
                self._btn_dark.setChecked(dark_mode)
                self._btn_dark.setText("☀ Light" if dark_mode else "🌙 Dark")
        js = self._engine.generate_refresh_js(
            book, paragraphs, self._current_href
        )
        if js:
            self._web.page().runJavaScript(js)
        self._update_status()

    def navigate_to_para(self, book, paragraphs, para_index: int, dark_mode: bool = None):
        """
        Switch to the chapter containing *para_index* and highlight it.
        """
        self._book       = book
        self._paragraphs = paragraphs
        if dark_mode is not None:
            self._dark_mode = dark_mode
        if para_index < 0 or para_index >= len(paragraphs):
            return
        para = paragraphs[para_index]
        href = para.get('item_href', '')
        if not href:
            return
        if href not in self._chapters:
            self._chapters = EPUBPreviewEngine.get_chapter_list(paragraphs)
        self._load_chapter(href, para)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class EPUBPreviewEngine:
    """
    Builds a preview HTML string for a given EPUB fragment.

    The engine makes a deep copy of the source .xhtml lxml tree so it
    never modifies the in-memory book.  All translated paragraphs that
    belong to the same .xhtml file are injected before the result is
    serialised to a string.
    """

    def __init__(self):
        self._ns = {'x': XHTML_NS}

    @staticmethod
    def get_chapter_list(paragraphs: list) -> list:
        """
        Return an ordered, deduplicated list of item_href values found in
        *paragraphs*.  The order mirrors the original paragraph sequence.
        """
        seen = []
        for para in paragraphs:
            href = para.get('item_href', '')
            if href and href not in seen:
                seen.append(href)
        return seen

    @staticmethod
    def first_para_index_for_chapter(paragraphs: list, item_href: str) -> int:
        """
        Return the index of the first paragraph whose item_href matches
        *item_href*, or -1 if none is found.
        """
        for i, para in enumerate(paragraphs):
            if para.get('item_href') == item_href:
                return i
        return -1

    def generate_preview_html(self, book, paragraphs, selected_para,
                              show_original_ids=None, dark_mode: bool = False,
                              reader_css: str = '', scroll_to_selected: bool = True,
                              reader_mode: bool = False, skip_mismatch: bool = False):
        """
        Return an XHTML string suitable for QWebEngineView.setContent().

        Parameters
        ----------
        book : epub_utils.OEBBook
        paragraphs : list[dict]
            All paragraph dicts loaded from the EPUB.
        selected_para : dict
            The paragraph that is currently selected in the UI.
        show_original_ids : set, optional
            Set of element IDs that should be displayed in their original
            (untranslated) form, even if a translation exists.
        dark_mode : bool, optional
            When True, inject a dark-background / light-text override CSS
            block so the preview is readable on dark-themed EPUBs or in
            dark-mode usage.  Defaults to False.

        Returns
        -------
        str
            UTF-8 serialisable XHTML.
        """
        item_href = selected_para.get('item_href')
        if not item_href:
            return self._error_html('Missing item_href in selected fragment.')

        item = next((i for i in book.items if i.href == item_href), None)
        if item is None or item.data is None:
            return self._error_html(f'File .xhtml not found: {item_href}')

        root = copy.deepcopy(item.data)

        _show_original = show_original_ids if show_original_ids is not None else set()

        for para in paragraphs:
            if para.get('item_href') != item_href:
                continue
            if not para.get('is_translated') or not para.get('translated_text', '').strip():
                continue
            if (skip_mismatch
                    and para.get('has_mismatch')
                    and not para.get('ignore_mismatch')
                    and not para.get('force_mismatch')):
                continue
            elem_id = para.get('id')
            if not elem_id:
                continue
            if elem_id in _show_original:
                continue
            elements = root.xpath(f'.//*[@id="{elem_id}"]')
            if not elements:
                continue
            element = elements[0]
            mode = para.get('processing_mode', 'inline')
            try:
                if mode == 'legacy':
                    self._inject_legacy(element, para)
                else:
                    self._inject_inline(element, para)
            except Exception as exc:
                logger.warning(
                    f'[EPUBPreview] Injection failed for id={elem_id}: {exc}',
                    exc_info=True
                )

        sel_id = selected_para.get('id')
        if sel_id:
            sel_elems = root.xpath(f'.//*[@id="{sel_id}"]')
            if sel_elems:
                el = sel_elems[0]
                el.set('data-epub-preview-selected', '1')

        for orig_id in _show_original:
            orig_elems = root.xpath(f'.//*[@id="{orig_id}"]')
            if orig_elems:
                orig_elems[0].set('data-epub-preview-show-original', '1')

        self._inject_head_style(root, _HIGHLIGHT_CSS)
        if not reader_mode:
            self._inject_head_style(root, _SELECTED_HIGHLIGHT_CSS)
            self._inject_head_style(root, _PREVIEW_SCROLLBAR_CSS)
        if dark_mode:
            self._inject_head_style(root, _DARK_MODE_CSS)
        if reader_mode:
            self._inject_head_style(root, _NO_SCROLLBAR_CSS)
        if reader_css:
            self._inject_head_style(root, reader_css)

        abs_content_dir = os.path.abspath(book.content_dir)
        item_subdir = os.path.dirname(item_href)
        self._fix_resource_paths(root, abs_content_dir, item_subdir)

        if scroll_to_selected:
            self._inject_body_script(root, _SCROLL_TO_SEL_JS)
        if reader_mode:
            self._inject_body_script(root, _DRAG_SCROLL_JS)
        else:
            self._inject_body_script(root, _INTERACTION_JS)

        return etree.tostring(root, encoding='unicode', method='xml')

    def _inject_legacy(self, element, para):
        """
        Inject translation for a legacy-mode paragraph.

        Priority:
          1. aligned_translated_html  → used when alignment has been run.
          2. Plain translated_text    → used when no alignment yet.
             (<id_XX> reserve-element placeholders are restored.)
        """
        aligned_html = para.get('aligned_translated_html', '').strip()

        if aligned_html:
            translated_text = para.get('translated_text', '').strip()
            if translated_text:
                aligned_plain = re.sub(r'<[^>]+>', '', aligned_html)
                aligned_plain = re.sub(r'\s+', ' ', aligned_plain).strip()
                original_html_str = para.get('original_html', '')
                original_plain = re.sub(r'<[^>]+>', '', original_html_str)
                original_plain = re.sub(r'\s+', ' ', original_plain).strip()
                if aligned_plain == original_plain:
                    logger.debug(
                        '[EPUBPreview] aligned_translated_html mirrors original '
                        f'(para id={para.get("id","?")}), falling back to plain text.'
                    )
                    aligned_html = ''

        if aligned_html:
            try:
                aligned_elem = etree.fromstring(aligned_html.encode('utf-8'))
                orig_id = element.get('id', '')
                if orig_id:
                    aligned_elem.set('id', orig_id)
                aligned_elem.tail = element.tail
                parent = element.getparent()
                if parent is not None:
                    parent.replace(element, aligned_elem)
                return
            except etree.XMLSyntaxError as exc:
                logger.warning(
                    f'[EPUBPreview] Cannot parse aligned_translated_html: {exc} '
                    '— falling back to plain-text injection.'
                )

        translation = self._restore_reserve_placeholders(
            para.get('translated_text', ''), para
        )
        self._replace_element_content_with_html(element, translation)

    def _inject_inline(self, element, para):
        """
        Inject translation for an inline-mode paragraph.

        Restores all placeholder types:
          - auto-wrap outer tags
          - prefix / suffix reserve tags
          - non-translatable <nt_XX/> markers
          - <id_XX> reserve-element placeholders
          - <p_XX> / </p_XX> inline-formatting placeholders
        Then rebuilds the element DOM from the resulting text.
        """
        translated_text = para.get('translated_text', '')
        inline_map      = para.get('inline_formatting_map', {})
        non_trans_map   = para.get('non_translatable_placeholders', {})
        reserve_elems   = para.get('reserve_elements', [])
        auto_wrap_tags  = para.get('auto_wrap_tags', [])
        prefix_tags     = para.get('prefix_reserve_tags', [])
        suffix_tags     = para.get('suffix_reserve_tags', [])

        full_text = translated_text
        if '\n' in full_text:
            full_text = re.sub(r'\n\s*(?=<)', '', full_text)
            full_text = re.sub(r'(?<=>)\s*\n', '', full_text)
            full_text = full_text.replace('\n', ' ')
            full_text = re.sub(r'  +', ' ', full_text)

        full_text = re.sub(r'</?ps(?:_\d{2})?>', '', full_text)
        full_text = re.sub(r'  +', ' ', full_text)

        for tag_info in reversed(auto_wrap_tags):
            opening = tag_info.get('opening', '')
            closing = tag_info.get('closing', '')
            full_text = opening + full_text + closing

        if prefix_tags:
            full_text = ' '.join(prefix_tags) + ' ' + full_text
        if suffix_tags:
            full_text = full_text.rstrip() + ' ' + ' '.join(suffix_tags)

        for tag_id, info in non_trans_map.items():
            marker   = f'<nt_{int(tag_id):02d}/>'
            original = info.get('full_match', '')
            if marker in full_text:
                full_text = full_text.replace(marker, original)
            else:
                full_text = full_text.rstrip() + original

        element.text = ''
        for child in list(element):
            element.remove(child)

        self._rebuild_element_from_placeholders(
            element, full_text, inline_map, reserve_elems
        )

    def _restore_reserve_placeholders(self, text, para):
        """Replace <id_XX> placeholders with their stored HTML strings."""
        reserve_elements    = para.get('reserve_elements', [])
        placeholder_pattern = para.get('placeholder_pattern', '<id_{:02d}>')
        for i, html_str in enumerate(reserve_elements):
            placeholder = placeholder_pattern.format(i)
            text = text.replace(placeholder, html_str)
        return text

    def _replace_element_content_with_html(self, element, html_content):
        """
        Replace the children of *element* by parsing *html_content* as XML.
        Falls back to stripping tags and using plain text on parse failure.
        """
        tag = etree.QName(element).localname

        attribs_str = ''
        for k, v in element.attrib.items():
            safe_v = v.replace('&', '&amp;').replace('"', '&quot;')
            attribs_str += f' {k}="{safe_v}"'

        wrapper_xml = (
            f'<{tag} xmlns="{XHTML_NS}"{attribs_str}>'
            f'{html_content}'
            f'</{tag}>'
        )
        try:
            new_elem = etree.fromstring(wrapper_xml.encode('utf-8'))
            new_elem.tail = element.tail
            parent = element.getparent()
            if parent is not None:
                parent.replace(element, new_elem)
        except etree.XMLSyntaxError:
            plain = re.sub(r'<[^>]+>', '', html_content)
            element.text = plain
            for child in list(element):
                element.remove(child)

    def _rebuild_element_from_placeholders(
        self, parent, text, formatting_map, reserve_elements
    ):
        """
        Walk *text* token by token, creating child lxml elements for
        <p_XX> / </p_XX> inline-tag markers and <id_XX> reserve-element
        markers.  Plain text between markers is appended as text/tail nodes.

        This is a faithful simplified copy of the same method in
        EPUBCreatorLxml, kept here so the preview module stays self-contained
        and independent of epub_creator_lxml.py.
        """
        pattern  = r'<p_(\d{2})>|</p_(\d{2})>|<id_(\d{2})>'
        stack    = [(parent, None)]
        buffer   = ''
        last_pos = 0

        for match in re.finditer(pattern, text):
            buffer += text[last_pos:match.start()]
            placeholder = match.group(0)

            if placeholder.startswith('<p_') and not placeholder.startswith('</p_'):
                tag_id = int(match.group(1))

                if buffer:
                    if tag_id in formatting_map:
                        info = formatting_map[tag_id]
                        if info.get('has_leading_space', False) and not buffer[-1].isspace():
                            buffer += ' '
                    self._append_text(stack[-1][0], buffer)
                    buffer = ''
                else:
                    if tag_id in formatting_map:
                        info = formatting_map[tag_id]
                        if info.get('has_leading_space', False):
                            cur = stack[-1][0]
                            if len(cur) > 0:
                                lc = cur[-1]
                                lt = lc.tail or ''
                                if not lt or not lt[-1].isspace():
                                    lc.tail = lt + ' '
                            else:
                                pt = cur.text or ''
                                if not pt or not pt[-1].isspace():
                                    cur.text = pt + ' '

                if tag_id in formatting_map:
                    new_elem = self._create_formatting_element(formatting_map[tag_id])
                    stack[-1][0].append(new_elem)
                    stack.append((new_elem, tag_id))
                else:
                    logger.debug(
                        f'[EPUBPreview] _rebuild: tag_id {tag_id} not in formatting_map, skipped.'
                    )

            elif placeholder.startswith('</p_'):
                tag_id = int(match.group(2))

                if buffer:
                    self._append_text(stack[-1][0], buffer)
                    buffer = ''

                if len(stack) > 1 and stack[-1][1] == tag_id:
                    stack.pop()
                    if tag_id in formatting_map:
                        info = formatting_map[tag_id]
                        if info.get('has_trailing_space', False):
                            next_ch = text[match.end():match.end() + 1]
                            if next_ch and not next_ch.isspace() and next_ch not in '.,;:!?':
                                buffer += ' '
                else:
                    logger.debug(
                        f'[EPUBPreview] _rebuild: stack mismatch on </p_{tag_id:02d}>'
                    )

            elif placeholder.startswith('<id_'):
                reserve_id = int(match.group(3))

                if buffer:
                    self._append_text(stack[-1][0], buffer)
                    buffer = ''

                if reserve_id < len(reserve_elements):
                    self._insert_reserve_html(stack[-1][0], reserve_elements[reserve_id])
                else:
                    logger.debug(
                        f'[EPUBPreview] _rebuild: invalid reserve_id {reserve_id} '
                        f'(max {len(reserve_elements) - 1})'
                    )

            last_pos = match.end()

        buffer += text[last_pos:]
        if buffer:
            self._append_text(stack[-1][0], buffer)

    def _create_formatting_element(self, info):
        """Create a new lxml element from an inline_formatting_map entry."""
        tag_name   = info['tag']
        attributes = info.get('attributes', {})
        nsmap      = {None: XHTML_NS}
        elem = etree.Element(f'{{{XHTML_NS}}}{tag_name}', nsmap=nsmap)
        for name, value in attributes.items():
            elem.set(name, value)
        return elem

    def _append_text(self, element, text):
        """Append *text* to the tail of the last child, or to element.text."""
        if len(element) == 0:
            element.text = (element.text or '') + text
        else:
            lc = element[-1]
            lc.tail = (lc.tail or '') + text

    def _insert_reserve_html(self, parent, reserve_html):
        """Parse *reserve_html* and append the resulting element to *parent*."""
        try:
            if 'xmlns' in reserve_html:
                elem = etree.fromstring(reserve_html.encode('utf-8'))
                parent.append(elem)
                return
            wrapped = f'<_temp xmlns="{XHTML_NS}">{reserve_html}</_temp>'
            temp_root = etree.fromstring(wrapped.encode('utf-8'))
            if len(temp_root) > 0:
                parent.append(temp_root[0])
                return
        except etree.XMLSyntaxError as exc:
            logger.debug(f'[EPUBPreview] Could not parse reserve HTML: {exc}')

        plain = re.sub(r'<[^>]+>', '', reserve_html)
        self._append_text(parent, plain)

    def _fix_resource_paths(self, root, abs_content_dir, item_subdir):
        """
        Rewrite relative href/src attributes to absolute file:// URLs so that
        QWebEngineView can load CSS files, images, fonts, etc.
        """
        if item_subdir:
            item_dir = os.path.normpath(
                os.path.join(abs_content_dir, item_subdir)
            )
        else:
            item_dir = abs_content_dir

        for elem in root.iter():
            for attr in ('href', 'src'):
                val = elem.get(attr)
                if not val:
                    continue
                if val.startswith(
                    ('http://', 'https://', 'data:', 'javascript:', 'file://')
                ):
                    continue
                if val.startswith('#'):
                    continue

                fragment = ''
                if '#' in val:
                    val, fragment = val.split('#', 1)
                    fragment = '#' + fragment

                if not val:
                    continue

                abs_path = os.path.normpath(os.path.join(item_dir, val))
                url = 'file:///' + abs_path.replace(os.sep, '/') + fragment
                elem.set(attr, url)

    def _inject_head_style(self, root, css_text):
        """Append a <style> block to the <head> element."""
        head = root.find(f'{{{XHTML_NS}}}head')
        if head is None:
            head = root.find('head')
        if head is None:
            return
        style = etree.SubElement(head, f'{{{XHTML_NS}}}style')
        style.set('type', 'text/css')
        style.text = css_text

    def _inject_body_script(self, root, js_text):
        """Append an inline <script> to the <body> element."""
        body = root.find(f'.//{{{XHTML_NS}}}body')
        if body is None:
            body = root.find('.//body')
        if body is None:
            return
        script = etree.SubElement(body, f'{{{XHTML_NS}}}script')
        script.set('type', 'text/javascript')
        script.text = js_text

    def generate_refresh_js(self, book, paragraphs, item_href, show_original_ids=None):
        item = next((i for i in book.items if i.href == item_href), None)
        if item is None or item.data is None:
            return ''

        _show_original = show_original_ids or set()
        js_parts = []

        for para in paragraphs:
            if para.get('item_href') != item_href:
                continue
            if not para.get('is_translated') or not para.get('translated_text', '').strip():
                continue
            if (para.get('has_mismatch')
                    and not para.get('ignore_mismatch')
                    and not para.get('force_mismatch')):
                continue
            elem_id = para.get('id')
            if not elem_id or elem_id in _show_original:
                continue

            elements = item.data.xpath(f'.//*[@id="{elem_id}"]')
            if not elements:
                continue

            elem_copy = copy.deepcopy(elements[0])
            wrapper = etree.Element('_wrapper')
            wrapper.append(elem_copy)

            mode = para.get('processing_mode', 'inline')
            try:
                if mode == 'legacy':
                    self._inject_legacy(elem_copy, para)
                else:
                    self._inject_inline(elem_copy, para)
            except Exception as exc:
                logger.warning(f'[EPUBPreview] JS refresh injection failed id={elem_id}: {exc}')
                continue

            result_elem = wrapper[0] if len(wrapper) > 0 else elem_copy
            inner = self._element_inner_html(result_elem)

            safe_id = elem_id.replace('\\', '\\\\').replace('"', '\\"')
            escaped = inner.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
            js_parts.append(
                f'(function(){{var e=document.getElementById("{safe_id}");'
                f'if(e){{e.innerHTML=`{escaped}`;}}}})();'
            )

        return '\n'.join(js_parts)

    def _element_inner_html(self, element):
        s = etree.tostring(element, encoding='unicode', method='html')
        try:
            start = s.index('>') + 1
            end = s.rindex('<')
            return s[start:end]
        except ValueError:
            return ''

    @staticmethod
    def _error_html(message):
        safe = (
            message
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
        )
        return (
            '<?xml version="1.0" encoding="utf-8"?>'
            f'<html xmlns="{XHTML_NS}">'
            '<head><title>Preview Error</title></head>'
            '<body>'
            '<p style="color:#ff6b6b;font-family:monospace;'
            'padding:24px;font-size:14px;">'
            f'⚠ Preview Error: {safe}'
            '</p>'
            '</body></html>'
        )