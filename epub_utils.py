# epub_utils.py
import logging
import os
import shutil
import tempfile
import zipfile
from lxml import etree
from pathlib import Path

logger = logging.getLogger(__name__)

NAMESPACES = {
    'x': 'http://www.w3.org/1999/xhtml',
    'opf': 'http://www.idpf.org/2007/opf',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'ncx': 'http://www.daisy.org/z3986/2005/ncx/'
}

for prefix, uri in NAMESPACES.items():
    if prefix != 'x':
        etree.register_namespace(prefix, uri)

class OEBItem:
    def __init__(self, href, content_bytes, media_type='text/html'):
        self.href = href
        self.media_type = media_type
        self.original_bytes = content_bytes

        if media_type in ('text/html', 'application/xhtml+xml'):
            try:
                parser = etree.XMLParser(
                    remove_blank_text=False,
                    resolve_entities=False,
                    strip_cdata=False
                )
                self.data = etree.fromstring(content_bytes, parser=parser)
            except etree.XMLSyntaxError:
                parser = etree.HTMLParser()
                self.data = etree.fromstring(content_bytes, parser=parser)
        else:
            self.data = None

    def get_id(self):
        return Path(self.href).stem

    def to_bytes(self):
        if self.data is not None:
            return etree.tostring(
                self.data,
                encoding='utf-8',
                xml_declaration=True,
                pretty_print=False,
                method='xml'
            )
        return self.original_bytes

class OEBBook:
    def __init__(self, epub_path):
        self.epub_path = epub_path
        self.temp_dir = None
        self.items = []
        self.metadata = {}
        self.opf_path = None
        self.content_dir = None

        self._extract_epub()
        self._parse_container()
        self._parse_opf()
        self._load_items()

    def _extract_epub(self):
        self.temp_dir = tempfile.mkdtemp(prefix='epub_')

        with zipfile.ZipFile(self.epub_path, 'r') as zf:
            zf.extractall(self.temp_dir)

        logger.debug(f"Extracted EPUB to: {self.temp_dir}")

    def _parse_container(self):
        container_path = os.path.join(self.temp_dir, 'META-INF', 'container.xml')

        if not os.path.exists(container_path):
            raise ValueError("Invalid EPUB: missing META-INF/container.xml")

        tree = etree.parse(container_path)

        rootfile = tree.xpath(
            '//ocf:rootfile[@media-type="application/oebps-package+xml"]',
            namespaces={'ocf': 'urn:oasis:names:tc:opendocument:xmlns:container'}
        )

        if not rootfile:
            raise ValueError("Invalid EPUB: no rootfile in container.xml")

        self.opf_path = os.path.join(self.temp_dir, rootfile[0].get('full-path'))
        self.content_dir = os.path.dirname(self.opf_path)

        logger.debug(f"Found OPF at: {self.opf_path}")

    def _parse_opf(self):
        tree = etree.parse(self.opf_path)
        root = tree.getroot()

        metadata_elem = root.find('.//opf:metadata', namespaces=NAMESPACES)
        if metadata_elem is not None:
            for elem in metadata_elem:
                tag = etree.QName(elem).localname
                self.metadata[tag] = elem.text

        manifest = root.find('.//opf:manifest', namespaces=NAMESPACES)
        self.manifest_items = []

        if manifest is not None:
            for item in manifest.findall('.//opf:item', namespaces=NAMESPACES):
                self.manifest_items.append({
                    'id': item.get('id'),
                    'href': item.get('href'),
                    'media_type': item.get('media-type')
                })

        logger.debug(f"Found {len(self.manifest_items)} items in manifest")

    def _load_items(self):
        for manifest_item in self.manifest_items:
            media_type = manifest_item['media_type']

            if media_type not in ('text/html', 'application/xhtml+xml'):
                continue

            href = manifest_item['href']
            file_path = os.path.join(self.content_dir, href)

            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue

            with open(file_path, 'rb') as f:
                content = f.read()

            item = OEBItem(href, content, media_type)
            self.items.append(item)

        logger.debug(f"Loaded {len(self.items)} HTML/XHTML items")

    def get_items_of_type(self, item_type='DOCUMENT'):
        return self.items

    def save(self, output_path):
        for item in self.items:
            file_path = os.path.join(self.content_dir, item.href)

            with open(file_path, 'wb') as f:
                f.write(item.to_bytes())

        self._create_epub_zip(output_path)

        logger.debug(f"Saved EPUB to: {output_path}")

    def _create_epub_zip(self, output_path):
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            mimetype_path = os.path.join(self.temp_dir, 'mimetype')
            if os.path.exists(mimetype_path):
                zf.write(
                    mimetype_path,
                    'mimetype',
                    compress_type=zipfile.ZIP_STORED
                )

            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    if file == 'mimetype':
                        continue

                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.temp_dir)

                    zf.write(file_path, arcname, compress_type=zipfile.ZIP_DEFLATED)

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.debug(f"Cleaned up temp dir: {self.temp_dir}")

    def __del__(self):
        self.cleanup()

def read_epub(epub_path):
    return OEBBook(epub_path)

def write_epub(output_path, book):
    book.save(output_path)
