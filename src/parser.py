import os
from pathlib import Path
from typing import List
from rich.progress import track
from docling.document_converter import DocumentConverter
from .config import CONFIG

SUPPORTED_EXTS = {".pdf"}

def parse_pdfs(source_dir: str = "tech_specs", out_dir: str = CONFIG.parsed_dir) -> List[Path]:
    os.makedirs(out_dir, exist_ok=True)
    converter = DocumentConverter()
    outputs: List[Path] = []

    for file in track(list(Path(source_dir).glob('*.pdf')), description="Parsing PDFs"):
        stem = file.stem
        out_md = Path(out_dir) / f"{stem}.md"
        if out_md.exists():
            outputs.append(out_md)
            continue
        try:
            result = converter.convert(file)
            # docling result has .document export methods
            md_text = result.document.export_to_markdown()
            # Basic cleaning (placeholder for custom regex cleaning)
            md_text = md_text.replace('\u00A0', ' ').strip()
            out_md.write_text(md_text, encoding='utf-8')
            outputs.append(out_md)
        except Exception as e:
            print(f"Failed to parse {file}: {e}")
    return outputs

if __name__ == "__main__":
    parse_pdfs()
