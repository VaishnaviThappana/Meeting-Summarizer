import os
import io
import base64

from docx import Document
from PyPDF2 import PdfReader
from fpdf import FPDF


# ── File Parsers ──────────────────────────────────────────────────────────────

def parse_docx(file) -> str:
    """Parse a .docx file and return its text content."""
    doc = Document(file)
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def parse_pdf(file) -> str:
    """Parse a PDF file and return its text content."""
    reader = PdfReader(file)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return "\n".join(pages_text)


# ── PDF Report Export ─────────────────────────────────────────────────────────

def _safe_text(text: str) -> str:
    """Encode text to latin-1 safely, replacing characters that cannot be represented."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def create_pdf_download_link(summary: str, action_items: dict, filename: str = "meeting_summary.pdf") -> str:
    """Generate a base64-encoded PDF report and return an HTML download link.

    Args:
        summary: The executive summary text.
        action_items: Dict with keys 'tasks', 'decisions', 'deadlines'.
        filename: Name for the downloaded file.

    Returns:
        An HTML anchor tag string for downloading the PDF.
    """
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Meeting Summary Report", ln=True, align="C")
    pdf.ln(8)

    # Executive Summary
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Executive Summary", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, _safe_text(summary))
    pdf.ln(4)

    # Tasks
    if action_items.get("tasks"):
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Tasks", ln=True)
        pdf.set_font("Arial", "", 11)
        for t in action_items["tasks"]:
            pdf.multi_cell(0, 7, _safe_text(f"  - {t}"))

    # Decisions
    if action_items.get("decisions"):
        pdf.ln(2)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Key Decisions", ln=True)
        pdf.set_font("Arial", "", 11)
        for d in action_items["decisions"]:
            pdf.multi_cell(0, 7, _safe_text(f"  - {d}"))

    # Deadlines
    if action_items.get("deadlines"):
        pdf.ln(2)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Deadlines", ln=True)
        pdf.set_font("Arial", "", 11)
        for dl in action_items["deadlines"]:
            pdf.multi_cell(0, 7, _safe_text(f"  - {dl['deadline']}: {dl['context']}"))

    # Output as bytes
    raw = pdf.output(dest="S")
    # fpdf 1.7.x returns a str; fpdf2 returns bytes — handle both
    if isinstance(raw, str):
        pdf_bytes = raw.encode("latin-1")
    else:
        pdf_bytes = raw

    b64 = base64.b64encode(pdf_bytes).decode()
    return (
        f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" '
        f'style="text-decoration:none; background-color:#2e6c80; color:white; '
        f'padding:10px 20px; border-radius:5px; display:inline-block;">'
        f'⬇️ Download PDF Report</a>'
    )


# ── Audio Transcription ───────────────────────────────────────────────────────

# Module-level cache: loaded once, reused across calls.
_asr_pipeline = None


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file to text using HuggingFace Whisper pipeline.

    Uses openai/whisper-small via the transformers ASR pipeline.
    No external ffmpeg binary required — librosa handles decoding.
    The pipeline is cached after first load.

    Args:
        audio_path: Path to the audio file (.mp3 / .wav / .m4a / .flac).

    Returns:
        The full transcript as a single string.

    Raises:
        RuntimeError: If transcription fails.
    """
    global _asr_pipeline

    try:
        import torch
        from transformers import pipeline as hf_pipeline
        import librosa
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency: {e}. "
            "Run: pip install transformers torch librosa"
        ) from e

    # ── Load pipeline once ────────────────────────────────────────────────────
    if _asr_pipeline is None:
        try:
            _asr_pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-small",
                device=-1,          # CPU
                chunk_length_s=30,  # stream long audio in 30-second windows
                stride_length_s=5,
            )
        except Exception:
            # Fallback to even smaller model
            _asr_pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=-1,
                chunk_length_s=30,
                stride_length_s=5,
            )

    # ── Load audio with librosa (handles mp3/wav/m4a/flac without system ffmpeg)
    try:
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not decode audio file '{audio_path}': {e}\n"
            "Ensure the file is a valid .mp3, .wav, .m4a, or .flac recording."
        ) from e

    # ── Run ASR ───────────────────────────────────────────────────────────────
    try:
        result = _asr_pipeline(
            {"array": audio_array, "sampling_rate": sampling_rate},
            return_timestamps=False,
            generate_kwargs={"task": "translate", "language": "en"},
        )
        text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
        if not text:
            raise RuntimeError("Whisper returned an empty transcript. The audio may be silent or too short.")
        return text
    except Exception as e:
        raise RuntimeError(f"Transcription error: {e}") from e
