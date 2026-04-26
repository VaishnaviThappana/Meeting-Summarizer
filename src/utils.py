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

# Module-level cache: model is loaded once and reused for all calls.
_whisper_model = None


def _ensure_ffmpeg_on_path() -> None:
    """Make the imageio-ffmpeg binary discoverable by faster-whisper.

    faster-whisper (via ctranslate2 / ffmpeg-python) respects the PATH
    environment variable. On Windows the env-var update is visible to the
    *current* Python process without a restart, which is all we need because
    faster-whisper spawns ffmpeg as a child process of this same process.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(os.path.abspath(ffmpeg_exe))

        # Prepend dir to PATH so child processes find it
        current = os.environ.get("PATH", "")
        if ffmpeg_dir not in current:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + current

        # Also set the explicit env-var that some Whisper builds honour
        os.environ.setdefault("FFMPEG_BINARY", ffmpeg_exe)

    except Exception as e:
        raise RuntimeError(
            f"Could not locate ffmpeg. "
            f"Run: pip install imageio-ffmpeg\n({e})"
        ) from e


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file to text using faster-whisper.

    The WhisperModel is cached after the first call so subsequent
    transcriptions do not pay the model-load cost.

    Args:
        audio_path: Path to the audio file on disk (must keep original extension
                    so ffmpeg can identify the format, e.g. .mp3, .wav, .m4a).

    Returns:
        The full transcript as a single string.

    Raises:
        RuntimeError: If ffmpeg is missing or transcription fails.
    """
    global _whisper_model

    # Ensure ffmpeg is findable before loading Whisper
    _ensure_ffmpeg_on_path()

    from faster_whisper import WhisperModel

    # Load once, reuse forever
    if _whisper_model is None:
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

    try:
        segments, _info = _whisper_model.transcribe(audio_path, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments)
    except Exception as e:
        raise RuntimeError(f"Transcription error: {e}") from e
