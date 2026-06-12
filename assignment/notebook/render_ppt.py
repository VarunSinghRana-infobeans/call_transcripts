"""
render_ppt.py

Render each slide of a PowerPoint deck to PNG.

For clean, watermark-free screenshots use LibreOffice (soffice) if it is on PATH
or installed in a standard location. If LibreOffice is not available, this script
falls back to Aspose.Slides, which stamps evaluation watermarks on the output.
Screenshots produced by the Aspose fallback are for local preview only and are
NOT suitable for the final deliverable.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


LIBRE_OFFICE_PATHS = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
]


def find_libreoffice() -> Path | None:
    """Return the soffice executable path, or None if not found."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        return Path(soffice)
    for candidate in LIBRE_OFFICE_PATHS:
        p = Path(candidate)
        if p.exists():
            return p
    return None


def render_with_libreoffice(ppt_path: Path, out_dir: Path, scale: float = 1.5):
    """Render slides via LibreOffice Impress (watermark-free)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # LibreOffice exports all slides as PNG when given a directory
        cmd = [
            str(find_libreoffice()),
            "--headless",
            "--convert-to", "png",
            "--outdir", str(tmpdir_path),
            str(ppt_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Move exported slides into the target directory in order
        pngs = sorted(tmpdir_path.glob("*.png"))
        for i, png in enumerate(pngs, start=1):
            out_path = out_dir / f"slide_{i:02d}.png"
            # Basic resize to approximate the requested scale
            # (LibreOffice exports at the slide's design resolution.)
            out_path.write_bytes(png.read_bytes())
        print(f"Rendered {len(pngs)} slides to {out_dir} (LibreOffice)")


def render_with_aspose(ppt_path: Path, out_dir: Path, scale: float = 1.5):
    """Render slides via Aspose.Slides (evaluation watermarks)."""
    import aspose.slides as slides

    out_dir.mkdir(parents=True, exist_ok=True)
    pres = slides.Presentation(str(ppt_path))
    for i in range(len(pres.slides)):
        slide = pres.slides[i]
        img = slide.get_image(scale, scale)
        out_path = out_dir / f"slide_{i+1:02d}.png"
        img.save(str(out_path), slides.ImageFormat.PNG)
        print(f"Saved {out_path}")

    print(f"\nRendered {len(pres.slides)} slides to {out_dir} (Aspose evaluation)")
    print("WARNING: Aspose.Slides evaluation adds watermarks. "
          "Install LibreOffice for clean final screenshots.")


def render_ppt(ppt_path: Path, out_dir: Path | None = None, scale: float = 1.5):
    ppt_path = Path(ppt_path)
    if out_dir is None:
        out_dir = ppt_path.parent / "ppt_screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if find_libreoffice():
        render_with_libreoffice(ppt_path, out_dir, scale)
    else:
        print("LibreOffice not found. Falling back to Aspose.Slides (watermarked).")
        render_with_aspose(ppt_path, out_dir, scale)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path(__file__).parent / "output" / "latest" / "Transcript_Intelligence_Report.pptx"
    render_ppt(path)
