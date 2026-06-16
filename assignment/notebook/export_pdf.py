"""
export_pdf.py

Convert the generated PowerPoint deck to a watermark-free PDF using LibreOffice
in headless mode.

Usage:
    python export_pdf.py

The script looks for a LibreOffice installation in this order:
1. Bundled LibreOffice Portable in .libreoffice_portable/extracted/
2. Any "soffice" binary on the system PATH

Output is written next to the PPTX in output/latest/.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from utils import NOTEBOOK_DIR, OUTPUT_DIR


def find_soffice() -> Path | None:
    """Locate the LibreOffice soffice binary."""
    # 1. Bundled portable copy
    bundled = (
        NOTEBOOK_DIR
        / ".libreoffice_portable"
        / "extracted"
        / "App"
        / "libreoffice"
        / "program"
        / "soffice.com"
    )
    if bundled.exists():
        return bundled

    # 2. System PATH
    soffice = shutil.which("soffice")
    if soffice:
        return Path(soffice)

    return None


def main() -> int:
    soffice = find_soffice()
    if soffice is None:
        print(
            "ERROR: LibreOffice not found.\n"
            "  Option 1: Download LibreOffice Portable and extract it to:\n"
            f"    {NOTEBOOK_DIR / '.libreoffice_portable' / 'extracted'}\n"
            "  Option 2: Install LibreOffice system-wide and make sure 'soffice' is on PATH.\n"
            "  Option 3: Open the PPTX in PowerPoint and export as PDF manually."
        )
        return 1

    pptx_path = OUTPUT_DIR / "Transcript_Intelligence_Report.pptx"
    if not pptx_path.exists():
        print(f"ERROR: PPTX not found at {pptx_path}. Run 06_generate_ppt.py first.")
        return 1

    outdir = pptx_path.parent
    print(f"Converting {pptx_path.name} to PDF...")
    print(f"Using LibreOffice: {soffice}")

    # Use a dedicated, writable user profile so the portable install can run
    # headlessly without relying on a pre-existing LibreOffice user directory.
    profile_dir = NOTEBOOK_DIR / ".libreoffice_portable" / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    user_install = "-env:UserInstallation=" + profile_dir.as_uri()

    # LibreOffice headless conversion
    cmd = [
        str(soffice),
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--norestore",
        "--convert-to", "pdf",
        "--outdir", str(outdir),
        str(pptx_path),
        user_install,
    ]
    env = os.environ.copy()
    # Prevent any GUI/autostart side effects
    env["LibreOffice_NoQuickstart"] = "1"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: LibreOffice conversion failed.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return result.returncode

    pdf_path = outdir / "Transcript_Intelligence_Report.pdf"
    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        print(f"PDF saved: {pdf_path} ({size_kb:.1f} KB)")
    else:
        print("WARNING: Conversion command succeeded but PDF file was not created.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
