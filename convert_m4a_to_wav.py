from pathlib import Path

from pydub import AudioSegment

root_dir = Path(__file__).parent.resolve()
data_dir = root_dir / "data"

for path in data_dir.glob("**/*"):
    if path.suffix == ".m4a":
        print(f"Processing {path}")
        audio = AudioSegment.from_file(path)
        export_to = path.with_suffix(".wav")
        audio.export(export_to, format="wav")
