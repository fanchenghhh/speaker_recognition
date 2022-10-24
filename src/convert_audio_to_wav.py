import argparse
from pathlib import Path

from pydub import AudioSegment

parser = argparse.ArgumentParser(
    description="Convert audio file to wav format.")
parser.add_argument("-f", "--format", required=True,
                    help="Original audio format.")

args = parser.parse_args()

root_dir = Path(__file__).parent.parent.resolve()
data_dir = root_dir / "data"

original_dir = data_dir / args.format
target_dir = data_dir / "wav"

for path in original_dir.glob("**/*"):
    if path.suffix == f".{args.format}":
        print(f"Processing {path}")
        audio = AudioSegment.from_file(path)
        export_to = target_dir / path.name
        export_to = export_to.with_suffix(".wav")
        audio.export(export_to, format="wav")
