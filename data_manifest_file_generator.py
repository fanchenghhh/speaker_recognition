import json
import random
import re
from pathlib import Path

from pydub import AudioSegment

root_dir = Path(__file__).parent.resolve() / "data"
data_dir = root_dir / "wav"


data = dict()
for path in data_dir.glob("**/*"):
    if path.suffix == ".wav":
        pattern = r"([A-z]+)\d+"
        try:
            found = re.search(pattern, path.name).group(1)
            if found in data.keys():
                data[found].append(path.absolute())
            else:
                data[found] = [path.absolute()]
        except AttributeError:
            print(f"Error happened at {path}")
            print("Please check the data file name. They should be concatenation of characters and numbers, e.g. fc1.m4a")

manifest_dict = dict()

speakers = data.keys()
speaker_to_ids = dict()
for i, speaker in enumerate(speakers):
    speaker_to_ids[speaker] = i

speech_id = 0
for speaker, speech_paths in data.items():
    speaker_id = speaker_to_ids[speaker]

    for speech_path in speech_paths:
        audio = AudioSegment.from_wav(speech_path)
        content = {
            "wav": str(speech_path),
            "length": audio.duration_seconds,
            "spk_id": speaker_id,
        }
        manifest_dict[speech_id] = content
        speech_id += 1


num = speech_id
test_ratio = 0.2
dev_ratio = 0.2

ids = list(range(num))

test_ids = random.sample(ids, int(num * test_ratio))
dev_ids = random.sample(
    [id for id in ids if id not in test_ids], int(num * dev_ratio))
train_ids = [id for id in ids if id not in test_ids and id not in dev_ids]

train_dict = {id: manifest_dict[id] for id in train_ids}
dev_dict = {id: manifest_dict[id] for id in dev_ids}
test_dict = {id: manifest_dict[id] for id in test_ids}

train_export_path = root_dir / "manifest/train.json"
dev_export_path = root_dir / "manifest/valid.json"
test_export_path = root_dir / "manifest/test.json"

with open(train_export_path, "w") as f:
    json.dump(train_dict, f, indent=4)
with open(dev_export_path, "w") as f:
    json.dump(dev_dict, f, indent=4)
with open(test_export_path, "w") as f:
    json.dump(test_dict, f, indent=4)
