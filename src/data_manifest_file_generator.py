import json
import random
import re
from pathlib import Path

import yaml
from pydub import AudioSegment

root_dir = Path(__file__).parent.parent.resolve()
config_file = root_dir / "src/data_manifest_file_generator.yaml"

with open(config_file, "r", encoding="utf-8") as yaml_file:
    configs = yaml.safe_load(yaml_file)

data_dir = root_dir / configs["data_dir"]
export_dir = root_dir / configs["export_dir"]
test_ratio = configs["test_ratio"]
valid_ratio = configs["valid_ratio"]

test_audios = configs["test_audios"] if configs["test_audios"] else []

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

name_to_ids = dict()
for i, content in manifest_dict.items():
    name_to_ids[Path(content["wav"]).name] = i

num = speech_id
ids = list(range(num))

test_ids = [name_to_ids[name] for name in configs["test_audios"]]

if not configs["only_test_audios"]:
    test_sample_num = int(num * test_ratio) - len(test_ids)
    extra_test_ids = list(random.sample([id for id in ids if id not in test_ids], test_sample_num)) if test_sample_num > 0 else []
    test_ids.extend(extra_test_ids)

valid_ids = random.sample([id for id in ids if id not in test_ids], int(num * valid_ratio))
train_ids = [id for id in ids if id not in test_ids and id not in valid_ids]

train_export_path = export_dir / "train.json"
valid_export_path = export_dir / "valid.json"
test_export_path = export_dir / "test.json"

def dump_to_json(path, ids, manifest_dict):
    temp_dict = {id: manifest_dict[id] for id in ids}
    with open(path, "w") as f:
        json.dump(temp_dict, f, indent=4)

dump_to_json(train_export_path, train_ids, manifest_dict)
dump_to_json(valid_export_path, valid_ids, manifest_dict)
dump_to_json(test_export_path, test_ids, manifest_dict)
