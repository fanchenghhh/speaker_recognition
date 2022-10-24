from pathlib import Path

import torchaudio
from speechbrain.pretrained import EncoderClassifier

NAMES = ["Yan", "B", "C", "Bing", "A", "Xi", "Cheng", "Xiang", "Cong"]

root_dir = Path(__file__).parent.parent.resolve()
classifier = EncoderClassifier.from_hparams(source=str(root_dir/"results/speaker_id/1986/save/CKPT+2022-10-24+23-28-06+00"), hparams_file=str(root_dir/"inference.yaml"), savedir=str(root_dir/"results/speaker_id"))
# Perform classification
wav_dir = root_dir / "data/wav"

audio_mapping = {
   "fc3.wav" : "Cheng",
   "fc6.wav" : "Cheng",
   "fc9.wav" : "Cheng",
   "a3.wav" : "A",
   "a6.wav" : "A",
   "a9.wav" : "A",
   "b3.wav" : "B",
   "b6.wav" : "B",
   "b9.wav" : "B",
   "c3.wav" : "C",
   "c6.wav" : "C",
   "c9.wav" : "C",
   "py3.wav" : "Yan",
   "py6.wav" : "Yan",
   "py9.wav" : "Yan",
   "py1.wav" : "Yan",
   "py2.wav" : "Yan",
   "py4.wav" : "Yan",
   "a1.wav" : "A",
   "a2.wav" : "A",
   "a4.wav" : "A",
   "a10.wav" : "A",
   "b1.wav" : "B",
   "b2.wav" : "B",
   "b4.wav" : "B",
   "b5.wav" : "B",
   "b7.wav" : "B",
}


for name in NAMES:
    classifier.hparams.label_encoder.add_label(name)

for file, name in audio_mapping.items():
    audio_file = wav_dir / file
    signal, fs = torchaudio.load(audio_file) # test_speaker: 5789
    output_probs, score, index, text_lab = classifier.classify_batch(signal)
    print(f'Audio:{file}, Speaker: {name}, Predicted: ' + text_lab[0])
