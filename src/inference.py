from pathlib import Path

import torchaudio
from speechbrain.pretrained import EncoderClassifier

NAMES = ["Yan", "B", "C", "Bing", "A", "Xi", "Cheng", "Xiang", "Cong"]

model_dir = "CKPT+2022-10-25+01-04-11+00"
root_dir = Path(__file__).parent.parent.resolve()
classifier = EncoderClassifier.from_hparams(source=str(root_dir/f"results/speaker_id/1986/save/{model_dir}"), hparams_file=str(root_dir/"inference.yaml"), savedir=str(root_dir/"results/speaker_id"))
# Perform classification
wav_dir = root_dir / "data/wav"

audio_mapping_test = {
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
   "wx3.wav" : "Xi",
   "wx6.wav" : "Xi",
   "wx9.wav" : "Xi",
   "wsc3.wav" : "Cong",
   "wsc6.wav" : "Cong",
   "wsc9.wav" : "Cong",
   "zlb3.wav" : "Bing",
   "zlb6.wav" : "Bing",
   "zlb9.wav" : "Bing",
}

audio_mapping_train = {
   "fc1.wav" : "Cheng",
   "fc2.wav" : "Cheng",
   "fc4.wav" : "Cheng",
   "a1.wav" : "A",
   "a2.wav" : "A",
   "a4.wav" : "A",
   "b1.wav" : "B",
   "b2.wav" : "B",
   "b4.wav" : "B",
   "c1.wav" : "C",
   "c2.wav" : "C",
   "c4.wav" : "C",
   "py1.wav" : "Yan",
   "py2.wav" : "Yan",
   "py4.wav" : "Yan",
   "wx1.wav" : "Xi",
   "wx2.wav" : "Xi",
   "wx4.wav" : "Xi",
   "wsc1.wav" : "Cong",
   "wsc2.wav" : "Cong",
   "wsc4.wav" : "Cong",
   "zlb1.wav" : "Bing",
   "zlb2.wav" : "Bing",
   "zlb4.wav" : "Bing",
}

for name in NAMES:
    classifier.hparams.label_encoder.add_label(name)

for file, name in audio_mapping_train.items():
    audio_file = wav_dir / file
    signal, fs = torchaudio.load(audio_file) # test_speaker: 5789
    output_probs, score, index, text_lab = classifier.classify_batch(signal)
    print(f'-TRAIN- Audio:{file}, Speaker: {name}, Predicted: ' + text_lab[0])

for file, name in audio_mapping_test.items():
    audio_file = wav_dir / file
    signal, fs = torchaudio.load(audio_file) # test_speaker: 5789
    output_probs, score, index, text_lab = classifier.classify_batch(signal)
    print(f'-TEST- Audio:{file}, Speaker: {name}, Predicted: ' + text_lab[0])

