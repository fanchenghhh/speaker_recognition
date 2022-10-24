from pathlib import Path

import torchaudio
from speechbrain.pretrained import EncoderClassifier

root_dir = Path(__file__).parent.parent
classifier = EncoderClassifier.from_hparams(source=root_dir/"results/speaker_id/1986/save/CKPT+2022-10-14+11-54-51+00", hparams_file=root_dir/'inference.yaml', savedir=root_dir/"results/speaker_id/")

# Perform classification
audio_file = '2022-10-06__16-15-34-0805.flac'
signal, fs = torchaudio.load(audio_file) # test_speaker: 5789
classifier.hparams.label_encoder.add_label('Christabel')
classifier.hparams.label_encoder.add_label('Yan')
classifier.hparams.label_encoder.add_label('Lukas')
output_probs, score, index, text_lab = classifier.classify_batch(signal)
print('Speaker: Yan, Predicted: ' + text_lab[0])
