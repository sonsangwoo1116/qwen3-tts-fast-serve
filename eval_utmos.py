import torch, os, sys
import soundfile as sf
import numpy as np
import librosa

predictor = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)
predictor.eval()

data_dir = sys.argv[1] if len(sys.argv) > 1 else '/data_processed'

files = sorted([f for f in os.listdir(data_dir) if f.endswith('.wav')])
voices = sorted(set(f.rsplit('_text', 1)[0] for f in files))

print("Voice                    Avg    Min    Max    Std")
print("-" * 50)

all_scores = []
for voice in voices:
    scores = []
    for i in range(1, 11):
        fp = os.path.join(data_dir, '%s_text%02d.wav' % (voice, i))
        if not os.path.exists(fp):
            continue
        d, sr = sf.read(fp)
        if sr != 16000:
            d = librosa.resample(d, orig_sr=sr, target_sr=16000)
        s = predictor(torch.from_numpy(d).float().unsqueeze(0), sr=16000).item()
        scores.append(s)
    if scores:
        all_scores.extend(scores)
        print("%-22s %6.3f %6.3f %6.3f %6.3f" % (voice, np.mean(scores), min(scores), max(scores), np.std(scores)))

print("-" * 50)
if all_scores:
    print("%-22s %6.3f %6.3f %6.3f %6.3f" % ("OVERALL", np.mean(all_scores), min(all_scores), max(all_scores), np.std(all_scores)))
