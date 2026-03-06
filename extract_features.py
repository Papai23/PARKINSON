import parselmouth
from parselmouth.praat import call
import numpy as np
import librosa
import nolds

def extract_features(audio):

    sound = parselmouth.Sound(audio)

    y, sr = librosa.load(audio, sr=44100)
    pitch = call(sound, "To Pitch", 0.0, 75, 500)

    Fo = call(pitch, "Get mean", 0, 0, "Hertz")
    Fhi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    Flo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 500)

    jitter = call(pointProcess, "Get jitter (local)",0,0,0.0001,0.02,1.3)

    shimmer = call([sound,pointProcess],"Get shimmer (local)",0,0,0.0001,0.02,1.3,1.6)

    harmonicity = call(sound, "To Harmonicity (cc)",0.01,75,0.1,1.0)

    hnr = call(harmonicity,"Get mean",0,0)


    rpde = nolds.sampen(y)
    dfa = nolds.dfa(y)

    spread1 = np.std(y)
    spread2 = np.var(y)

    d2 = nolds.corr_dim(y,10)
    pitch_values = librosa.yin(y, fmin=75, fmax=500)
    ppe = np.std(np.diff(y))
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 500)

    jitter_percent = call(pointProcess, "Get jitter (local)", 0,0,0.0001,0.02,1.3)

    jitter_abs = call(pointProcess, "Get jitter (local, absolute)", 0,0,0.0001,0.02,1.3)

    rap = call(pointProcess, "Get jitter (rap)", 0,0,0.0001,0.02,1.3)

    ppq = call(pointProcess, "Get jitter (ppq5)", 0,0,0.0001,0.02,1.3)

    ddp = 3 * rap
    shimmer = call([sound, pointProcess], "Get shimmer (local)",0,0,0.0001,0.02,1.3,1.6)

    shimmer_db = call([sound, pointProcess], "Get shimmer (local_dB)",0,0,0.0001,0.02,1.3,1.6)

    apq3 = call([sound, pointProcess], "Get shimmer (apq3)",0,0,0.0001,0.02,1.3,1.6)

    apq5 = call([sound, pointProcess], "Get shimmer (apq5)",0,0,0.0001,0.02,1.3,1.6)

    apq = call([sound, pointProcess], "Get shimmer (apq11)",0,0,0.0001,0.02,1.3,1.6)

    dda = 3 * apq3
    
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

    hnr = call(harmonicity, "Get mean", 0, 0)

    nhr = 1 / (hnr + 1e-6)
    features = [Fo, Fhi, Flo,
    jitter_percent,
    jitter_abs,
    rap,
    ppq,
    ddp,
    shimmer,
    shimmer_db,
    apq3,
    apq5,
    apq,
    dda,
    nhr,
    hnr,
    rpde,
    dfa,
    spread1,
    spread2,
    d2,
    ppe
]
    return np.array(features)