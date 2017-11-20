# Check full librosa spectrogram tutorial in the following IPython notebook:
# http://nbviewer.jupyter.org/github/bmcfee/librosa/blob/master/examples/LibROSA%20demo.ipynb

import librosa
from librosa import display
import matplotlib.pyplot as plt

import numpy as np
import warnings
import os

# Root directory where you downloaded FMA dataset with .mp3s
rootdir = '/home/kristijan/FER/projekt/Deep-Music-Tagger/data/fma_medium'

# Surpress UserWarnings from matplotlib; they occur as we are saving only plots' content, not axes and borders
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def generate_plot(fname, log_S, sr):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(24, 8)

    # Fill the whole plot
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.tight_layout()

    fig.savefig('../out/mel-specs/{}'.format(fname.replace('.mp3', '.png')), bbox_inches='tight')


def extract_melspec(audio_fpath, audio_fname):
    # Load sound file
    y, sr = librosa.load(audio_fpath)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.logamplitude(S, ref_power=np.max)

    generate_plot(audio_fname, log_S, sr)


i = 0
nfiles = sum([len(files) for r, d, files in os.walk(rootdir)])

print('Extracting mel-spectrograms from raw data root directory...')
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.lower().endswith('.mp3'):
            fpath = os.path.join(subdir, file)
            print('{0}........................................................{1:.2f}%'.format(fpath, i / nfiles * 100))
            extract_melspec(fpath, file)
            i += 1
            if i == 100:
                # Generate a 100 samples for testing purposes; they still need to be labeled
                exit(0)
        else:
            continue
