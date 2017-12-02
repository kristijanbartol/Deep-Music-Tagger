# Check full librosa spectrogram tutorial in the following IPython notebook:
# http://nbviewer.jupyter.org/github/bmcfee/librosa/blob/master/examples/LibROSA%20demo.ipynb

import librosa

import numpy as np
import scipy.misc
import os
import time

music_dir = '../data/fma_medium'      # directory where you extracted FMA dataset with .mp3s
spectr_path = '../in/mel-specs/{}'
logs_file = '../out/logs/mel-spec.log'


def __get_subdir(spectr_fname):
    """
    Creates subdirectory if it's not already created
    so that the structure is the same as in original music directory.

    :param spectr_fname:
    :return:
    """
    spectr_subdir = spectr_path.format(spectr_fname[:3] + '/')
    if not os.path.exists(spectr_subdir):
        os.makedirs(spectr_subdir)
    return spectr_subdir + '{}'


def __extract_hpss_melspec(audio_fpath, audio_fname):
    """
    Extension of :func:`__extract_melspec`.
    Not used as it's about ten times slower, but
    if you have resources, try it out.

    :param audio_fpath:
    :param audio_fname:
    :return:
    """
    y, sr = librosa.load(audio_fpath, sr=44100)

    # Harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    S_h = librosa.feature.melspectrogram(y_harmonic, sr=sr, n_mels=128)
    S_p = librosa.feature.melspectrogram(y_percussive, sr=sr, n_mels=128)

    log_S_h = librosa.logamplitude(S_h, ref_power=np.max)
    log_S_p = librosa.logamplitude(S_p, ref_power=np.max)

    audio_filename, _ = os.path.splitext(audio_fname)

    spectr_fname_h = (audio_filename + '_h.png')
    spectr_fname_p = (audio_filename + '_p.png')

    subdir_path = __get_subdir(audio_fname)

    print('Saving harmonic spectrogram in {}'.format(subdir_path.format(spectr_fname_h)))
    scipy.misc.toimage(log_S_h).save(subdir_path.format(spectr_fname_h))
    print('Saving percussive spectrogram in {}'.format(subdir_path.format(spectr_fname_p)))
    scipy.misc.toimage(log_S_p).save(subdir_path.format(spectr_fname_p))


def __extract_melspec(audio_fpath, audio_fname):
    """
    Using librosa to calculate log mel spectrogram values
    and scipy.misc to draw and store them (in grayscale).

    :param audio_fpath:
    :param audio_fname:
    :return:
    """
    # Load sound file
    y, sr = librosa.load(audio_fpath, sr=44100)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.logamplitude(S, ref_power=np.max)

    spectr_fname = audio_fname.replace('.mp3', '.png')
    subdir_path = __get_subdir(spectr_fname)

    # Draw log values matrix in grayscale
    print('Saving harmonic spectrogram in {}'.format(subdir_path.format(spectr_fname)))
    scipy.misc.toimage(log_S).save(subdir_path.format(spectr_fname))


def __get_times():
    current_time = time.time()
    op_time = current_time - op_start_time
    overall_time = current_time - start_time
    h, m, s = int(overall_time) // 3600, (int(overall_time) % 3600) // 60, overall_time % 60

    return op_time, h, m, s


def __log(outcome):
    op_time, h, m, s = __get_times()
    logline = '{0} | {1:.2f} seconds | {2:02d}:{3:02d}:{4:02d} | {5} | {6}/{7} [{8:.2f}%]'\
        .format(outcome, op_time, h, m, int(s), fpath, i, nfiles, i / nfiles * 100)

    print(logline)
    flogs.write(logline + '\n')


if __name__ == '__main__':
    start_time = time.time()

    flogs = open(logs_file, 'a')    # create a file without truncating it in case it exists
    nfiles = sum([len(files) for r, d, files in os.walk(music_dir)])
    i = 0

    for subdir, dirs, files in os.walk(music_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                fpath = os.path.join(subdir, file)
                op_start_time = time.time()
                try:
                    __extract_melspec(fpath, file)
                except:
                    __log('FAILED')
                    continue
                __log('OK')
                i += 1
            else:
                continue

    flogs.close()
