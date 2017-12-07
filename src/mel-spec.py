# Check full librosa spectrogram tutorial in the following IPython notebook:
# http://nbviewer.jupyter.org/github/bmcfee/librosa/blob/master/examples/LibROSA%20demo.ipynb

import librosa

import numpy as np
import scipy.misc
import os
import time
import argparse
from PIL import Image

music_dir = '../data/fma_medium/'      # directory where you extracted FMA dataset with .mp3s
spectr_dir = '../in/mel-specs/'
spectr_template = '../in/mel-specs/{}'
logs_file = '../out/logs/mel-spec.log'

regenerate = False


def __get_subdir(spectr_fname):
    """
    Creates subdirectory if it's not already created
    so that the structure is the same as in original music directory.

    :param spectr_fname:
    :return:
    """
    spectr_subdir = spectr_template.format(spectr_fname[:3] + '/')
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

    spectr_fname_h = (audio_fname + '_h.png')
    spectr_fname_p = (audio_fname + '_p.png')

    subdir_path = __get_subdir(audio_fname)

    scipy.misc.toimage(log_S_h).save(subdir_path.format(spectr_fname_h))
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
    y, sr = librosa.load(audio_fpath, sr=12000)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, hop_length=256, n_mels=96)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.logamplitude(S, ref_power=np.max)

    spectr_fname = audio_fname + '.png'
    subdir_path = __get_subdir(spectr_fname)

    print(type(log_S))
    print(log_S.shape)

    # Draw log values matrix in grayscale
    scipy.misc.toimage(log_S).save(subdir_path.format(spectr_fname))


def __unify_img_sizes(min_width, expected_width):
    deleted_cnt = 0
    failed_dlt_cnt = 0
    for subdir, _, files in os.walk(spectr_dir):
        for file in files:
            fpath = os.path.join(subdir, file)
            img = Image.open(fpath)
            width = img.size[0]
            height = img.size[1]
            # no use of problematic spectrograms much shorter than min_width
            if width < min_width:
                try:
                    print('DELETE | {} | {}x{} (width < min_width ({}))'
                          .format(fpath, height, width, min_width))
                    os.remove(fpath)
                    deleted_cnt += 1
                except:
                    print('Error occured while deleting mel-spec')
                    failed_dlt_cnt += 1
                continue
            elif width > expected_width:
                print('CROP | {} | {}x{} | width > expected_width ({})'
                      .format(fpath, height, width, expected_width))
                # crop to (height, expected_width) and remove third dimension (channel) to draw grayscale
                img_as_np = np.asarray(img.getdata()).reshape(height, width, -1)[:, :expected_width, :]\
                    .reshape(height, -1)
            elif width < expected_width:
                print('APPEND | {} | {}x{} | min_width ({}) < width < expected_width ({})'
                      .format(fpath, height, width, min_width, expected_width))
                img_as_np = np.asarray(img.getdata()).reshape(height, width, -1)
                # fill in image with black pixels up to (height, expected_width)
                img_as_np = np.hstack((img_as_np.reshape(height, -1), np.zeros((height, expected_width - width))))
            else:
                continue

            # Replace spectrograms
            os.remove(fpath)
            scipy.misc.toimage(img_as_np).save(fpath)

    return deleted_cnt, failed_dlt_cnt


def __get_times():
    current_time = time.time()
    op_time = current_time - op_start_time
    overall_time = current_time - start_time
    h, m, s = int(overall_time) // 3600, (int(overall_time) % 3600) // 60, overall_time % 60

    return op_time, h, m, s


def __log(outcome):
    i = ok_cnt + fail_cnt
    op_time, h, m, s = __get_times()
    logline = '{0} | {1:.2f} seconds | {2:02d}:{3:02d}:{4:02d} | {5} | {6}/{7} [{8:.2f}%]'\
        .format(outcome, op_time, h, m, int(s), fpath, i, nfiles, i / nfiles * 100)

    print(logline)
    flogs.write(logline + '\n')
    flogs.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regenerate', action='store_true', help='ovewrite existing spectrograms')
    args = parser.parse_args()

    if args.regenerate:
        regenerate = True

    start_time = time.time()

    flogs = open(logs_file, 'a')    # create a file without truncating it in case it exists
    nfiles = sum([len(files) for r, d, files in os.walk(music_dir)])
    ok_cnt = 0
    fail_cnt = 0

    for subdir, _, files in os.walk(music_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                fpath = os.path.join(subdir, file)
                fname, _ = os.path.splitext(file)         # (filename, extension)
                # check if spectrogram is already generated
                if not regenerate and os.path.isfile(spectr_template.format(fname[:3] + '/' + fname + '.png')):
                    continue
                op_start_time = time.time()
                try:
                    __extract_melspec(fpath, fname)
                except:
                    __log('FAILED')
                    fail_cnt += 1
                    continue
                ok_cnt += 1
                __log('OK')
            else:
                continue

    flogs.close()
    print('Generating spectrogram finished! Generated {}/{} images successfully'.format(ok_cnt, ok_cnt + fail_cnt))

    # aligning spectrograms to the same dimensions to feed convolutional input properly
    deleted, failed_dlt = __unify_img_sizes(1404, 1406)
    print('Finished alinging image sizes! Deleted problematic spectrograms: {}/{}'.format(deleted, deleted+failed_dlt))
