# Deep Music Classifier

## Note

Beware that this is project is **in progress**, but is regurarly updated. If you urge to get some information, please feel free to contact me on *kristijan.bartol@gmail.com* or *kb47186@fer.hr*.

## About

The ideal goal of this project is to be able to say "This part of the song has the elements of jazz, progressive rock and a bit of grunge.". This could be possible to achieve defining the problem as multi-output classification.

Deep model is based on [Dec 2016.] [Convolutional Recurrent Neural Networks for Music Classification](https://arxiv.org/abs/1609.04243) (Keunwoo Choi, George Fazekas, Mark Sandler, Kyunghyun Cho) [1], i.e. using convolutional recurrent neural network deep model for multi-output classification task (tagging each music piece using a subset of labels).

## Prerequisite

To be able to run all parts of this project, you will need the following additional Python packages (recommended is Python 3.6):

- **keras** - build and train the high-level model
- **librosa** - extract mel-spectrograms
- **pandas** - analyze FMA metadata
- **numpy** - efficiently work with linear algebra operations
- **tensorflow** (GPU recommended) - modify keras backend
- **matplotlib** - plot various graphs and use it extract librosa spectrograms

## Input features

[Mel-spectrograms](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) are extracted from .mp3s and used as model inputs. An example of such a spectrogram is: ![Mel-spectrogram example](https://github.com/kristijanbartol/Deep-Music-Tagger/blob/master/out/plot.png)

However, when generating images for out network, we save only the content inside a plot (and it takes a while):
```
/usr/bin/python3.6 /home/kristijan/FER/projekt/Deep-Music-Tagger/src/mel-spec.py
Extracting mel-spectrograms from raw data root directory...
/home/kristijan/FER/projekt/Deep-Music-Tagger/data/fma_medium/106/106864.mp3 (extracting)..........0/25002 (0.00%)
Saving spectrogram in ../out/mel-specs/106864.png
/home/kristijan/FER/projekt/Deep-Music-Tagger/data/fma_medium/106/106343.mp3 (extracting)..........1/25002 (0.00%)
Saving spectrogram in ../out/mel-specs/106343.png
/home/kristijan/FER/projekt/Deep-Music-Tagger/data/fma_medium/106/106870.mp3 (extracting)..........2/25002 (0.01%)
Saving spectrogram in ../out/mel-specs/106870.png
/home/kristijan/FER/projekt/Deep-Music-Tagger/data/fma_medium/106/106342.mp3 (extracting)..........3/25002 (0.01%)
Saving spectrogram in ../out/mel-specs/106342.png
/home/kristijan/FER/projekt/Deep-Music-Tagger/data/fma_medium/106/106630.mp3 (extracting)..........4/25002 (0.02%)
Saving spectrogram in ../out/mel-specs/106630.png
/home/kristijan/FER/projekt/Deep-Music-Tagger/data/fma_medium/106/106881.mp3 (extracting)..........5/25002 (0.02%)

(...)
```

However, other spectrograms could also be used as described and compared in detail in [5]. In this work, except mel-spectrograms, raw audio input will also be tested [6].

## Data

Using [FMA dataset (A Dataset For Music Analysis)](https://github.com/mdeff/fma) [2]. It is a collection of freely available MP3s (under Creative Commons license) most convenient for research projects and (currently) only publicly available music dataset of a kind.

## Usage

...

## Relevant literature

[1] [CRNN for Music Classification](https://arxiv.org/abs/1609.04243)

[2] [FMA: A Dataset For Music Analysis](https://arxiv.org/abs/1612.01840)

[3] [Music Information Retrival (origin of "MIR", Downie)](http://www.music.mcgill.ca/~ich/classes/mumt611_08/downie_mir_arist37.pdf)

[4] [A Tutorial on Deep Learning for Music Information Retrieval](https://arxiv.org/pdf/1709.04396.pdf)

[5] [Comparison on Audio Signal Preprocessing Methods for Deep Neural Networks on Music Tagging](https://arxiv.org/pdf/1709.01922.pdf)

[6] [End-to-end learning for music audio tagging at scale (1D convolution)](https://arxiv.org/pdf/1711.02520.pdf)

For broader references on music information retrieval, check https://github.com/ybayle/awesome-deep-learning-music.
