# Deep Music Classifier

## About

Project mostly based on [Dec 2016.] [Convolutional Recurrent Neural Networks for Music Classification](https://arxiv.org/abs/1609.04243) (Keunwoo Choi, George Fazekas, Mark Sandler, Kyunghyun Cho), i.e. using convolutional recurrent neural network deep model for multi-output classification task (tagging each music piece using a subset of labels).

## Input features

[Mel-spectrograms](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) are extracted from .mp3s and used as model inputs. An example of such a spectrogram is: ![Mel-spectrogram example](https://github.com/kristijanbartol/Deep-Music-Tagger/blob/master/out/mel-spec.png)

However, when generating images for out network, we save only the content inside a plot (and it takes a while):
```
/usr/bin/python3.6 /home/kristijan/FER/projekt/Deep-Music-Tagger/src/mel-spec.py
Extracting mel-spectrograms from raw data root directory...
/home/kristijan/FER/projekt/Deep-Music-Tagger/data/fma_medium/106/106864.mp3 (extracting)..................0/25002 (0.00%)
Saving spectrogram in ../out/mel-specs/106864.png
/home/kristijan/FER/projekt/Deep-Music-Tagger/data/fma_medium/106/106343.mp3 (extracting)..................1/25002 (0.00%)
Saving spectrogram in ../out/mel-specs/106343.png
/home/kristijan/FER/projekt/Deep-Music-Tagger/data/fma_medium/106/106870.mp3 (extracting)..................2/25002 (0.01%)
Saving spectrogram in ../out/mel-specs/106870.png
/home/kristijan/FER/projekt/Deep-Music-Tagger/data/fma_medium/106/106342.mp3 (extracting)..................3/25002 (0.01%)
Saving spectrogram in ../out/mel-specs/106342.png
/home/kristijan/FER/projekt/Deep-Music-Tagger/data/fma_medium/106/106630.mp3 (extracting)..................4/25002 (0.02%)
Saving spectrogram in ../out/mel-specs/106630.png
/home/kristijan/FER/projekt/Deep-Music-Tagger/data/fma_medium/106/106881.mp3 (extracting)..................5/25002 (0.02%)

(...)
```

## Data

Using [FMA dataset (A Dataset For Music Analysis)](https://github.com/mdeff/fma). It is a collection of freely available MP3s (under Creative Commons license) most convenient for research projects and (currently) only publicly available music dataset of a kind.

## Relevant literature

[CRNN for Music Classification](https://arxiv.org/abs/1609.04243)

[FMA: A Dataset For Music Analysis](https://arxiv.org/abs/1612.01840)

[Music Information Retrival (MIR, Downie)](http://www.music.mcgill.ca/~ich/classes/mumt611_08/downie_mir_arist37.pdf)

[A Tutorial on Deep Learning for Music Information Retrieval](https://arxiv.org/pdf/1709.04396.pdf)
