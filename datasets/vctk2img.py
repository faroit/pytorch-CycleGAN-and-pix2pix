from __future__ import print_function
import numpy as np
import soundfile as sf
import os
import os.path as op
import glob
import re
import itertools
import random
import errno
from random import randint
import pandas as pd
import resampy
import norbert
import webrtcvad
from scipy import interpolate
np.random.seed(42)
random.seed(42)


def audio2image(
    audio,
    samplerate,
    n_fft=1024,
):

    audio = np.atleast_2d(audio).T
    # set up modules
    tf = norbert.TF(n_fft=n_fft, n_hopsize=256)
    ls = norbert.LogScaler()
    qt = norbert.Quantizer()

    # complex spectrogram
    Xc = tf.transform(audio)
    # log scale
    Xs = ls.scale(Xc)
    # quantize to 8bit
    Xq = qt.quantize(Xs)
    # write as jpg image and save bounds values
    return Xq


def get_sentences(df, speaker):
    query = df[df['speakerid'] == str(speaker)].sort_values(['sentence'])
    return query['file'].tolist()

def create_dataset(
    d,
    root_dir="vctk/raw/VCTK-Corpus",
    nb_srcs=1,
    target_rate=16000,
    augmentation=None,
    track_overlap=0,
):

    """Parsing VCTK file structure.

    wav48/p225/p225_001.wav
          _---__---_---_
            |    |   |
            |    |   `--- Sentence ID
            |    `------- Speaker ID
            `-----------/
    """

    regex = re.compile(
        "(?P<basepath>.*)\/(?P<speakerid>.*)\/(?P<sentence>.*)\.(?P<ext>\w+)"
    )

    max_frames = 512
    hop_size = 256

    data = []
    vctk_files = glob.glob(op.join(root_dir, 'wav48/*/*.wav'))

    item_offset = 300

    for f in vctk_files:
        match = regex.match(f)
        if match:
            m = match.groupdict()
            m['file'] = f
            m['set'] = d

            if int(m['speakerid'][1:]) < item_offset and d == 'test':
                continue
            elif int(m['speakerid'][1:]) > item_offset and d == 'train':
                continue
            else:
                data.append(m)

    df = pd.DataFrame(data)

    speakers = df.groupby("speakerid").groups.keys()

    item_count = 0

    A_path = d + "A"
    B_path = d + "B"
    if not os.path.isdir(A_path):
        os.makedirs(A_path)
        os.makedirs(B_path)

    for comb in groups(1, speakers):
        tracks = []
        for speaker in comb:
            speaker_audio = []

            sentence_paths = get_sentences(df, speaker)
            # shuffle the sentences
            random.shuffle(sentence_paths)

            speaker_audio = []
            for filename in sentence_paths:
                audio, rate = sf.read(filename)
                audio = resampy.resample(audio, rate, target_rate)
                rate = target_rate
                try:
                    # detect silence in beginning and end of recording
                    audio = remove_silence(audio, rate)
                    # append audio
                    speaker_audio.append(audio)
                except IndexError:
                    print("no active speech found")

                # stop if we have enough material to fill max_len
                if np.concatenate(speaker_audio).shape[0] > rate * (max_frames/rate * hop_size):
                    break

            speaker_audio = np.concatenate(speaker_audio)
            tracks.append(
                {
                    'id': str(speaker),
                    'audio': speaker_audio,
                    'rate': rate
                }
            )

        audio_mixture = mix(tracks)

        # compute features
        A = audio2image(audio_mixture, rate)
        A = A[:512, :512, :]
        B = np.copy(A)
        B[:, 128:, :] = 0
        im = norbert.Coder(format='jpg', quality=85)

        im.encode(A, os.path.join(A_path, str(item_count) + "_A.jpg"))
        im.encode(B, os.path.join(B_path, str(item_count) + "_B.jpg"))
        print(item_count)
        if item_count > 100:
            break
        item_count += 1

def mix(tracks):
    """Mix Audio to a minimum length of the audio files or max_len."""

    # all should have same length, so just add them
    audio_list = []
    for track in tracks:
        audio_list.append(track['audio'])

    audio_matrix = np.array(audio_list)
    out = np.sum(audio_matrix, axis=0)

    if np.any(out > 1.0):
        print("clipping!")

    return out


def remove_silence(audio, rate):
    activation = vad(audio, rate)
    activation = interp(activation, len(audio)).astype(np.bool)
    active_elements = np.nonzero(activation)[0]
    start = active_elements[0]
    end = active_elements[-1]
    return audio[start:end]


def groups(n, speakers):
    """Returns a random combination of n speakers"""
    while True:
        # make sure the speakers are unique in each sample
        yield random_combination(speakers, n)


def interp(in_array, out_len, interpolation_type='nearest'):
    ''' stretch input array by output '''
    ip1d = interpolate.interp1d(
        np.arange(in_array.shape[0]),
        in_array,
        kind=interpolation_type
    )
    return ip1d(
        np.linspace(0, in_array.shape[0] - 1, out_len)
    ).astype(in_array.dtype)


def vad(input_audio, rate, sensitivity=1):
    # convert to 16bit signed int
    audio = np.int16(input_audio * 32767)
    vad = webrtcvad.Vad()

    # mode 3 (max=3) means, very sensitive regarding to non-speech
    vad.set_mode(sensitivity)

    # window size 10ms
    n = int(rate * 0.01)

    # window without overlap
    chunks = list(
        audio[pos:pos + n] for pos in range(0, len(audio), n)
    )
    if len(chunks[-1]) != n:
        chunks = chunks[:-1]

    voiced = []
    for chunk in chunks:
        voiced.append(vad.is_speech(chunk.tobytes(), rate))

    return np.array(voiced)


def prepare(
    root="vctk",
    downsample=True,
    transform=None,
    target_transform=None,
    download=False,
    dev_mode=True
):
    raw_folder = 'raw'
    url = 'http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz'
    dset_path = 'VCTK-Corpus'

    def _check_exists():
        return os.path.exists(os.path.join(root, raw_folder, dset_path))

    """Download the VCTK data if it doesn't exist in processed_folder already."""
    from six.moves import urllib
    import tarfile

    if _check_exists():
        return

    raw_abs_dir = os.path.join(root, raw_folder)
    dset_abs_path = os.path.join(
        root, raw_folder, dset_path)

    # download files
    try:
        os.makedirs(os.path.join(root, raw_folder))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    print('Downloading ' + url)
    filename = url.rpartition('/')[2]
    file_path = os.path.join(root, raw_folder, filename)
    if not os.path.isfile(file_path):
        urllib.request.urlretrieve(url, file_path)
    if not os.path.exists(dset_abs_path):
        with tarfile.open(file_path) as zip_f:
            zip_f.extractall(raw_abs_dir)
    else:
        print("Using existing raw folder")


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def random_product(*args, **kwds):
    "Random selection from itertools.product(*args, **kwds)"
    pools = map(tuple, args) * kwds.get('repeat', 1)
    return tuple(random.choice(pool) for pool in pools)


if __name__ == '__main__':
    prepare()
    create_dataset(d="test")
    create_dataset(d="train")
