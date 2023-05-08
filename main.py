import scipy as sp
import numpy as np


def main():
    samples = np.loadtxt('audio.txt', dtype=int)

    dts = np.diff(samples)

    data = []

    value = 20000  # so not full volume
    # recreate the original square wave
    for dt in dts:
        data.extend([value] * dt)
        value = -value

    # more or less correct
    target_samplerate = 44100
    factor = 23
    original_samplerate = target_samplerate * factor  # 1014300

    # keep exactly 10 seconds at 44100 * 23 Hz
    original = np.array(data[: original_samplerate * 10])
    sp.io.wavfile.write('out.org.wav', original_samplerate, original.astype(np.int16))

    # AppleWin: averages every 23 samples
    averaged = np.mean(original.reshape(-1, factor), axis=1)
    sp.io.wavfile.write('out.ave.wav', target_samplerate, averaged.astype(np.int16))

    # apply a low-pass filter at 22050Hz which is the Nyquist's frequency of the output
    nyquist = target_samplerate / 2
    sos = sp.signal.butter(10, nyquist, 'lowpass', fs=original_samplerate, output='sos')
    filtered_signal = sp.signal.sosfilt(sos, original)
    sp.io.wavfile.write('out.filt.wav', original_samplerate, filtered_signal.astype(np.int16))

    # downsample either the original or the low-passed version
    samples_to_use = original

    # downsample using scipy decimate
    decimated = sp.signal.decimate(samples_to_use, factor)
    sp.io.wavfile.write('out.deci.wav', target_samplerate, decimated.astype(np.int16))

    # downsample using scipy resample
    resampled = sp.signal.resample(samples_to_use, decimated.size)
    sp.io.wavfile.write('out.resamp.wav', target_samplerate, resampled.astype(np.int16))



if __name__ == '__main__':
    main()
