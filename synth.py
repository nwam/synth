import numpy as np
import scipy.io.wavfile

fs = 44100

def fourier_synth(a, phi, f_0=440, T=2, fs=44100):
    '''
    Generates Fourier Synth samples

    inputs
        a: list of amplitudes for each harmonic
        phi: list of phase shifts for each harmonic
        f_0: fundamental frequency
        T: length of the synth wave
        fs: sample rate

    output
        samples: samples of fourier synth (y axis)
        ts: the time ticks in seconds (x axis)
    '''
    assert(len(a) == len(phi))
    K = len(a)
    num_samples = int(fs*T)

    ts  = np.arange(num_samples)/fs # sample times in seconds

    samples = np.zeros(num_samples)

    for k in range(K):
        samples += a[k]*np.cos(2*np.pi*(k+1)*f_0*ts+phi[k])

    return samples / samples.max()


def functional_wave(a_function, phi_function, K=9, f_0=440, T=2, fs=44100):
    '''
    Generates a fourier synth based on functions for a and phi

    inputs
        a_function: function to calculate amplitude for each harmonic
            takes one parameter: harmonic number (e.g. 1,2,3,...)
        phi_function: function to calculate shift for each harmonic
            takes one parameter: harmonic number
        K: the number of harmonics to use in the fourier approximation
        f_0: fundamental frequency
        T: time in seconds of wave
        fs: sampling rate
    '''
    a_function = np.vectorize(a_function)
    phi_function = np.vectorize(phi_function)
    a = a_function(np.arange(1,K+1))
    phi = phi_function(np.arange(1,K+1))
    return fourier_synth(a, phi, f_0=f_0, T=T, fs=fs)


def sin_wave(f_0=440, T=2, fs=44100):
    ''' Generates a sin signal '''
    return fourier_synth([1],[-np.pi/2], f_0=f_0, T=T, fs=fs)


def square_wave(K=9, f_0=440, T=2, fs=44100):
    ''' Generates a square wave using K harmonics '''
    return functional_wave(
        lambda k: 1/k * (4/np.pi) if k%2==1 else 0,
        lambda k: np.pi/2,
        K=K, f_0=f_0, T=T, fs=fs)


def triangle_wave(K=9, f_0=440, T=2, fs=44100):
    ''' Generates a triangle wave using K harmonics '''
    return functional_wave(
        lambda k: 1/k**2 * (8/np.pi**2) if k%2==1 else 0,
        lambda k: 0,
        K=K, f_0=f_0, T=T, fs=fs)


def sawtooth_wave(K=9, f_0=440, T=2, fs=44100):
    ''' Generates a sawtooth wave using K harmonics '''
    return functional_wave(
        lambda k: 1/k * 2,
        lambda k: np.pi/2,
        K=K, f_0=f_0, T=T, fs=fs)


def decay(samples, a=1, tau=0.1, start=0, end=None, fs=44100):
    '''
    Adds an decay to samples

    input
        samples: the samples to alter
        a: amplitude constant
        tau: decay constant
        fs: sample rate

    output
        samples with decay
    '''

    num_samples = len(samples)
    if end is None:
        end = num_samples/fs

    start_sample = int(start*fs)
    end_sample = int(end*fs)

    ts  = np.arange(start_sample, end_sample)/fs # sample times in seconds
    decay = a*np.exp(-(ts-start)/tau)

    samples[start_sample:end_sample] = samples[start_sample:end_sample]*decay

    return samples

# TODO: attack and decay are bascially the same function
def attack(samples, a=1, tau=0.1, start=0, end=None, fs=44100):
    '''
    Adds an attack to samples

    input
        samples: the samples to alter
        a: amplitude constant
        tau: attack constant
        fs: sample rate

    output
        samples with attack
    '''

    num_samples = len(samples)
    if end is None:
        end = num_samples/fs

    start_sample = int(start*fs)
    end_sample = int(end*fs)

    ts  = np.arange(start_sample, end_sample)/fs # sample times in seconds
    attack = 1 - a*np.exp(-(ts-start)/tau)

    samples[start_sample:end_sample] = samples[start_sample:end_sample]*attack

    return samples


# TODO: amp_vibrato, attack and decay are bascially the same function
def amp_vibrato(samples, a=0.1, tau=10, start=0, end=None, fs=44100):
    '''
    Adds vibrato to amplitude

    input
        samples: the samples to alter
        a: vibrato amplitude
        tau: vibrato speed
        fs: sample rate

    output
        samples with amplitude vibrations
    '''

    num_samples = len(samples)
    if end is None:
        end = num_samples/fs

    start_sample = int(start*fs)
    end_sample = int(end*fs)

    ts  = np.arange(start_sample, end_sample)/fs # sample times in seconds
    attack = (1-a) + a*np.sin(2*np.pi*tau*ts)

    samples[start_sample:end_sample] = samples[start_sample:end_sample]*attack

    return samples

def white_noise(T, a=1, fs = 44100):
    '''
    Generates white noise

    input
        T: Time in seconds
        a: Maximum amplitude of white noise

    output
        White noise of time T
    '''

    N = T*fs
    return np.random.rand(N)*a


if __name__ == '__main__':
    # Generate a sin wave
    scipy.io.wavfile.write('examples/sin.wav', fs, sin_wave())

    # Generate a square wave
    scipy.io.wavfile.write('examples/square.wav', fs, square_wave(9))

    # Generate a triangle wave
    triangle_samples = triangle_wave(9)
    scipy.io.wavfile.write('examples/triangle.wav', fs, triangle_samples)

    # Generate a sawtooth wave
    scipy.io.wavfile.write('examples/sawtooth.wav', fs, sawtooth_wave(9))

    # Generate a wave with specified properties
    a = np.array([ .05, .15, .22, .22, .17, .10, .05, .02, .008])
    phi = np.array([0, np.pi/2, 0, -np.pi/2, 0, np.pi/2, 0, -np.pi/2, 0])
    spec_samples = fourier_synth(a, phi)
    scipy.io.wavfile.write('examples/somewave.wav', fs, spec_samples)

    # Make a triangle wave that decays 1
    tridecay1 = triangle_samples.copy()
    tridecay1 = decay(tridecay1, a=1, tau=0.1)
    scipy.io.wavfile.write('examples/tridecay.wav', fs, tridecay1)

    # Make a triangle wave with an envelope
    trienvelope = triangle_samples.copy()
    trienvelope = attack(trienvelope, a=1, tau=0.2, end=1)
    trienvelope = decay(trienvelope, a=1, tau=0.1, start=1)
    scipy.io.wavfile.write('examples/trivelope.wav', fs, trienvelope)

    trienvelope = amp_vibrato(trienvelope, a=0.2, start=0.2)
    scipy.io.wavfile.write('examples/trivelope_vibrate.wav', fs, trienvelope)

    # Add white noise to a sound
    noisy_trienvelope = trienvelope + white_noise(T=2, a=0.1)
    scipy.io.wavfile.write('examples/trivelope_vibrate_noise.wav', fs, noisy_trienvelope)
