from random import random
import torch, torchaudio


def random_angle(min_f=20, max_f=8000, sr=24000):
    min_f = torch.log(min_f)
    max_f = torch.log(max_f)
    rand = torch.exp(random() * (max_f - min_f) + min_f)
    rand = 2 * torch.pi * rand / sr
    return rand

def pole_to_z_filter(omega, amplitude=.9):
    z0 = amplitude * torch.exp(1j * omega)
    a = [1, -2 * torch.real(z0), abs(z0)**2]
    b = [abs(z0)**2, -2 * torch.real(z0), 1]
    return b, a

def random_phase_mangle(x, min_f, max_f, amp, sr):
    angle = random_angle(min_f, max_f, sr)
    b, a = pole_to_z_filter(angle, amp)
    return torchaudio.functional.lfilter(b, a, x)

def get_derivator(sr: int):
    alpha = 1 / (1 + 1 / sr * 2 * torch.pi * 10)
    derivator = ([.5, -.5], [1])
    return lambda x: torchaudio.functional.lfilter(*derivator, x) 

def get_integrator(sr: int):
    alpha = 1 / (1 + 1 / sr * 2 * torch.pi * 10)
    integrator = ([alpha**2, -alpha**2], [1, -2 * alpha, alpha**2])
    return lambda x: torchaudio.functional.lfilter(*integrator, x)

def normalize_signal(x, max_gain_db: int = 30):
    peak = x.abs().amax()
    if peak == 0: return x

    log_peak = 20 * torch.log10(peak)
    log_gain = min(max_gain_db, -log_peak)
    gain = 10**(log_gain / 20)
    return x * gain
