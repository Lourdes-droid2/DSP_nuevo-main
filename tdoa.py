import numpy as np
from scipy.signal import correlate, correlation_lags
from numpy.fft import fft, ifft, fftshift
import time

def estimate_tdoa_cc(sig1, sig2, fs):
    start_time = time.perf_counter()
    sig1 = np.asarray(sig1).flatten()
    sig2 = np.asarray(sig2).flatten()
    if len(sig1) == 0 or len(sig2) == 0:
        end_time = time.perf_counter()
        return np.nan, end_time - start_time
    try:
        corr = correlate(sig1, sig2, mode='full')
        lags_samples = correlation_lags(len(sig1), len(sig2), mode='full')
    except Exception:
        end_time = time.perf_counter()
        return np.nan, end_time - start_time
    if len(lags_samples) == 0 or len(corr) != len(lags_samples):
        end_time = time.perf_counter()
        return np.nan, end_time - start_time
    lags_seconds = lags_samples / fs
    tdoa_idx = np.argmax(corr)
    tdoa = lags_seconds[tdoa_idx]
    end_time = time.perf_counter()
    return tdoa, end_time - start_time

def estimate_tdoa_gcc(sig1, sig2, fs, method='phat'):
    start_time = time.perf_counter()
    sig1 = np.asarray(sig1).flatten()
    sig2 = np.asarray(sig2).flatten()
    len_sig1, len_sig2 = len(sig1), len(sig2)
    if len_sig1 == 0 or len_sig2 == 0:
        end_time = time.perf_counter()
        return np.nan, end_time - start_time
    n = len_sig1 + len_sig2 - 1
    if n <= 0:
        end_time = time.perf_counter()
        return np.nan, end_time - start_time
    try:
        SIG1 = fft(sig1, n=n)
        SIG2 = fft(sig2, n=n)
    except Exception:
        end_time = time.perf_counter()
        return np.nan, end_time - start_time

    R = SIG1 * np.conj(SIG2)
    R_weighted = R

    if method.lower() == 'phat':
        R_abs = np.abs(R)
        if np.all(R_abs < 1e-12):
            R_weighted = R
        else:
            R_weighted = R / (R_abs + 1e-10)
    elif method.lower() == 'scot':
        G11 = np.abs(SIG1)**2
        G22 = np.abs(SIG2)**2
        den_scot = np.sqrt(G11 * G22 + 1e-10)
        if np.all(den_scot < 1e-12):
            R_weighted = R
        else:
            R_weighted = R / (den_scot + 1e-10)
    elif method.lower() == 'ml':
        G11 = np.abs(SIG1)**2
        G22 = np.abs(SIG2)**2
        abs_R_sq = np.abs(R)**2
        denominator_coherence = G11 * G22
        coherence_sq = np.zeros_like(abs_R_sq)
        valid_coh_indices = denominator_coherence > 1e-12
        coherence_sq[valid_coh_indices] = abs_R_sq[valid_coh_indices] / denominator_coherence[valid_coh_indices]
        coherence_sq = np.clip(coherence_sq, 0.0, 1.0 - 1e-7)
        Psi_ML_weight = coherence_sq / (1.0 - coherence_sq + 1e-10)
        R_weighted = R * Psi_ML_weight
    else:
        end_time = time.perf_counter()
        raise ValueError("Método GCC no reconocido. Use 'phat', 'scot' o 'ml'.")
    try:
        cc = fftshift(ifft(R_weighted).real)
    except Exception:
        end_time = time.perf_counter()
        return np.nan, end_time - start_time
    if len(cc) == 0:
        end_time = time.perf_counter()
        return np.nan, end_time - start_time
    lags_vector = correlation_lags(len_sig1, len_sig2, mode='full') / fs
    tdoa_index = np.argmax(cc)
    tdoa = lags_vector[tdoa_index]
    end_time = time.perf_counter()
    return tdoa, end_time - start_time