import numpy as np
from scipy.signal import correlate, correlation_lags
from numpy.fft import fft, ifft, fftshift # Usar numpy.fft consistentemente

def estimate_tdoa_cc(sig1, sig2, fs):
    """
    Estima el TDOA entre dos señales usando correlación cruzada clásica.
    Devuelve el TDOA en segundos.
    """
    # Asegurarse de que las señales sean 1D arrays
    sig1 = np.asarray(sig1).flatten()
    sig2 = np.asarray(sig2).flatten()

    corr = correlate(sig1, sig2, mode='full')
    # correlation_lags devuelve los lags en muestras, dividimos por fs para obtener segundos
    lags_samples = correlation_lags(len(sig1), len(sig2), mode='full')
    lags_seconds = lags_samples / fs
    tdoa = lags_seconds[np.argmax(corr)]
    return tdoa

def estimate_tdoa_gcc(sig1, sig2, fs, method='phat'):
    """
    Estima el TDOA entre dos señales usando Generalized Cross-Correlation (GCC).
    Permite los métodos PHAT (Phase Transform) o SCOT (Smoothed Coherence Transform).
    Devuelve el TDOA estimado en segundos.
    """
    # Asegurarse de que las señales sean 1D arrays
    sig1 = np.asarray(sig1).flatten()
    sig2 = np.asarray(sig2).flatten()

    # Longitud para la FFT, asegurando que sea suficiente para la correlación lineal
    n = len(sig1) + len(sig2) - 1

    SIG1 = fft(sig1, n=n)
    SIG2 = fft(sig2, n=n)

    R = SIG1 * np.conj(SIG2)

    if method.lower() == 'phat':
        # Transformada de Fase: normaliza por la magnitud del espectro cruzado
        R_abs = np.abs(R)
        R = R / (R_abs + 1e-10) # Añadir epsilon para evitar división por cero
    elif method.lower() == 'scot':
        # SCOT: normaliza por la raíz cuadrada del producto de las auto-potencias espectrales
        P1 = np.abs(SIG1)**2
        P2 = np.abs(SIG2)**2
        den = np.sqrt(P1 * P2)
        R = R / (den + 1e-10) # Añadir epsilon
    else:
        raise ValueError("Método GCC no reconocido. Use 'phat' o 'scot'.")

    # Correlación cruzada en el dominio del tiempo
    cc = fftshift(ifft(R).real) # Tomar la parte real y aplicar fftshift para centrar el lag cero

    # Crear el vector de lags en segundos
    # El resultado de ifft(R) tiene longitud n. fftshift lo centra.
    # Los lags van de -n/2 a n/2 (aproximadamente)
    lags_samples = np.arange(-n//2, n//2 + n%2) # Ajuste para n par/impar si es necesario, pero linspace es más robusto
    if n % 2 == 0: # n es par, el centro no es un único punto
        lags_samples = np.arange(-n//2, n//2)
    else: # n es impar, el centro es 0
        lags_samples = np.arange(-(n-1)//2, (n-1)//2 + 1)

    # Usar np.linspace es generalmente más seguro para generar lags simétricos para fftshift
    # El rango de lags después de fftshift es de aproximadamente - (n/2)/fs a + (n/2)/fs
    # Para n impar, lags_samples va de -(n-1)/2 a (n-1)/2.
    # Para n par, lags_samples va de -n/2 a n/2 - 1.
    # np.fft.fftfreq(n, d=1/fs) genera frecuencias, np.fft.fftshift las centra.
    # Los lags correspondientes al resultado de ifft(fft(sig1)*conj(fft(sig2)))
    # y luego fftshift, si n es la longitud de la fft:
    # lags = np.arange(-n // 2, n - n // 2) / fs # Esto no es del todo correcto con fftshift
    # La forma correcta con fftshift es que el índice 0 corresponde a lag 0 (DC),
    # luego vienen los lags positivos, y luego los negativos envueltos.
    # fftshift reordena esto a [-fs/2, fs/2] (en términos de frecuencia).
    # Para lags de tiempo, el centro de `cc` (después de fftshift) es el lag cero.

    # Generar lags de forma que el centro del array 'cc' corresponda al lag 0.
    # Si n es la longitud de la FFT, cc también tiene longitud n.
    # El índice central de cc (después de fftshift) es n // 2.
    # Este índice central debe corresponder a un lag de 0.
    # Los índices de 0 a n//2 -1 son lags negativos.
    # Los índices de n//2 + 1 a n-1 son lags positivos.

    # Ejemplo de lags para fftshift:
    # Si N es la longitud de la señal (después de padding para FFT)
    # lags_indices = np.arange(N) - N // 2
    # lags_sec = lags_indices / fs
    # Esta es la forma más común de generar lags para un resultado de IFFT que ha sido `fftshift`eado.
    lags_vector = (np.arange(n) - n // 2) / fs


    tdoa_index = np.argmax(cc)
    tdoa = lags_vector[tdoa_index]

    return tdoa