import numpy as np

C = 343  # Velocidad del sonido en m/s

def estimate_doa_from_tdoa(tdoa, d=0.1, c=C):
    """
    Estima el ángulo de llegada (DOA) en grados a partir del TDOA entre un par de micrófonos.
    Asume un modelo de campo lejano y que el TDOA ya ha sido calculado.
    El ángulo se calcula con respecto a la normal del eje del par de micrófonos.
    Si los micrófonos están en el eje x, arccos(c*tdoa/d) da el ángulo con el eje x.
    La interpretación del ángulo (azimut, elevación) depende de la orientación del par.
    Para un par en el eje x, este sería el ángulo con el eje x (o su complemento con el eje y).
    La descripción original decía "ángulo de elevación ... array lineal sobre eje x",
    interpretaremos theta_rad como el ángulo con la normal (broadside) del array.
    Si d es la separación, y theta es el ángulo con la normal, entonces path_diff = d * sin(theta).
    tdoa = path_diff / c = d * sin(theta) / c.
    sin(theta) = tdoa * c / d.
    theta = arcsin(tdoa * c / d).

    Si la fórmula original usaba arccos, implica que theta era el ángulo con el eje del array.
    cos(theta_axis) = tdoa * c / d.
    Vamos a seguir la fórmula original proporcionada por el usuario (con arccos),
    asumiendo que 'theta_rad' es el ángulo con el eje del par de micrófonos.

    Parameters:
    tdoa (float): Diferencia de tiempo de llegada en segundos.
    d (float): Distancia entre los dos micrófonos en metros (por defecto 0.1m).
    c (float): Velocidad del sonido en m/s (por defecto usa la constante C).

    Returns:
    float: Ángulo estimado en grados.
    """
    val = tdoa * c / d

    # Control de dominio para np.arccos, que debe estar en [-1, 1]
    # Si val está fuera de este rango, significa que el TDOA medido es físicamente imposible
    # para la distancia 'd' dada, o hay mucho ruido.
    if not (-1.0 <= val <= 1.0):
        # print(f"ADVERTENCIA: Valor para arccos ({val:.4f}) fuera del rango [-1, 1]. TDOA={tdoa:.2e}s, d={d}m. Puede indicar TDOA no físico.")
        # Se podría devolver NaN, un valor por defecto, o clampear. Clampeamos para obtener un ángulo.
        pass # Se clampeará abajo con np.clip

    val = np.clip(val, -1.0, 1.0)

    theta_rad = np.arccos(val)    # Ángulo con el eje del par de micrófonos, en radianes.
                                  # Si tdoa es positivo, la señal llega primero al mic de referencia (implícito en el cálculo del tdoa).
                                  # Si el par está en el eje x, mic1 en origen, mic2 en (d,0),
                                  # y tdoa = t_mic1 - t_mic2.
                                  # Si tdoa > 0, llega antes a mic2. Si tdoa < 0, llega antes a mic1.
                                  # Un ángulo de 0 rad (0 deg) significa que la fuente está en la dirección del vector mic1->mic2.
                                  # Un ángulo de pi rad (180 deg) significa dirección opuesta.
                                  # Un ángulo de pi/2 rad (90 deg) significa que está en la perpendicular (broadside).

    return np.degrees(theta_rad)