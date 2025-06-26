import numpy as np

C_SOUND_DEFAULT = 343.0  # Velocidad del sonido en m/s por defecto

def estimate_doa_from_tdoa(tdoa, d, c=C_SOUND_DEFAULT):
    """
    Estima el Ángulo de Llegada (DOA) en grados a partir del TDOA entre un par de micrófonos.
    El ángulo (theta) se calcula respecto al EJE del par de micrófonos (endfire = 0°, broadside = 90°, endfire opuesto = 180°).
    Rango: 0 a 180 grados.

    Parameters:
    tdoa (float): Diferencia de tiempo de llegada en segundos.
    d (float): Distancia entre los dos micrófonos en metros.
    c (float): Velocidad del sonido en m/s.

    Returns:
    float: Ángulo estimado en grados (0 a 180).
    """
    if np.isnan(tdoa) or np.isinf(tdoa):
        return np.nan
    if d <= 0:
        return np.nan

    val = (c * tdoa) / d
    val_clipped = np.clip(val, -1.0, 1.0)
    theta_rad = np.arccos(val_clipped)
    return np.degrees(theta_rad)

# Ejemplos de prueba (pueden ejecutarse si el archivo se corre directamente)
if __name__ == '__main__':
    print("--- Pruebas para doa.py (0-180 grados, respecto al eje) ---")
    test_d = 0.1 # 10 cm

    # Caso 1: Fuente a 30 grados respecto al eje (theta=30)
    angle_deg_test1 = 30.0
    tdoa_val_1 = (test_d * np.cos(np.deg2rad(angle_deg_test1))) / C_SOUND_DEFAULT
    doa1 = estimate_doa_from_tdoa(tdoa_val_1, d=test_d)
    print(f"Test 1 (Fuente a {angle_deg_test1:.1f} deg respecto al eje): TDOA={tdoa_val_1*1e6:.2f} us -> DOA={doa1:.2f} deg")

    # Caso 2: Fuente a 120 grados respecto al eje (theta=120)
    angle_deg_test2 = 120.0
    tdoa_val_2 = (test_d * np.cos(np.deg2rad(angle_deg_test2))) / C_SOUND_DEFAULT
    doa2 = estimate_doa_from_tdoa(tdoa_val_2, d=test_d)
    print(f"Test 2 (Fuente a {angle_deg_test2:.1f} deg respecto al eje): TDOA={tdoa_val_2*1e6:.2f} us -> DOA={doa2:.2f} deg")

    # Caso 3: Broadside (90 grados respecto al eje)
    tdoa_val_3 = (test_d * np.cos(np.deg2rad(90.0))) / C_SOUND_DEFAULT
    doa3 = estimate_doa_from_tdoa(tdoa_val_3, d=test_d)
    print(f"Test 3 (Fuente a 90 deg respecto al eje): TDOA={tdoa_val_3*1e6:.2f} us -> DOA={doa3:.2f} deg")

    # Caso 4: Endfire (0 grados respecto al eje)
    tdoa_val_4 = (test_d * np.cos(np.deg2rad(0.0))) / C_SOUND_DEFAULT
    doa4 = estimate_doa_from_tdoa(tdoa_val_4, d=test_d)
    print(f"Test 4 (Fuente a 0 deg respecto al eje): TDOA={tdoa_val_4*1e6:.2f} us -> DOA={doa4:.2f} deg")

    # Caso 5: Endfire opuesto (180 grados respecto al eje)
    tdoa_val_5 = (test_d * np.cos(np.deg2rad(180.0))) / C_SOUND_DEFAULT
    doa5 = estimate_doa_from_tdoa(tdoa_val_5, d=test_d)
    print(f"Test 5 (Fuente a 180 deg respecto al eje): TDOA={tdoa_val_5*1e6:.2f} us -> DOA={doa5:.2f} deg")

    # Caso 6: TDOA NaN
    tdoa_val_6 = np.nan
    doa6 = estimate_doa_from_tdoa(tdoa_val_6, d=test_d)
    print(f"Test 6 (TDOA NaN): DOA={doa6}")