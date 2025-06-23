import os
import numpy as np
import soundfile as sf # Para load_rirs

# Importar funciones de los módulos actualizados
from tdoa import estimate_tdoa_cc, estimate_tdoa_gcc
from doa import estimate_doa_from_tdoa, C # Importar C si se va a usar directamente aquí

# Nota: La constante C también está definida en doa.py y es usada por defecto
# por estimate_doa_from_tdoa. No es estrictamente necesario redefinirla aquí
# a menos que se quiera usar explícitamente en main.py para otros cálculos.
# Por consistencia, podemos quitar la redefinición de C = 343 aquí si doa.py ya la tiene.

def load_rirs(base_filepath_template, num_mics, config_suffix=""):
    """
    Carga las RIRs generadas por simulation.py desde archivos .wav.
    El base_filepath_template es el nombre del archivo SIN el _micidx_TAG.
    Ejemplo: "rir_dataset_user_defined/rir_rt60_0.5_room_6x5x3.0_src_1x1x1.5_config_3.wav"
    donde "_config_3" es el config_suffix si se usa.
    La función quitará la extensión .wav, y luego añadirá _micidx_ y la extensión.
    """
    rirs = []
    fs_rir = -1  # Para almacenar la frecuencia de muestreo, asumimos que todas las RIRs la comparten

    # Quitar la extensión .wav del template para construir nombres base correctos
    base_name_no_ext, _ = os.path.splitext(base_filepath_template)

    for idx in range(num_mics):
        # Construir el nombre de archivo específico para cada RIR de micrófono
        rir_filename = f"{base_name_no_ext}_micidx_{idx}.wav"

        if os.path.exists(rir_filename):
            try:
                rir_signal, current_fs = sf.read(rir_filename)
                rirs.append(rir_signal)
                if fs_rir == -1:
                    fs_rir = current_fs
                elif fs_rir != current_fs:
                    print(f"ADVERTENCIA: Frecuencias de muestreo inconsistentes entre RIRs! {rir_filename} tiene {current_fs} Hz, se esperaba {fs_rir} Hz.")
                    # Podría decidirse manejar este error de forma más estricta.
            except Exception as e:
                print(f"ADVERTENCIA: No se pudo leer el archivo RIR (aunque existe): {rir_filename}. Error: {e}")
        else:
            print(f"ADVERTENCIA: Archivo RIR no encontrado: {rir_filename}")

    if not rirs:
        print(f"ERROR: No se cargaron RIRs para la base: {base_filepath_template}. Verifique los nombres de archivo y la salida de simulation.py.")
        return [], -1 # Devolver lista vacía y fs inválida si no se carga nada.

    return rirs, fs_rir


def process_configuration(base_filepath_template, num_mics_in_config, mic_distance=0.1):
    """
    Procesa una configuración completa de RIRs: carga RIRs, estima TDOAs y DOAs
    entre pares consecutivos de micrófonos.
    """
    print(f"\nProcesando configuración basada en: {base_filepath_template}")
    print(f"Esperando {num_mics_in_config} micrófonos para esta configuración.")

    rirs, fs = load_rirs(base_filepath_template, num_mics_in_config)

    if not rirs or fs == -1:
        print(f"No se pudieron cargar RIRs o fs inválida para {base_filepath_template}. Abortando procesamiento para esta configuración.")
        return

    if len(rirs) < 2:
        print(f"Se necesitan al menos 2 RIRs para estimar TDOA/DOA. Se cargaron {len(rirs)}. Abortando.")
        return

    # Si se cargaron menos RIRs de las esperadas pero al menos 2, advertir pero continuar.
    if len(rirs) < num_mics_in_config:
        print(f"ADVERTENCIA: Se esperaban {num_mics_in_config} RIRs, pero solo se cargaron {len(rirs)}. Se procesarán los pares disponibles.")

    actual_num_mics_loaded = len(rirs)

    print(f"RIRs cargadas exitosamente. Frecuencia de muestreo: {fs} Hz.")

    for i in range(actual_num_mics_loaded - 1):
        # Asumimos que las RIRs son señales mono. Si son estéreo, tomar solo un canal.
        sig1 = rirs[i]
        if sig1.ndim > 1: sig1 = sig1[:, 0] # Tomar primer canal si es multicanal

        sig2 = rirs[i+1]
        if sig2.ndim > 1: sig2 = sig2[:, 0] # Tomar primer canal si es multicanal

        print(f"\n  Calculando para par de micrófonos original: Mic {i} vs Mic {i+1}")
        # (Nota: si algunos micrófonos intermedios no se cargaron, los índices 'i' y 'i+1'
        # se refieren a los índices en la lista 'rirs' cargada, no necesariamente a los
        # índices originales absolutos si hubo fallos de carga no consecutivos)

        # Estimación de TDOA
        tdoa_cc_val = estimate_tdoa_cc(sig1, sig2, fs)
        tdoa_phat_val = estimate_tdoa_gcc(sig1, sig2, fs, method='phat')
        tdoa_scot_val = estimate_tdoa_gcc(sig1, sig2, fs, method='scot')

        # Estimación de DOA
        # Usamos la constante C importada o definida en doa.py por defecto.
        doa_cc_val = estimate_doa_from_tdoa(tdoa_cc_val, d=mic_distance)
        doa_phat_val = estimate_doa_from_tdoa(tdoa_phat_val, d=mic_distance)
        doa_scot_val = estimate_doa_from_tdoa(tdoa_scot_val, d=mic_distance)

        print(f"    TDOA CC:    {tdoa_cc_val*1e6:.2f} µs  | DOA: {doa_cc_val:.2f}°")
        print(f"    TDOA PHAT:  {tdoa_phat_val*1e6:.2f} µs  | DOA: {doa_phat_val:.2f}°")
        print(f"    TDOA SCOT:  {tdoa_scot_val*1e6:.2f} µs  | DOA: {doa_scot_val:.2f}°")


if __name__ == "__main__":
    print("==============================================================")
    print("Iniciando script de procesamiento de RIRs y estimación TDOA/DOA (main.py)")
    print("==============================================================")

    # --- Configuración del Ejemplo de Procesamiento ---
    # Este es el nombre BASE del archivo que simulation.py genera ANTES de añadir "_micidx_N.wav"
    # El usuario debe asegurarse de que este nombre base y el número de micrófonos coincidan
    # con una de las configuraciones generadas por simulation.py.

    # Ejemplo basado en una de las configuraciones de simulation.py:
    # Configuración original en simulation.py:
    # {
    #     "rt60_tgt": 0.5,
    #     "room_dim": [6, 5, 3.0],
    #     "source_pos": [1, 1, 1.5],
    #     "mic_positions": [[5, 4, 1.5], [1.5, 4, 1.5], [3, 2.5, 2.0]], # Tres micrófonos
    #     "filename_suffix": "config_3" # (o el default si no se provee)
    # }
    # El nombre de archivo base generado por simulation.py para esto sería:
    # "rir_rt60_0.5_room_6x5x3.0_src_1x1x1.5_config_3.wav" (antes de añadir _micidx_N)

    base_rir_filepath = "rir_dataset_user_defined/rir_rt60_0.5_room_6x5x3.0_src_1x1x1.5_config_3.wav"
    number_of_microphones_in_this_config = 3
    assumed_mic_distance_for_doa = 0.1 # Distancia entre micrófonos para el cálculo de DOA (ej. 10 cm)
                                       # Esto es una simplificación si el array no es uniforme o lineal.
                                       # La función estimate_doa_from_tdoa asume esta 'd' para el par.

    # Verificar si el directorio de RIRs existe para dar un mejor feedback
    rir_dir = os.path.dirname(base_rir_filepath)
    if not os.path.exists(rir_dir):
        print(f"ERROR: El directorio de RIRs '{rir_dir}' no existe.")
        print("Por favor, ejecute simulation.py primero para generar las RIRs.")
    else:
        # Llamar a la función de procesamiento para la configuración de ejemplo
        process_configuration(
            base_filepath_template=base_rir_filepath,
            num_mics_in_config=number_of_microphones_in_this_config,
            mic_distance=assumed_mic_distance_for_doa
        )

    # Se podrían añadir más llamadas a process_configuration para otros conjuntos de RIRs.
    # Por ejemplo, para la primera configuración de simulation.py (small_room_short_rt60_2mics):
    # base_rir_filepath_2 = "rir_dataset_user_defined/rir_rt60_0.3_room_5x4x2.8_src_1.5x1.0x1.2_small_room_short_rt60_2mics.wav"
    # number_of_microphones_2 = 2
    # process_configuration(base_rir_filepath_2, number_of_microphones_2, assumed_mic_distance_for_doa)

    print("\n==============================================================")
    print("Procesamiento en main.py finalizado.")
    print("==============================================================")