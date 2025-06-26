import os
import numpy as np
import soundfile as sf
import pandas as pd
import time
# Asumiendo que los siguientes archivos están en el mismo directorio o en PYTHONPATH
from load_signal import load_signal_from_wav
from tdoa import estimate_tdoa_cc, estimate_tdoa_gcc
from doa import estimate_doa_from_tdoa

# --- Constantes y Configuraciones Globales ---
RIR_DATASET_DIR = "rir_dataset_user_defined"
METADATA_FILENAME = os.path.join(RIR_DATASET_DIR, "simulation_metadata.csv")
ANECHOIC_SIGNAL_PATH = "p336_007.wav" # Asegúrate que este archivo exista y sea accesible
# SNRS_TO_TEST_DB = [-5, 0, 5, 10, 15, 20, 25, 30] # Original
SNRS_TO_TEST_DB = [0, 10, 20] # Reducido para prueba de timeout
C_SOUND = 343.0

def calculate_real_tdoa(source_pos, mic_a_pos, mic_b_pos, c=C_SOUND):
    """Calcula TDOA real basado en geometría."""
    dist_source_mic_a = np.linalg.norm(np.array(source_pos) - np.array(mic_a_pos))
    dist_source_mic_b = np.linalg.norm(np.array(source_pos) - np.array(mic_b_pos))
    tdoa_real = (dist_source_mic_a - dist_source_mic_b) / c
    return tdoa_real

def add_noise_for_snr(signal, target_snr_db, fs, signal_power=None):
    """Añade ruido AWGN a una señal para un SNR objetivo."""
    if signal_power is None:
        signal_power = np.mean(signal**2)
    if signal_power == 0: return signal
    snr_linear = 10**(target_snr_db / 10.0)
    noise_power_target = signal_power / snr_linear
    noise = np.random.normal(0, 1, len(signal))
    current_noise_power = np.mean(noise**2)
    if current_noise_power == 0: current_noise_power = 1e-10
    scaled_noise = noise * np.sqrt(noise_power_target / current_noise_power)
    return signal + scaled_noise

def process_simulation_data():
    print("--- main.py: Iniciando procesamiento de datos de simulación ---")
    if not os.path.exists(METADATA_FILENAME):
        print(f"Error: Archivo de metadatos no encontrado: {METADATA_FILENAME}")
        return

    metadata_df_full = pd.read_csv(METADATA_FILENAME)
    print(f"Metadatos cargados: {len(metadata_df_full)} configuraciones encontradas en CSV.")

    # Filtrar para un subconjunto de configuraciones para prueba de timeout
    # configs_to_test = ['small_medrev_4mics', 'large_highrev_8mics']
    # metadata_df = metadata_df_full[metadata_df_full['config_id'].isin(configs_to_test)]
    # print(f"Procesando un subconjunto de {len(metadata_df)} configuraciones para prueba.")

    # Procesar todas las configuraciones del CSV. El filtro .head(1) se elimina.
    metadata_df = metadata_df_full
    print(f"Procesando {len(metadata_df)} configuraciones encontradas en el CSV.")


    anechoic_signal, fs_anechoic = load_signal_from_wav(ANECHOIC_SIGNAL_PATH, target_fs=48000)
    if anechoic_signal is None:
        print(f"Error: No se pudo cargar la señal anecoica de {ANECHOIC_SIGNAL_PATH}")
        return
    print(f"Señal anecoica cargada: {ANECHOIC_SIGNAL_PATH} (Fs: {fs_anechoic} Hz)")

    all_experiment_results = []
    # tdoa_methods = ['cc', 'phat', 'scot', 'ml'] # Original
    tdoa_methods = ['phat'] # Reducido para prueba de timeout
    print(f"Métodos TDOA a probar: {tdoa_methods}")

    for index, sim_params in metadata_df.iterrows():
        print(f"\nProcesando Config ID: {sim_params['config_id']} ({index+1}/{len(metadata_df)})..." )
        fs_sim = sim_params['fs_hz']
        if fs_sim != fs_anechoic:
            print(f"  Advertencia: Fs de simulación ({fs_sim}) no coincide con Fs anecoica ({fs_anechoic}). Saltando config.")
            continue

        mic_rirs = []
        mic_positions_actual = []
        valid_rirs_loaded = True
        # Usar 'num_mics_processed' que refleja el número real de RIRs/mics disponibles
        num_available_mics = int(sim_params['num_mics_processed'])
        for i in range(num_available_mics):
            rir_path = os.path.join(RIR_DATASET_DIR, f"{sim_params['rir_file_basename']}_micidx_{i}.wav")
            if os.path.exists(rir_path):
                try:
                    rir_data, _ = sf.read(rir_path)
                    mic_rirs.append(rir_data)
                    mic_pos_x_key = f'mic{i}_pos_x'
                    # Validar que las claves de posición de micrófono existen y no son NaN
                    if not (mic_pos_x_key in sim_params and \
                            f'mic{i}_pos_y' in sim_params and \
                            f'mic{i}_pos_z' in sim_params and \
                            not pd.isna(sim_params[mic_pos_x_key]) and \
                            not pd.isna(sim_params[f'mic{i}_pos_y']) and \
                            not pd.isna(sim_params[f'mic{i}_pos_z'])):
                        print(f"  Advertencia: Posición incompleta o NaN para micrófono {i} en config {sim_params['config_id']}. Saltando config.")
                        valid_rirs_loaded = False; break
                    mic_positions_actual.append([sim_params[f'mic{i}_pos_x'], sim_params[f'mic{i}_pos_y'], sim_params[f'mic{i}_pos_z']])
                except Exception as e:
                    print(f"  Error cargando RIR {rir_path}: {e}. Saltando config.")
                    valid_rirs_loaded = False; break
            else:
                print(f"  Error: RIR no encontrada: {rir_path}. Saltando config.")
                valid_rirs_loaded = False; break

        if not valid_rirs_loaded or len(mic_rirs) != num_available_mics:
            if valid_rirs_loaded: # Solo imprimir si el problema fue la cuenta y no un error previo
                 print(f"  Discrepancia en RIRs para {sim_params['config_id']}. Esperadas: {num_available_mics}, Cargadas: {len(mic_rirs)}. Saltando config.")
            continue

        reverberant_signals = [np.convolve(anechoic_signal, rir, mode='full') for rir in mic_rirs]
        source_pos_actual = [sim_params['source_pos_x'], sim_params['source_pos_y'], sim_params['source_pos_z']]
        real_doa_deg = sim_params['actual_azimuth_src_to_array_center_deg']

        for snr_db_val in SNRS_TO_TEST_DB:
            noisy_signals = [add_noise_for_snr(sig, snr_db_val, fs_sim) for sig in reverberant_signals]
            mic_sep = sim_params['mic_separation_m']
            mic_pairs_info = []

            # Iterar sobre todos los pares de micrófonos únicos
            for i in range(num_available_mics):
                for j in range(i + 1, num_available_mics):
                    pair_d = abs(j - i) * mic_sep # Asume array lineal con separación uniforme

                    # Asegurar que las posiciones de los micrófonos para este par son válidas
                    # (mic_positions_actual debería tener longitud num_available_mics si todo fue bien)
                    if i >= len(mic_positions_actual) or j >= len(mic_positions_actual):
                         print(f"  Advertencia: Índices de micrófono {i} o {j} fuera de rango para mic_positions_actual (longitud {len(mic_positions_actual)}) en config {sim_params['config_id']}. Saltando par.")
                         continue

                    real_tdoa_pair = calculate_real_tdoa(source_pos_actual, mic_positions_actual[i], mic_positions_actual[j])
                    mic_pairs_info.append({'mic1_idx': i, 'mic2_idx': j, 'd': pair_d, 'real_tdoa': real_tdoa_pair})

            estimated_doas_for_array = {method: [] for method in tdoa_methods}

            for pair_info in mic_pairs_info:
                idx1, idx2, d_pair, real_tdoa_p = pair_info['mic1_idx'], pair_info['mic2_idx'], pair_info['d'], pair_info['real_tdoa']
                sig_a, sig_b = noisy_signals[idx1], noisy_signals[idx2]

                result_entry_base = sim_params.to_dict()
                result_entry_base.update({
                    'snr_db': snr_db_val, 'mic_pair': f"{idx1}-{idx2}",
                    'mic_pair_distance_m': d_pair, 'tdoa_real_s': real_tdoa_p
                })

                for tdoa_method_name in tdoa_methods:
                    tdoa_val, comp_time = np.nan, np.nan
                    if tdoa_method_name == 'cc':
                        tdoa_val, comp_time = estimate_tdoa_cc(sig_a, sig_b, fs_sim)
                    else:
                        tdoa_val, comp_time = estimate_tdoa_gcc(sig_a, sig_b, fs_sim, method=tdoa_method_name)

                    tdoa_error_s = tdoa_val - real_tdoa_p if not np.isnan(tdoa_val) else np.nan
                    doa_from_pair = estimate_doa_from_tdoa(tdoa_val, d_pair)

                    current_pair_results = result_entry_base.copy()
                    current_pair_results.update({
                        'tdoa_method': tdoa_method_name, 'tdoa_estimated_s': tdoa_val,
                        'tdoa_error_s': tdoa_error_s, 'tdoa_computation_time_s': comp_time,
                        'doa_estimated_from_pair_deg': doa_from_pair
                    })
                    all_experiment_results.append(current_pair_results)

                    if abs(idx1-idx2) == 1 and not np.isnan(doa_from_pair):
                        estimated_doas_for_array[tdoa_method_name].append(doa_from_pair)

            for method_name, doas in estimated_doas_for_array.items():
                if doas:
                    avg_doa_array = np.mean(doas)
                    error_doa_array = avg_doa_array - real_doa_deg if not np.isnan(avg_doa_array) else np.nan
                    array_doa_entry = sim_params.to_dict()
                    array_doa_entry.update({
                        'snr_db': snr_db_val, 'mic_pair': 'array_avg_adj_pairs',
                        'tdoa_method_for_avg_doa': method_name,
                        'doa_array_estimated_deg': avg_doa_array,
                        'doa_array_real_deg': real_doa_deg, 'doa_array_error_deg': error_doa_array
                    })
                    all_experiment_results.append(array_doa_entry)

    if all_experiment_results:
        results_df = pd.DataFrame(all_experiment_results)
        output_csv_path = "full_experiment_results.csv"
        try:
            results_df.to_csv(output_csv_path, index=False)
            print(f"\nResultados ({len(results_df)} filas) guardados en: {output_csv_path}")
        except Exception as e:
            print(f"Error al guardar CSV: {e}")
    else:
        print("No se generaron resultados.")
    print("--- main.py: Procesamiento finalizado ---")

if __name__ == "__main__":
    if not os.path.exists(ANECHOIC_SIGNAL_PATH):
        raise ValueError(f"Archivo anecoico {ANECHOIC_SIGNAL_PATH} no encontrado. Por favor, asegúrate de que el archivo existe en el directorio actual.")
        #print(f"Advertencia: Archivo anecoico {ANECHOIC_SIGNAL_PATH} no encontrado.")

    if not os.path.exists(METADATA_FILENAME):
        raise ValueError(f"Archivo de metadatos {METADATA_FILENAME} no encontrado. Por favor, asegúrate de que el archivo existe en el directorio {RIR_DATASET_DIR}.")
        #print(f"Advertencia: Archivo de metadatos {METADATA_FILENAME} no encontrado. Creando dummy.")

    process_simulation_data()
