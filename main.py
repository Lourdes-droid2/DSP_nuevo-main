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
SNRS_TO_TEST_DB = [-5, 0, 5, 10, 15, 20, 25, 30]
C_SOUND = 343.0

def calculate_real_tdoa(source_pos, mic_a_pos, mic_b_pos, c=C_SOUND):
    """Calcula TDOA real basado en geometría."""
    dist_source_mic_a = np.linalg.norm(np.array(source_pos) - np.array(mic_a_pos))
    dist_source_mic_b = np.linalg.norm(np.array(source_pos) - np.array(mic_b_pos))
    # TDOA = (tiempo_mic_B - tiempo_mic_A)
    # Los estimadores en tdoa.py para (sig_a, sig_b) devuelven t_b - t_a.
    # mic_a corresponde a sig_a (señal 1), mic_b corresponde a sig_b (señal 2).
    tdoa_real = (dist_source_mic_b - dist_source_mic_a) / c # CORREGIDO SIGNO
    return tdoa_real

def add_noise_for_snr(signal, target_snr_db, fs, signal_power=None):
    """Añade ruido AWGN a una señal para un SNR objetivo."""
    if signal_power is None:
        signal_power = np.mean(signal**2)
    if signal_power == 0:
        return signal

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

    try:
        metadata_df = pd.read_csv(METADATA_FILENAME)
    except pd.errors.EmptyDataError:
        print(f"Error: El archivo de metadatos {METADATA_FILENAME} está vacío.")
        return
    print(f"Metadatos cargados: {len(metadata_df)} configuraciones encontradas.")
    if metadata_df.empty:
        print("El DataFrame de metadatos está vacío. No hay nada que procesar.")
        return

    anechoic_signal, fs_anechoic = load_signal_from_wav(ANECHOIC_SIGNAL_PATH, target_fs=48000)
    if anechoic_signal is None:
        print(f"Error: No se pudo cargar la señal anecoica de {ANECHOIC_SIGNAL_PATH}")
        return
    print(f"Señal anecoica cargada: {ANECHOIC_SIGNAL_PATH} (Fs: {fs_anechoic} Hz)")

    all_experiment_results = []
    tdoa_methods = ['cc', 'phat', 'scot', 'ml']

    for index, sim_params in metadata_df.iterrows():
        config_id_for_print = sim_params.get('config_id', f'fila_{index}')
        print(f"\nProcesando Config ID: {config_id_for_print} ({index+1}/{len(metadata_df)})..." )

        required_keys = ['fs_hz', 'num_mics_in_array', 'rir_file_basename',
                         'source_pos_x', 'source_pos_y', 'source_pos_z',
                         'actual_azimuth_src_to_array_center_deg', 'mic_separation_m']

        skip_config = False
        for key in required_keys:
            if key not in sim_params or pd.isna(sim_params[key]):
                print(f"  Advertencia: Clave '{key}' faltante o NaN para Config ID: {config_id_for_print}. Saltando config.")
                skip_config = True; break
        if skip_config: continue

        try:
            fs_sim = float(sim_params['fs_hz'])
            num_mics_for_config = int(sim_params['num_mics_in_array'])
            if num_mics_for_config <= 0:
                 print(f"  Advertencia: 'num_mics_in_array' debe ser positivo. Saltando config.")
                 continue
            source_pos_actual = [float(sim_params['source_pos_x']), float(sim_params['source_pos_y']), float(sim_params['source_pos_z'])]
            real_doa_deg = float(sim_params['actual_azimuth_src_to_array_center_deg'])
            mic_sep = float(sim_params['mic_separation_m'])
        except ValueError as e:
            print(f"  Error convirtiendo parámetros a numérico para Config ID: {config_id_for_print}: {e}. Saltando config.")
            continue

        if fs_sim != fs_anechoic:
            print(f"  Advertencia: Fs de simulación ({fs_sim}) no coincide con Fs anecoica ({fs_anechoic}). Saltando config.")
            continue

        mic_rirs = []
        mic_positions_actual = []
        valid_rirs_loaded = True
        for i in range(num_mics_for_config):
            rir_path = os.path.join(RIR_DATASET_DIR, f"{sim_params['rir_file_basename']}_micidx_{i}.wav")
            if os.path.exists(rir_path):
                try:
                    rir_data, _ = sf.read(rir_path)
                    mic_rirs.append(rir_data)
                    mic_pos_keys = [f'mic{i}_pos_x', f'mic{i}_pos_y', f'mic{i}_pos_z']
                    if not all(key in sim_params and not pd.isna(sim_params[key]) for key in mic_pos_keys):
                        print(f"  Advertencia: Posiciones para micrófono {i} faltantes o NaN. Saltando config.")
                        valid_rirs_loaded = False; break
                    mic_positions_actual.append([float(sim_params[f'mic{i}_pos_x']), float(sim_params[f'mic{i}_pos_y']), float(sim_params[f'mic{i}_pos_z'])])
                except Exception as e:
                    print(f"  Error cargando RIR {rir_path} o procesando posiciones: {e}. Saltando config.")
                    valid_rirs_loaded = False; break
            else:
                print(f"  Error: RIR no encontrada: {rir_path}. Saltando config.")
                valid_rirs_loaded = False; break
        if not valid_rirs_loaded or len(mic_rirs) != num_mics_for_config:
            continue

        reverberant_signals = [np.convolve(anechoic_signal, rir, mode='full') for rir in mic_rirs]

        for snr_db_val in SNRS_TO_TEST_DB:
            noisy_signals = [add_noise_for_snr(sig, snr_db_val, fs_sim) for sig in reverberant_signals]
            mic_pairs_info = []
            for i in range(len(noisy_signals)):
                for j in range(i + 1, len(noisy_signals)):
                    if abs(i-j) == 1: pair_d = mic_sep
                    elif abs(i-j) == 2: pair_d = 2 * mic_sep
                    elif abs(i-j) == 3: pair_d = 3 * mic_sep
                    else: continue

                    if i >= len(mic_positions_actual) or j >= len(mic_positions_actual): continue
                    real_tdoa_pair = calculate_real_tdoa(source_pos_actual, mic_positions_actual[i], mic_positions_actual[j])
                    mic_pairs_info.append({'mic1_idx': i, 'mic2_idx': j, 'd': pair_d, 'real_tdoa': real_tdoa_pair})

            estimated_doas_for_array = {method: [] for method in tdoa_methods}

            for pair_info in mic_pairs_info:
                idx1, idx2, d_pair, real_tdoa_p = pair_info['mic1_idx'], pair_info['mic2_idx'], pair_info['d'], pair_info['real_tdoa']
                if idx1 >= len(noisy_signals) or idx2 >= len(noisy_signals): continue
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

                    tdoa_error_s = tdoa_val - real_tdoa_p if not np.isnan(tdoa_val) and not pd.isna(real_tdoa_p) else np.nan
                    phi_from_pair = estimate_doa_from_tdoa(tdoa_val, d_pair)

                    current_pair_results = result_entry_base.copy()
                    current_pair_results.update({
                        'tdoa_method': tdoa_method_name, 'tdoa_estimated_s': tdoa_val,
                        'tdoa_error_s': tdoa_error_s, 'tdoa_computation_time_s': comp_time,
                        'doa_estimated_from_pair_deg': phi_from_pair
                    })
                    all_experiment_results.append(current_pair_results)

                    if abs(idx1-idx2) == 1 and not np.isnan(phi_from_pair):
                        estimated_doas_for_array[tdoa_method_name].append(phi_from_pair)

            for method_name, phis in estimated_doas_for_array.items():
                if phis:
                    avg_phi_array = np.mean(phis)
                    avg_azimuth_array_estimated = 90.0 - avg_phi_array
                    avg_azimuth_array_estimated = (avg_azimuth_array_estimated + 180) % 360 - 180

                    error_doa_array = np.nan
                    if not np.isnan(avg_azimuth_array_estimated) and not pd.isna(real_doa_deg):
                        error_doa_array = avg_azimuth_array_estimated - real_doa_deg

                    array_doa_entry = sim_params.to_dict()
                    array_doa_entry.update({
                        'snr_db': snr_db_val, 'mic_pair': 'array_avg_adj_pairs',
                        'tdoa_method_for_avg_doa': method_name,
                        'doa_array_estimated_deg': avg_azimuth_array_estimated,
                        'doa_array_real_deg': real_doa_deg,
                        'doa_array_error_deg': error_doa_array
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
        print(f"Advertencia: Archivo anecoico {ANECHOIC_SIGNAL_PATH} no encontrado. Creando dummy.")
        sf.write(ANECHOIC_SIGNAL_PATH, np.random.randn(48000 * 2), 48000)

    if not os.path.exists(RIR_DATASET_DIR):
        os.makedirs(RIR_DATASET_DIR, exist_ok=True)
    if not os.path.exists(METADATA_FILENAME):
        print(f"Advertencia: Archivo de metadatos {METADATA_FILENAME} no encontrado. Creando dummy.")
        dummy_meta_data = [{
            'config_id': 'dummy_cfg1', 'fs_hz': 48000.0, 'room_dim_x': 5.0, 'room_dim_y': 4.0, 'room_dim_z': 3.0,
            'rt60_target_s': 0.5, 'is_anechoic': False,
            'source_pos_x': 1.0, 'source_pos_y': 1.0, 'source_pos_z': 1.5,
            'array_center_x': 2.5, 'array_center_y': 2.0, 'array_center_z': 1.5,
            'actual_dist_src_to_array_center_m': 2.0, 'actual_azimuth_src_to_array_center_deg': 45.0,
            'num_mics_in_array': 2, 'mic_separation_m': 0.1, 'rir_file_basename': 'dummy_rir_cfg1',
            'mic0_pos_x': 2.45, 'mic0_pos_y': 2.0, 'mic0_pos_z': 1.5,
            'mic1_pos_x': 2.55, 'mic1_pos_y': 2.0, 'mic1_pos_z': 1.5
        }]
        for row in dummy_meta_data: # Asegurar floats
            for key, value in row.items():
                if isinstance(value, int) and key not in ['num_mics_in_array']: row[key] = float(value)
                if key == 'fs_hz': row[key] = float(value)
        pd.DataFrame(dummy_meta_data).to_csv(METADATA_FILENAME, index=False)
        for i in range(dummy_meta_data[0]['num_mics_in_array']):
            dummy_rir_path = os.path.join(RIR_DATASET_DIR, f"{dummy_meta_data[0]['rir_file_basename']}_micidx_{i}.wav")
            if not os.path.exists(dummy_rir_path):
                sf.write(dummy_rir_path, np.random.randn(100), int(dummy_meta_data[0]['fs_hz']))
    process_simulation_data()
