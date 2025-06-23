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
    # TDOA = (tiempo_mic_A - tiempo_mic_B)
    # Si señal llega antes a A, dist_source_mic_a es menor, tdoa es negativo.
    # Esto depende de la convención de los algoritmos TDOA. 
    # Por ahora: (dist_B - dist_A) / c para que sea positivo si B está más lejos.
    # Ajustar según la convención de tdoa_estimate (sig1 vs sig2)
    # Si tdoa_estimate es t1-t2, y mic_A es sig1, mic_B es sig2:
    # tdoa_real = (dist_source_mic_a / c) - (dist_source_mic_b / c)
    tdoa_real = (dist_source_mic_a - dist_source_mic_b) / c
    return tdoa_real

def add_noise_for_snr(signal, target_snr_db, fs, signal_power=None):
    """Añade ruido AWGN a una señal para un SNR objetivo."""
    if signal_power is None:
        signal_power = np.mean(signal**2)
    if signal_power == 0: # Señal es silencio
        return signal # No se puede añadir ruido basado en SNR a una señal de potencia cero
    
    snr_linear = 10**(target_snr_db / 10.0)
    noise_power_target = signal_power / snr_linear
    
    # Generar ruido blanco gaussiano
    noise = np.random.normal(0, 1, len(signal))
    current_noise_power = np.mean(noise**2)
    if current_noise_power == 0: current_noise_power = 1e-10 # Evitar división por cero
        
    scaled_noise = noise * np.sqrt(noise_power_target / current_noise_power)
    return signal + scaled_noise

def process_simulation_data():
    print("--- main.py: Iniciando procesamiento de datos de simulación ---")
    if not os.path.exists(METADATA_FILENAME):
        print(f"Error: Archivo de metadatos no encontrado: {METADATA_FILENAME}")
        return
    
    metadata_df = pd.read_csv(METADATA_FILENAME)
    print(f"Metadatos cargados: {len(metadata_df)} configuraciones encontradas.")

    anechoic_signal, fs_anechoic = load_signal_from_wav(ANECHOIC_SIGNAL_PATH, target_fs=48000)
    if anechoic_signal is None:
        print(f"Error: No se pudo cargar la señal anecoica de {ANECHOIC_SIGNAL_PATH}")
        return
    print(f"Señal anecoica cargada: {ANECHOIC_SIGNAL_PATH} (Fs: {fs_anechoic} Hz)")

    all_experiment_results = []
    tdoa_methods = ['cc', 'phat', 'scot', 'ml']

    for index, sim_params in metadata_df.iterrows():
        print(f"\nProcesando Config ID: {sim_params['config_id']} ({index+1}/{len(metadata_df)})..." )
        fs_sim = sim_params['fs_hz']
        if fs_sim != fs_anechoic:
            print(f"  Advertencia: Fs de simulación ({fs_sim}) no coincide con Fs anecoica ({fs_anechoic}). Saltando config.")
            continue

        # Cargar RIRs para esta configuración
        mic_rirs = []
        mic_positions_actual = []
        valid_rirs_loaded = True
        for i in range(int(sim_params['num_mics_in_array'])):
            rir_path = os.path.join(RIR_DATASET_DIR, f"{sim_params['rir_file_basename']}_micidx_{i}.wav")
            if os.path.exists(rir_path):
                try:
                    rir_data, _ = sf.read(rir_path)
                    mic_rirs.append(rir_data)
                    mic_positions_actual.append([sim_params[f'mic{i}_pos_x'], sim_params[f'mic{i}_pos_y'], sim_params[f'mic{i}_pos_z']])
                except Exception as e:
                    print(f"  Error cargando RIR {rir_path}: {e}. Saltando config.")
                    valid_rirs_loaded = False; break
            else:
                print(f"  Error: RIR no encontrada: {rir_path}. Saltando config.")
                valid_rirs_loaded = False; break
        if not valid_rirs_loaded or len(mic_rirs) != sim_params['num_mics_in_array']:
            continue

        # Convolución
        reverberant_signals = [np.convolve(anechoic_signal, rir, mode='full') for rir in mic_rirs]
        source_pos_actual = [sim_params['source_pos_x'], sim_params['source_pos_y'], sim_params['source_pos_z']]
        real_doa_deg = sim_params['actual_azimuth_src_to_array_center_deg'] # Asumiendo que este es el DOA de referencia

        for snr_db_val in SNRS_TO_TEST_DB:
            # print(f"  Procesando SNR: {snr_db_val} dB")
            noisy_signals = [add_noise_for_snr(sig, snr_db_val, fs_sim) for sig in reverberant_signals]
            
            # Pares de micrófonos y sus distancias (d)
            # Asumimos array lineal en X, separación uniforme sim_params['mic_separation_m']
            mic_sep = sim_params['mic_separation_m']
            mic_pairs_info = [] # (idx1, idx2, distance_d, real_tdoa_for_pair)
            for i in range(len(noisy_signals)):
                for j in range(i + 1, len(noisy_signals)):
                    # Solo considerar pares con separación conocida si es necesario, o todos
                    # Para el array lineal de 4 mics, los pares relevantes son:
                    # (0,1), (1,2), (2,3) con d=mic_sep
                    # (0,2), (1,3) con d=2*mic_sep
                    # (0,3) con d=3*mic_sep
                    if abs(i-j) == 1: pair_d = mic_sep
                    elif abs(i-j) == 2: pair_d = 2 * mic_sep
                    elif abs(i-j) == 3: pair_d = 3 * mic_sep
                    else: continue # O manejar otros pares si es necesario

                    real_tdoa_pair = calculate_real_tdoa(source_pos_actual, mic_positions_actual[i], mic_positions_actual[j])
                    mic_pairs_info.append({'mic1_idx': i, 'mic2_idx': j, 'd': pair_d, 'real_tdoa': real_tdoa_pair})

            estimated_doas_for_array = {method: [] for method in tdoa_methods} # Para promediar DOAs de pares adyacentes

            for pair_info in mic_pairs_info:
                idx1, idx2, d_pair, real_tdoa_p = pair_info['mic1_idx'], pair_info['mic2_idx'], pair_info['d'], pair_info['real_tdoa']
                sig_a, sig_b = noisy_signals[idx1], noisy_signals[idx2]
                
                result_entry_base = sim_params.to_dict() # Copia parámetros de simulación
                result_entry_base.update({
                    'snr_db': snr_db_val,
                    'mic_pair': f"{idx1}-{idx2}",
                    'mic_pair_distance_m': d_pair,
                    'tdoa_real_s': real_tdoa_p
                })

                for tdoa_method_name in tdoa_methods:
                    tdoa_val, comp_time = np.nan, np.nan
                    if tdoa_method_name == 'cc':
                        tdoa_val, comp_time = estimate_tdoa_cc(sig_a, sig_b, fs_sim)
                    else: # phat, scot, ml
                        tdoa_val, comp_time = estimate_tdoa_gcc(sig_a, sig_b, fs_sim, method=tdoa_method_name)
                    
                    tdoa_error_s = tdoa_val - real_tdoa_p if not np.isnan(tdoa_val) else np.nan
                    doa_from_pair = estimate_doa_from_tdoa(tdoa_val, d_pair) # Usa C_SOUND por defecto

                    current_pair_results = result_entry_base.copy()
                    current_pair_results.update({
                        'tdoa_method': tdoa_method_name,
                        'tdoa_estimated_s': tdoa_val,
                        'tdoa_error_s': tdoa_error_s,
                        'tdoa_computation_time_s': comp_time,
                        'doa_estimated_from_pair_deg': doa_from_pair
                        # No calculamos error DOA por par aquí, sino para el array completo
                    })
                    all_experiment_results.append(current_pair_results)

                    # Si es par adyacente, guardar su DOA para el promedio del array
                    if abs(idx1-idx2) == 1 and not np.isnan(doa_from_pair):
                        estimated_doas_for_array[tdoa_method_name].append(doa_from_pair)
            
            # Calcular DOA promedio del array para cada método TDOA base
            for method_name, doas in estimated_doas_for_array.items():
                if doas: # Si hay DOAs de pares adyacentes para promediar
                    avg_doa_array = np.mean(doas)
                    error_doa_array = avg_doa_array - real_doa_deg if not np.isnan(avg_doa_array) else np.nan
                    # Guardar este resultado de DOA de array (podría ser una entrada separada o añadir a las existentes)
                    # Para simplificar, añadimos una entrada por cada método TDOA que produjo un DOA de array
                    array_doa_entry = sim_params.to_dict()
                    array_doa_entry.update({
                        'snr_db': snr_db_val,
                        'mic_pair': 'array_avg_adj_pairs', # Indicador de que es un resultado de array
                        'tdoa_method_for_avg_doa': method_name, # Qué TDOA se usó para los DOAs promediados
                        'doa_array_estimated_deg': avg_doa_array,
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
    # Crear un archivo p336_007.wav dummy si no existe, para que el script corra
    # En un entorno real, este archivo debe ser una señal anecoica real.
    if not os.path.exists(ANECHOIC_SIGNAL_PATH):
        print(f"Advertencia: Archivo anecoico {ANECHOIC_SIGNAL_PATH} no encontrado. Creando dummy.")
        sf.write(ANECHOIC_SIGNAL_PATH, np.random.randn(48000 * 2), 48000) # 2 seg de ruido blanco
    
    # Crear un dummy metadata.csv si no existe para permitir que el script se ejecute
    # En un entorno real, este archivo lo genera simulation.py
    if not os.path.exists(METADATA_FILENAME):
        print(f"Advertencia: Archivo de metadatos {METADATA_FILENAME} no encontrado. Creando dummy.")
        dummy_meta_data = [{
            'config_id': 'dummy_cfg1', 'fs_hz': 48000, 'room_dim_x': 5, 'room_dim_y': 4, 'room_dim_z': 3,
            'rt60_target_s': 0.5, 'is_anechoic': False, 
            'source_pos_x': 1, 'source_pos_y': 1, 'source_pos_z': 1.5,
            'array_center_x': 2.5, 'array_center_y': 2, 'array_center_z': 1.5, 
            'actual_dist_src_to_array_center_m': 2.0, 'actual_azimuth_src_to_array_center_deg': 45.0,
            'num_mics_in_array': 2, 'mic_separation_m': 0.1, 'rir_file_basename': 'dummy_rir_cfg1',
            'mic0_pos_x': 2.45, 'mic0_pos_y': 2, 'mic0_pos_z': 1.5,
            'mic1_pos_x': 2.55, 'mic1_pos_y': 2, 'mic1_pos_z': 1.5
        }]
        # Crear directorio si no existe
        os.makedirs(RIR_DATASET_DIR, exist_ok=True)
        pd.DataFrame(dummy_meta_data).to_csv(METADATA_FILENAME, index=False)
        # Crear dummy RIR files para el dummy metadata
        # Esto es solo para que el script no falle por archivos faltantes en una ejecución de prueba inicial.
        if not os.path.exists(os.path.join(RIR_DATASET_DIR, 'dummy_rir_cfg1_micidx_0.wav')):
            sf.write(os.path.join(RIR_DATASET_DIR, 'dummy_rir_cfg1_micidx_0.wav'), np.random.randn(100), 48000)
            sf.write(os.path.join(RIR_DATASET_DIR, 'dummy_rir_cfg1_micidx_1.wav'), np.random.randn(100), 48000)

    process_simulation_data()
