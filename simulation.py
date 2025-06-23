import pyroomacoustics as pra
import numpy as np
import soundfile as sf
import os
import random
import csv

def calculate_angle_and_distance(source_pos, array_center):
    """Calculates distance and azimuth from source to array center."""
    delta = np.array(source_pos) - np.array(array_center)
    distance = np.linalg.norm(delta)
    azimuth = np.degrees(np.arctan2(delta[1], delta[0])) # Azimuth in XY plane
    return distance, azimuth

def create_rir_example(base_output_filename, rt60_tgt, room_dim, source_pos, mic_positions, fs, is_anechoic=False):
    """
    Creates Room Impulse Responses (RIRs) and saves them to WAV files.
    Returns the count of successfully created RIR files and a list of actual microphone positions used.
    """
    files_created_count = 0
    actual_mic_positions_in_room = []
    try:
        # Pyroomacoustics margin for objects not to be exactly on the wall
        margin = 0.01 

        # Basic validation for room dimensions
        if not all(d > 2 * margin for d in room_dim):
            # This print is for debugging, can be removed or logged if too verbose
            # print(f"INFO: Skipping config for {base_output_filename}. Room dimensions {room_dim} too small.")
            return files_created_count, actual_mic_positions_in_room

        # Validate source position against exact room boundaries + margin
        if not all(margin <= source_pos[i] < room_dim[i] - margin for i in range(3)):
            # print(f"INFO: Skipping config for {base_output_filename}. Source {source_pos} outside room {room_dim} (considering margin {margin}).")
            return files_created_count, actual_mic_positions_in_room

        original_mic_indices_to_process = []
        mic_positions_for_pra = []

        for idx, mic_coord in enumerate(mic_positions):
            # Validate mic position against exact room boundaries + margin
            if not all(margin <= mic_coord[i] < room_dim[i] - margin for i in range(3)):
                # This individual mic is skipped
                continue 
            
            # Validate mic distance from source (minimal distance for PRA stability/realism)
            if np.linalg.norm(np.array(source_pos) - np.array(mic_coord)) < 0.1: # Min 10cm separation
                # This individual mic is skipped
                continue
            
            original_mic_indices_to_process.append(idx)
            mic_positions_for_pra.append(mic_coord)

        if len(mic_positions_for_pra) < 1: # Need at least one valid microphone
            # print(f"INFO: Skipping config for {base_output_filename}. No valid microphones after PRA pre-check.")
            return files_created_count, actual_mic_positions_in_room

        # Determine absorption and max_order
        # For anechoic, RT60 is very low. inverse_sabine still needs a positive RT60.
        # The max_order=0 is what makes it anechoic.
        current_max_order = 0
        if is_anechoic:
            # Use a small rt60_tgt for inverse_sabine to get some absorption value if needed, though max_order=0 is key.
            e_absorption, _ = pra.inverse_sabine(0.05, room_dim) 
            current_max_order = 0
        else:
            e_absorption, current_max_order = pra.inverse_sabine(rt60_tgt, room_dim)

        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.Material(e_absorption),
            max_order=current_max_order # Use the determined max_order
        )
        room.add_source(source_pos)
        
        mic_array_for_pra_np = np.array(mic_positions_for_pra).T
        room.add_microphone_array(pra.MicrophoneArray(mic_array_for_pra_np, fs=room.fs))
        
        actual_mic_positions_in_room = mic_positions_for_pra # Store mics passed to PRA

        room.compute_rir()

        for pyroom_mic_idx in range(len(mic_positions_for_pra)):
            original_mic_idx_in_config = original_mic_indices_to_process[pyroom_mic_idx]
            rir_signal = room.rir[pyroom_mic_idx][0]
            
            path_parts = os.path.splitext(base_output_filename)
            output_filename_mic = f"{path_parts[0]}_micidx_{original_mic_idx_in_config}{path_parts[1]}"
            
            sf.write(output_filename_mic, rir_signal, fs)
            files_created_count += 1

    except ValueError as ve: # Catches issues from inverse_sabine (e.g. absorption > 1) or other PRA value errors
        print(f"ERROR: PRA ValueError for {base_output_filename}: {ve}. Params: RT60={rt60_tgt}, Room={room_dim}")
    except Exception as e:
        print(f"ERROR: Generic error in create_rir_example for {base_output_filename}: {e}")
        
    return files_created_count, actual_mic_positions_in_room

if __name__ == "__main__":
    print("--- simulation.py: Iniciando generación de RIRs y metadatos ---")

    FS = 48000
    OUTPUT_DIR = "rir_dataset_user_defined"
    METADATA_FILENAME = "simulation_metadata.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Acoustic/Placement Parameters
    min_wall_dist = 0.5  # Minimum distance from any source/mic to any wall
    min_src_mic_dist = 0.5 # Minimum distance between a source and any microphone

    user_defined_simulations = []

    # Configuraciones manuales
    user_defined_simulations.extend([
        {
            "config_id_suffix": "sala_pequena_reverb", "room_dim": [5.0, 4.0, 2.8],"rt60_tgt": 0.4,
            "is_anechoic": False, "array_center_target": [2.5, 2.0, 1.5], "array_orientation_axis": 'x',
            "source_pos": [1.0, 1.0, 1.5], "num_mics": 4
        },
        {
            "config_id_suffix": "fuente_izquierda", "room_dim": [6.0, 4.0, 2.5], "rt60_tgt": 0.4,
            "is_anechoic": False, "array_center_target": [4.5, 2.0, 1.5], "array_orientation_axis": 'x',
            "source_pos": [1.0, 2.0, 1.5], "num_mics": 4
        },
        {
            "config_id_suffix": "sala_con_8mics", "room_dim": [6.0, 5.0, 2.5], "rt60_tgt": 0.6,
            "is_anechoic": False, "array_center_target": [3.0, 2.5, 1.5], "array_orientation_axis": 'x',
            "source_pos": [1.0, 2.5, 1.5], "num_mics": 8
        }
    ])

    # Simulaciones aleatorias
    num_random_configs = 10
    print(f"Generando {num_random_configs} configuraciones aleatorias...")
    for i in range(num_random_configs):
        # Robust room dimension generation
        min_dim_component = 2 * min_wall_dist + 0.1 # e.g., 1.1m if min_wall_dist=0.5. Ensures space.
        
        # Ensure room dimensions are sampled between 2.0 and 25.0 AND meet min_dim_component requirement
        dim_x = round(np.random.uniform(max(2.0, min_dim_component), 25.0), 2)
        dim_y = round(np.random.uniform(max(2.0, min_dim_component), 25.0), 2)
        dim_z = round(np.random.uniform(max(2.0, min_dim_component), 25.0), 2)
        room_dim_vals = [dim_x, dim_y, dim_z]

        rt60_val = round(np.random.uniform(0.2, 0.8), 2)

        array_center_val = [
            round(np.random.uniform(min_wall_dist, room_dim_vals[0] - min_wall_dist), 2),
            round(np.random.uniform(min_wall_dist, room_dim_vals[1] - min_wall_dist), 2),
            round(np.random.uniform(min_wall_dist, room_dim_vals[2] - min_wall_dist), 2)
        ]
        
        source_pos_val = array_center_val[:] # Initialize, then ensure distance
        attempts = 0
        max_attempts_random_pos = 100
        while np.linalg.norm(np.array(source_pos_val) - np.array(array_center_val)) < min_src_mic_dist and attempts < max_attempts_random_pos:
            source_pos_val = [
                round(np.random.uniform(min_wall_dist, room_dim_vals[0] - min_wall_dist), 2),
                round(np.random.uniform(min_wall_dist, room_dim_vals[1] - min_wall_dist), 2),
                round(np.random.uniform(min_wall_dist, room_dim_vals[2] - min_wall_dist), 2)
            ]
            attempts += 1
        
        if attempts >= max_attempts_random_pos: 
            print(f"WARN: Could not place source sufficiently far ({min_src_mic_dist}m) from array center for random_config_{i} after {max_attempts_random_pos} attempts. Skipping this random config.")
            continue

        user_defined_simulations.append({
            "config_id_suffix": f"random_config_{i}", "room_dim": room_dim_vals, "rt60_tgt": rt60_val,
            "is_anechoic": False, "array_center_target": array_center_val, 
            "array_orientation_axis": random.choice(['x', 'y']), "source_pos": source_pos_val, "num_mics": 4
        })

    all_metadata_entries = []
    total_rirs_generated_overall = 0
    successful_configurations_count = 0

    # Determine the maximum number of configured mics across all simulations for CSV header
    max_configured_mics_overall = 0
    if user_defined_simulations:
        for sim_conf_to_find_max_mics in user_defined_simulations:
            max_configured_mics_overall = max(max_configured_mics_overall, sim_conf_to_find_max_mics.get("num_mics", 4))
    if max_configured_mics_overall == 0 and user_defined_simulations: # Fallback if all configs somehow had 0 mics explicitly
        max_configured_mics_overall = 4 # Default to 4 if calculation failed but there are sims

    print(f"\n--- Iniciando procesamiento de {len(user_defined_simulations)} configuraciones totales ---")
    for idx, config in enumerate(user_defined_simulations):
        config_name_for_print = config.get('config_id_suffix', f'unnamed_config_{idx}')
        print(f"\nProcesando configuración {idx+1}/{len(user_defined_simulations)}: {config_name_for_print}")

        NUM_MICS = config.get("num_mics", 4) # Default to 4 mics if not specified
        MIC_SEPARATION = 0.10  # 10 cm, fixed for now
        ARRAY_LENGTH = (NUM_MICS - 1) * MIC_SEPARATION

        room_dim_current = config["room_dim"]
        rt60_current = config["rt60_tgt"]
        is_anechoic_current = config["is_anechoic"]
        array_center_config_current = config["array_center_target"] 
        orientation_current = config["array_orientation_axis"].lower()
        source_pos_current = config["source_pos"]
        
        axis_map = {'x':0, 'y':1, 'z':2}
        axis_idx_current = axis_map.get(orientation_current)

        if axis_idx_current is None:
            print(f"INFO: Config {config_name_for_print} skipped. Orientación de array inválida: {orientation_current}")
            continue
        
        if room_dim_current[axis_idx_current] < ARRAY_LENGTH + 2 * min_wall_dist:
            print(f"INFO: Config {config_name_for_print} skipped. Longitud de array ({ARRAY_LENGTH:.2f}m) + distancias a paredes ({2*min_wall_dist:.2f}m) excede dimensión de sala ({room_dim_current[axis_idx_current]:.2f}m) en eje '{orientation_current}'.")
            continue

        mic_positions_calculated = []
        half_array_len = ((NUM_MICS - 1) / 2.0) * MIC_SEPARATION
        for i in range(NUM_MICS):
            offset = i * MIC_SEPARATION - half_array_len
            pos = list(array_center_config_current) 
            pos[axis_idx_current] = round(array_center_config_current[axis_idx_current] + offset, 2)
            mic_positions_calculated.append(pos)

        current_config_valid = True
        # Validate all points (source and calculated mics) against min_wall_dist
        combined_points_for_validation = mic_positions_calculated + [source_pos_current]
        for p_idx, p_coord in enumerate(combined_points_for_validation):
            is_source = (p_idx == len(mic_positions_calculated)) 
            point_label = "Fuente" if is_source else f"Micrófono calculado {p_idx}"
            for dim_idx_val, dim_name in enumerate(['x', 'y', 'z']): 
                if not (min_wall_dist <= p_coord[dim_idx_val] < room_dim_current[dim_idx_val] - min_wall_dist):
                    print(f"INFO: Config {config_name_for_print} skipped. {point_label} pos {p_coord} viola min_wall_dist ({min_wall_dist}m) en eje '{dim_name}' de sala {room_dim_current}.")
                    current_config_valid = False; break
            if not current_config_valid: break
        if not current_config_valid: continue

        # Validate all calculated mics against min_src_mic_dist
        for mic_idx_val, mp_coord_val in enumerate(mic_positions_calculated):
            if np.linalg.norm(np.array(mp_coord_val) - np.array(source_pos_current)) < min_src_mic_dist:
                print(f"INFO: Config {config_name_for_print} skipped. Micrófono calc. {mic_idx_val} pos {mp_coord_val} muy cerca de fuente {source_pos_current} (min_src_mic_dist: {min_src_mic_dist}m).")
                current_config_valid = False; break
        if not current_config_valid: continue

        config_id_str_current = f"{config_name_for_print}_{'anechoic' if is_anechoic_current else f'rt{rt60_current:.2f}'}"
        rir_base_filename_str_current = os.path.join(OUTPUT_DIR, f"rir_{config_id_str_current}")

        rirs_created_for_config, mic_positions_used_by_pra = create_rir_example(
            rir_base_filename_str_current, rt60_current, room_dim_current, source_pos_current, 
            mic_positions_calculated, FS, is_anechoic_current
        )

        if rirs_created_for_config == 0:
             print(f"INFO: Config {config_id_str_current} resultó en 0 RIRs de create_rir_example. Omitiendo metadatos para esta.")
             continue
        
        if rirs_created_for_config < NUM_MICS:
            print(f"INFO: Config {config_id_str_current} creó {rirs_created_for_config}/{NUM_MICS} RIRs (incompleto). Omitiendo metadatos y RIRs parciales.")
            # Clean up partially created files for this configuration
            for i_mic_cleanup in range(NUM_MICS): # Check all potential original indices
                 temp_fn_cleanup = f"{rir_base_filename_str_current}_micidx_{i_mic_cleanup}.wav"
                 if os.path.exists(temp_fn_cleanup):
                     try:
                         os.remove(temp_fn_cleanup)
                     except OSError as e_remove:
                         print(f"WARN: No se pudo eliminar el archivo RIR parcial {temp_fn_cleanup}: {e_remove}")
            continue

        total_rirs_generated_overall += rirs_created_for_config
        successful_configurations_count +=1
        
        actual_array_center_coords = []
        dist_src_actual_val, azimuth_src_actual_val = "N/A", "N/A" # Placeholders
        if mic_positions_used_by_pra: # Should be true if rirs_created_for_config > 0
            actual_array_center_coords = list(np.mean(mic_positions_used_by_pra, axis=0))
            dist_src_actual_val, azimuth_src_actual_val = calculate_angle_and_distance(source_pos_current, actual_array_center_coords)
        
        entry = {
            "config_id": config_id_str_current, "fs_hz": FS,
            "room_dim_x": room_dim_current[0], "room_dim_y": room_dim_current[1], "room_dim_z": room_dim_current[2],
            "rt60_target_s": rt60_current, "is_anechoic": is_anechoic_current,
            "source_pos_x": source_pos_current[0], "source_pos_y": source_pos_current[1], "source_pos_z": source_pos_current[2],
            "array_center_x_config": array_center_config_current[0], 
            "array_center_y_config": array_center_config_current[1], 
            "array_center_z_config": array_center_config_current[2],
            "array_orientation_axis": orientation_current,
            "num_mics_configured": NUM_MICS, "num_mics_processed": rirs_created_for_config,
            "mic_separation_m": MIC_SEPARATION,
            "rir_file_basename": os.path.basename(rir_base_filename_str_current)
        }

        if actual_array_center_coords: # Check if list is not empty
            entry["array_center_x_actual"] = round(actual_array_center_coords[0], 2)
            entry["array_center_y_actual"] = round(actual_array_center_coords[1], 2)
            entry["array_center_z_actual"] = round(actual_array_center_coords[2], 2)
            entry["actual_dist_src_to_array_center_m"] = round(dist_src_actual_val, 3)
            entry["actual_azimuth_src_to_array_center_deg"] = round(azimuth_src_actual_val, 2)
        else: # Fill with N/A if actual center couldn't be calculated
            entry["array_center_x_actual"] = "N/A"; entry["array_center_y_actual"] = "N/A"; entry["array_center_z_actual"] = "N/A"
            entry["actual_dist_src_to_array_center_m"] = "N/A"; entry["actual_azimuth_src_to_array_center_deg"] = "N/A"

        for i_mic_meta, mic_c_meta in enumerate(mic_positions_used_by_pra):
            entry[f"mic{i_mic_meta}_pos_x"] = round(mic_c_meta[0], 2)
            entry[f"mic{i_mic_meta}_pos_y"] = round(mic_c_meta[1], 2)
            entry[f"mic{i_mic_meta}_pos_z"] = round(mic_c_meta[2], 2)
        
        all_metadata_entries.append(entry)
        print(f"INFO: Configuración {config_id_str_current} procesada exitosamente con {rirs_created_for_config} RIRs.")


    # CSV Writing
    if all_metadata_entries:
        metadata_filepath = os.path.join(OUTPUT_DIR, METADATA_FILENAME)
        
        ordered_base_fieldnames = [
            "config_id", "fs_hz", "room_dim_x", "room_dim_y", "room_dim_z", "rt60_target_s", "is_anechoic",
            "source_pos_x", "source_pos_y", "source_pos_z", "array_center_x_config", "array_center_y_config", 
            "array_center_z_config", "array_center_x_actual", "array_center_y_actual", "array_center_z_actual",
            "array_orientation_axis", "actual_dist_src_to_array_center_m", "actual_azimuth_src_to_array_center_deg",
            "num_mics_configured", "num_mics_processed", "mic_separation_m", "rir_file_basename"
        ]
        
        # Dynamically discover any other keys that might exist in entries (should be none if structure is fixed)
        temp_all_keys_in_entries = set()
        for entry_data_keys in all_metadata_entries:
            temp_all_keys_in_entries.update(entry_data_keys.keys())
        
        additional_base_fieldnames = sorted([
            k_add for k_add in temp_all_keys_in_entries 
            if k_add not in ordered_base_fieldnames and not (k_add.startswith("mic") and "_pos_" in k_add)
        ])
        final_base_fieldnames_csv = ordered_base_fieldnames + additional_base_fieldnames
        
        mic_coord_fieldnames_csv = []
        if max_configured_mics_overall > 0 : # Only add mic fields if there's a possibility of mics
            for i_mic_header in range(max_configured_mics_overall): 
                mic_coord_fieldnames_csv.append(f"mic{i_mic_header}_pos_x")
                mic_coord_fieldnames_csv.append(f"mic{i_mic_header}_pos_y")
                mic_coord_fieldnames_csv.append(f"mic{i_mic_header}_pos_z")
        
        final_fieldnames_for_csv = final_base_fieldnames_csv + mic_coord_fieldnames_csv

        if not final_fieldnames_for_csv:
            print("ERROR: No se pudieron determinar los nombres de campo para el CSV de metadatos, aunque existen entradas.")
        else:
            try:
                with open(metadata_filepath, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=final_fieldnames_for_csv, quoting=csv.QUOTE_NONNUMERIC, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(all_metadata_entries)
                print(f"Metadatos guardados en: {metadata_filepath}")
            except IOError as e_io:
                print(f"ERROR: No se pudo escribir el archivo CSV de metadatos {metadata_filepath}: {e_io}")
            except Exception as e_csv:
                print(f"ERROR: Ocurrió un error al escribir el CSV de metadatos: {e_csv}")
    else:
        print("No se generaron entradas de metadatos.")

    print(f"\nTotal de RIRs individuales generadas: {total_rirs_generated_overall}")
    print(f"Configuraciones procesadas exitosamente (completas): {successful_configurations_count}")
    print("--- simulation.py: Finalizado ---")
