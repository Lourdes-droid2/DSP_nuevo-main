import pyroomacoustics as pra
import numpy as np
import soundfile as sf
import os
import random # Mantendremos random por si el usuario quiere reintroducir aleatoriedad controlada más tarde

def create_rir_example(base_output_filename, rt60_tgt, room_dim, source_pos, mic_positions, fs):
    """
    Creates Room Impulse Responses (RIRs) for multiple microphones and saves each to a separate WAV file.

    Parameters:
    base_output_filename (str): Base path to save the WAV files (mic index will be appended).
    rt60_tgt (float): Target reverberation time in seconds.
    room_dim (list): Dimensions of the room [x, y, z] in meters.
    source_pos (list): Position of the sound source [x, y, z] in meters.
    mic_positions (list of lists): List of microphone positions, e.g., [[x1,y1,z1], [x2,y2,z2]].
    fs (int): Sampling frequency in Hz.
    """
    files_created_count = 0
    try:
        # Validate source position
        margin = 0.01
        for i in range(3): # x, y, z
            if not (margin <= source_pos[i] < room_dim[i] - margin):
                print(f"Skipping configuration for base {base_output_filename}: Source position {source_pos} is outside room dimensions {room_dim} or too close to walls.")
                return files_created_count

        # Validate microphone positions
        original_mic_indices_to_process = []
        mic_positions_for_pra = []

        for idx, mic_coord in enumerate(mic_positions):
            valid_for_dim = True
            for i in range(3): # x, y, z
                if not (margin <= mic_coord[i] < room_dim[i] - margin):
                    print(f"INFO: Mic {idx} for base {base_output_filename}: Position {mic_coord} is outside room dimensions {room_dim} or too close to walls. It will be skipped.")
                    valid_for_dim = False
                    break
            if not valid_for_dim:
                continue

            if np.linalg.norm(np.array(source_pos) - np.array(mic_coord)) < 0.1: # At least 0.1m apart
                print(f"INFO: Mic {idx} for base {base_output_filename}: Position {mic_coord} is too close to source {source_pos}. It will be skipped.")
                continue

            # If valid so far
            original_mic_indices_to_process.append(idx)
            mic_positions_for_pra.append(mic_coord)

        if not mic_positions_for_pra: # Renamed from valid_mic_positions
            print(f"Skipping configuration for base {base_output_filename}: No valid microphone positions remaining after validation.")
            return files_created_count

        # Calculate absorption and max_order from rt60_tgt and room_dim
        # This calculation might fail if rt60_tgt is too large for the room size/absorption.
        e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

        # Create the shoebox room
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.Material(e_absorption),
            max_order=max_order
        )

        # Add a source
        room.add_source(source_pos)

        # Add microphones
        mic_array_pos = np.array(mic_positions_for_pra).T
        room.add_microphone_array(pra.MicrophoneArray(mic_array_pos, fs=room.fs))

        # Compute the RIRs
        room.compute_rir()

        # Save RIR for each microphone that was processed by pyroomacoustics
        # room.rir is a list of lists: room.rir[pyroom_mic_idx][src_idx]
        # We have 1 source (src_idx=0) and multiple mics
        # The pyroom_mic_idx corresponds to the order in mic_positions_for_pra
        for pyroom_mic_idx in range(len(mic_positions_for_pra)):
            original_mic_idx = original_mic_indices_to_process[pyroom_mic_idx] # Get original index for filename
            rir_signal = room.rir[pyroom_mic_idx][0]

            # Construct individual filename using the original microphone index
            path_parts = os.path.splitext(base_output_filename)
            output_filename_mic = f"{path_parts[0]}_micidx_{original_mic_idx}{path_parts[1]}"

            sf.write(output_filename_mic, rir_signal, fs)
            print(f"Successfully created RIR: {output_filename_mic}")
            files_created_count += 1

    except ValueError as ve:
        print(f"Error processing configuration for base {base_output_filename} (ValueError, possibly rt60 related or no mics left): {ve}")
        print(f"Parameters: rt60_tgt={rt60_tgt}, room_dim={room_dim}")
    except Exception as e:
        print(f"Error processing configuration for base {base_output_filename} (General Error): {e}")

    return files_created_count

if __name__ == "__main__":
    # Script renamed from create_rir_dataset.py to simulation.py
    print("simulation.py (formerly create_rir_dataset.py) script started with user-defined configurations.")

    # --- Configuration ---
    fs = 44100  # Sampling frequency
    output_directory = "rir_dataset_user_defined" # Manteniendo el mismo dir de salida por ahora
    os.makedirs(output_directory, exist_ok=True)

    # --- User-defined Configurations ---
    # El usuario debe poblar esta lista con diccionarios.
    # Cada diccionario define una simulación de RIR.
    # Claves requeridas:
    #   "rt60_tgt": float (e.g., 0.5)
    #   "room_dim": list [x, y, z] (e.g., [5, 4, 3])
    #   "source_pos": list [x, y, z] (e.g., [1, 1, 1.5])
    #   "mic_positions": list of lists [[x1,y1,z1], [x2,y2,z2], ...] (e.g., [[2,2,1.5], [2.5,2,1.5]])
    # Clave opcional:
    #   "filename_suffix": str (e.g., "my_specific_setup") - se añadirá al nombre del archivo base.

    configurations = [
        # Ejemplos (el usuario deberá modificar o añadir los suyos):
        {
            "rt60_tgt": 0.3,
            "room_dim": [5, 4, 2.8],
            "source_pos": [1.5, 1.0, 1.2],
            "mic_positions": [[3.5, 3.0, 1.5], [3.8, 3.0, 1.5]], # Dos micrófonos
            "filename_suffix": "small_room_short_rt60_2mics"
        },
        {
            "rt60_tgt": 0.7,
            "room_dim": [8, 6, 3.5],
            "source_pos": [2.0, 1.5, 1.7],
            "mic_positions": [[6.0, 4.5, 1.7]], # Un micrófono
            "filename_suffix": "medium_room_long_rt60_1mic"
        },
        {
            "rt60_tgt": 0.5,
            "room_dim": [6, 5, 3.0],
            "source_pos": [1, 1, 1.5],
            "mic_positions": [[5, 4, 1.5], [1.5, 4, 1.5], [3, 2.5, 2.0]], # Tres micrófonos
                                      # No suffix, usará config_idx
        },
        # Ejemplo de configuración con un micrófono fuera (se omitirá ese micrófono, los otros válidos se procesarán)
        {
            "rt60_tgt": 0.4,
            "room_dim": [4, 3, 2.5],
            "source_pos": [1, 1, 1],
            "mic_positions": [[2, 2, 1], [4.5, 2, 1], [1.5, 1.5, 1.5]], # El segundo mic está fuera
            "filename_suffix": "one_mic_outside"
        },
        # Ejemplo de configuración donde todos los micrófonos están muy cerca de la fuente
        {
            "rt60_tgt": 0.4,
            "room_dim": [4, 3, 2.5],
            "source_pos": [1, 1, 1],
            "mic_positions": [[1.05, 1.05, 1.05], [1.06, 1.06, 1.06]], # Ambos muy cerca
            "filename_suffix": "all_mics_too_close"
        },
        # Ejemplo de configuración con fuente fuera de la sala (se omitirá toda la configuración)
         {
            "rt60_tgt": 0.4,
            "room_dim": [4, 3, 2.5],
            "source_pos": [4.5, 1, 1], # Fuente fuera
            "mic_positions": [[2.0, 2.0, 1.0]],
            "filename_suffix": "source_outside"
        }
    ]

    # --- Dataset Generation Loop ---
    total_individual_rirs_generated = 0
    if not configurations:
        print("No configurations provided. Please add configurations to the 'configurations' list in the script.")
    else:
        for i, config in enumerate(configurations):
            print(f"\nProcessing configuration {i+1}/{len(configurations)}:")

            rt60 = config.get("rt60_tgt")
            room_dim = config.get("room_dim")
            source_pos = config.get("source_pos")
            mic_positions = config.get("mic_positions") # Cambiado de mic_pos
            suffix = config.get("filename_suffix", f"config_{i+1}")

            # Validar que los parámetros esenciales están presentes
            if not all([rt60 is not None, room_dim, source_pos, mic_positions is not None]): # mic_positions puede ser lista vacía teóricamente
                print(f"Skipping configuration {i+1}: Missing one or more required parameters (rt60_tgt, room_dim, source_pos, mic_positions).")
                continue
            if not mic_positions: # Si la lista de mic_positions está vacía
                 print(f"Skipping configuration {i+1}: mic_positions list is empty.")
                 continue

            print(f"  RT60: {rt60}s, Room: {room_dim}, Source: {source_pos}, Mic_Positions: {mic_positions}")

            # Construct base filename (sin el índice del micrófono, eso se añade en create_rir_example)
            # Usamos solo el primer mic para el nombre base representativo, o podríamos omitirlo del nombre base.
            # Por simplicidad, mantenemos una estructura similar.
            base_filename = f"rir_rt60_{rt60}_room_{room_dim[0]}x{room_dim[1]}x{room_dim[2]}_src_{source_pos[0]}x{source_pos[1]}x{source_pos[2]}_{suffix}.wav"
            base_filename = base_filename.replace(" ", "_").replace("[", "").replace("]", "").replace(",", "") # Sanitización básica

            base_output_filepath = os.path.join(output_directory, base_filename)

            num_created_for_this_config = create_rir_example(
                base_output_filename=base_output_filepath,
                rt60_tgt=rt60,
                room_dim=room_dim,
                source_pos=source_pos,
                mic_positions=mic_positions, # Pasando la lista de posiciones
                fs=fs
            )
            total_individual_rirs_generated += num_created_for_this_config


    print(f"\n--- Dataset generation complete ---")
    print(f"Total individual RIR files successfully generated: {total_individual_rirs_generated}")
    print(f"Dataset saved in: {output_directory}")