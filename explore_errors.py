import pandas as pd
import numpy as np

RESULTS_CSV_PATH = "full_experiment_results.csv"

print(f"--- Análisis Exploratorio de Errores de {RESULTS_CSV_PATH} ---")

if not pd.io.common.file_exists(RESULTS_CSV_PATH):
    print(f"Error: Archivo de resultados no encontrado: {RESULTS_CSV_PATH}")
else:
    try:
        df = pd.read_csv(RESULTS_CSV_PATH)
        print(f"Resultados cargados: {len(df)} filas.")

        if df.empty:
            print("El DataFrame está vacío.")
        else:
            # --- Análisis de TDOA Errors ---
            if 'tdoa_error_s' in df.columns:
                print("\n--- Descripción de 'tdoa_error_s' (Error TDOA en segundos) ---")
                # Convertir a numérico, errores a NaN
                df['tdoa_error_s_numeric'] = pd.to_numeric(df['tdoa_error_s'], errors='coerce')
                tdoa_errors = df['tdoa_error_s_numeric'].dropna()
                if not tdoa_errors.empty:
                    print(tdoa_errors.describe())
                    print("\nPercentiles de error TDOA (absoluto):")
                    tdoa_abs_errors = tdoa_errors.abs()
                    print(tdoa_abs_errors.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]))

                    # Contar NaNs originales y después de coerción
                    original_nans_tdoa = df['tdoa_error_s'].isna().sum()
                    coerced_nans_tdoa = df['tdoa_error_s_numeric'].isna().sum()
                    print(f"NaNs originales en tdoa_error_s: {original_nans_tdoa}")
                    if coerced_nans_tdoa > original_nans_tdoa:
                        print(f"NaNs adicionales por coerción a numérico en tdoa_error_s: {coerced_nans_tdoa - original_nans_tdoa}")

                else:
                    print("No hay datos válidos de tdoa_error_s para analizar después de limpiar NaNs.")
            else:
                print("Columna 'tdoa_error_s' no encontrada.")

            # --- Análisis de DOA Array Errors ---
            if 'doa_array_error_deg' in df.columns:
                print("\n--- Descripción de 'doa_array_error_deg' (Error DOA de Array en grados) ---")
                 # Convertir a numérico, errores a NaN
                df['doa_array_error_deg_numeric'] = pd.to_numeric(df['doa_array_error_deg'], errors='coerce')
                doa_errors = df['doa_array_error_deg_numeric'].dropna()

                if not doa_errors.empty:
                    print("Errores DOA (como están en el CSV):")
                    print(doa_errors.describe())

                    print("\nErrores DOA (normalizados a [-180, 180) antes de abs):")
                    doa_errors_normalized = (doa_errors + 180) % 360 - 180
                    print(doa_errors_normalized.describe())

                    print("\nPercentiles de error DOA (absoluto, después de normalizar error a [-180, 180)):")
                    doa_abs_errors_normalized = doa_errors_normalized.abs()
                    print(doa_abs_errors_normalized.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]))

                    original_nans_doa = df['doa_array_error_deg'].isna().sum()
                    coerced_nans_doa = df['doa_array_error_deg_numeric'].isna().sum()
                    print(f"NaNs originales en doa_array_error_deg: {original_nans_doa}")
                    if coerced_nans_doa > original_nans_doa:
                        print(f"NaNs adicionales por coerción a numérico en doa_array_error_deg: {coerced_nans_doa - original_nans_doa}")
                else:
                    print("No hay datos válidos de doa_array_error_deg para analizar después de limpiar NaNs.")
            else:
                print("Columna 'doa_array_error_deg' no encontrada.")

    except pd.errors.EmptyDataError:
        print(f"Error: El archivo de resultados {RESULTS_CSV_PATH} está vacío o corrupto.")
    except Exception as e:
        print(f"Ocurrió un error al procesar el archivo de resultados: {e}")

print("\n--- Fin del Análisis Exploratorio ---")
