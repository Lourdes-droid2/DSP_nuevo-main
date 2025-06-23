import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paletas de colores sugeridas (ejemplos)
# Para múltiples métodos:
# QUALITATIVE_PALETTE = sns.color_palette("Paired", 10)
# Para SNR o RT60 (secuencial):
# SEQUENTIAL_PALETTE = sns.color_palette("viridis", as_cmap=True)

DEFAULT_FIG_SIZE = (10, 6) # Reducido de (12,7) para un tamaño más estándar inicial
DEFAULT_DPI = 100 # Reducido de 150 para pruebas más rápidas, se puede aumentar para finales
DEFAULT_TITLE_FONTSIZE = 14
DEFAULT_LABEL_FONTSIZE = 12
DEFAULT_TICK_FONTSIZE = 10

def _calculate_ylim_from_percentiles(data_series, low_percentile, high_percentile, min_range=0.1):
    """Calcula los límites del eje Y basados en percentiles, asegurando un rango mínimo."""
    if data_series.empty:
        return 0, min_range # Rango por defecto si no hay datos

    # Asegurar que los percentiles estén en el rango [0, 100]
    low_percentile = np.clip(low_percentile, 0, 100)
    high_percentile = np.clip(high_percentile, 0, 100)
    if low_percentile >= high_percentile: # Asegurar que bajo < alto
        low_percentile = max(0, high_percentile - 1)

    min_val = np.percentile(data_series, low_percentile)
    max_val = np.percentile(data_series, high_percentile)

    if pd.isna(min_val) or pd.isna(max_val): # Si los percentiles dan NaN (pocos datos)
        min_val = data_series.min()
        max_val = data_series.max()

    if min_val == max_val: # Si todos los valores en el rango del percentil son iguales
        # Expandir un poco para que no sea una línea plana si es posible
        # Si es cero, expandir a un rango pequeño. Si no, expandir un % del valor.
        if min_val == 0:
            min_val = 0
            max_val = min_range
        else:
            expansion = abs(min_val * 0.1) if min_val != 0 else min_range
            min_val -= expansion
            max_val += expansion

    # Asegurar un rango mínimo visible
    if (max_val - min_val) < min_range:
        mid_point = (max_val + min_val) / 2
        max_val = mid_point + min_range / 2
        min_val = mid_point - min_range / 2

    # Añadir un pequeño margen
    margin = (max_val - min_val) * 0.05
    # Evitar margen cero si el rango es diminuto pero no cero, o si todo es cero
    if margin < 1e-9 :
        if max_val > 1e-9 or min_val < -1e-9: # Si el rango original no era cero
            margin = (abs(max_val) + abs(min_val) + 1e-9) * 0.05
        else: # Si el rango original era prácticamente cero en cero
             margin = min_range * 0.05


    final_bottom = min_val - margin
    final_top = max_val + margin

    # Para errores absolutos, el límite inferior no debe ser menor que 0.
    # Esto se manejará fuera, ya que esta función es genérica para data_series.

    if final_top <= final_bottom: # Última verificación
        final_top = final_bottom + min_range

    return final_bottom, final_top


def create_error_plot(
    df_full,
    x_col,
    y_col, # Columna de error original (ej. 'tdoa_error_s', 'doa_array_error_deg')
    title,
    output_filename,
    plot_type='boxplot', # 'boxplot', 'violinplot', 'lineplot_ci'
    hue_col=None, # Columna para agrupar por color/estilo (ej. 'tdoa_method')
    is_angular_error=False,
    ylim_strategy='percentile', # 'percentile', 'iqr', 'auto', o tupla (min, max)
    ylim_params={'low': 1, 'high': 99, 'min_range_abs': 0.1}, # Para 'percentile' o 'iqr'
                                                              # min_range_abs es para error absoluto
    param_bins=10, # Para discretizar x_col si es numérico y se usa box/violin
    x_label=None,
    y_label=None, # Etiqueta para el eje Y (ej. "Error Absoluto TDOA (s)")
    palette="Paired",
    fig_size=DEFAULT_FIG_SIZE,
    dpi=DEFAULT_DPI,
    title_fontsize=DEFAULT_TITLE_FONTSIZE,
    label_fontsize=DEFAULT_LABEL_FONTSIZE,
    tick_fontsize=DEFAULT_TICK_FONTSIZE,
    show_fliers_boxplot=False, # Para boxplot
    boxplot_whis_percentiles=(5,95), # Percentiles para los bigotes del boxplot
    lineplot_estimator='mean', # 'mean' o 'median' para lineplot_ci
    lineplot_ci=95, # Intervalo de confianza para lineplot_ci (o 'sd' para std dev)
    output_dir="analysis_plots"
):
    """
    Función mejorada para crear gráficos de error vs. un parámetro.
    """
    if y_col not in df_full.columns or x_col not in df_full.columns:
        print(f"ADVERTENCIA (create_error_plot): Columnas '{y_col}' o '{x_col}' no encontradas. Saltando gráfico: {title}")
        return
    if hue_col and hue_col not in df_full.columns:
        print(f"ADVERTENCIA (create_error_plot): Columna hue '{hue_col}' no encontrada. Se graficará sin hue. Gráfico: {title}")
        hue_col = None # Continuar sin hue

    df_plot = df_full.dropna(subset=[x_col, y_col]).copy()
    if df_plot.empty:
        print(f"ADVERTENCIA (create_error_plot): No hay datos para graficar para '{title}' después de dropna inicial. Saltando.")
        return

    # 1. Procesar la columna de error (y_col) para obtener 'error_abs_processed'
    error_series_original = pd.to_numeric(df_plot[y_col], errors='coerce')
    if is_angular_error:
        error_series_normalized = (error_series_original + 180) % 360 - 180
        df_plot['error_abs_processed'] = error_series_normalized.abs()
    else:
        df_plot['error_abs_processed'] = error_series_original.abs()

    df_plot.replace([np.inf, -np.inf], np.nan, inplace=True) # Reemplazar Inf generados por abs() o cálculos
    df_plot.dropna(subset=['error_abs_processed', x_col], inplace=True) # Dropear NaNs en error procesado o x_col

    if df_plot.empty:
        print(f"ADVERTENCIA (create_error_plot): No hay datos válidos para 'error_abs_processed' en '{title}'. Saltando.")
        return

    # 2. Preparar figura y ejes
    plt.figure(figsize=fig_size)
    ax = plt.gca()

    # 3. Determinar límites del eje Y (ylim)
    final_ylim_bottom, final_ylim_top = None, None
    min_range_y = ylim_params.get('min_range_abs', 0.1) if y_label and "Absoluto" in y_label else ylim_params.get('min_range', 0.1)

    if isinstance(ylim_strategy, tuple) and len(ylim_strategy) == 2:
        final_ylim_bottom, final_ylim_top = ylim_strategy
    elif ylim_strategy == 'percentile':
        low_p = ylim_params.get('low', 1)
        high_p = ylim_params.get('high', 99)
        final_ylim_bottom, final_ylim_top = _calculate_ylim_from_percentiles(df_plot['error_abs_processed'], low_p, high_p, min_range_y)
    elif ylim_strategy == 'iqr':
        q1 = df_plot['error_abs_processed'].quantile(0.25)
        q3 = df_plot['error_abs_processed'].quantile(0.75)
        iqr_val = q3 - q1
        factor = ylim_params.get('factor', 1.5)
        final_ylim_bottom = q1 - factor * iqr_val
        final_ylim_top = q3 + factor * iqr_val
        # Asegurar un rango mínimo si IQR es muy pequeño
        if (final_ylim_top - final_ylim_bottom) < min_range_y :
             mid = (final_ylim_top + final_ylim_bottom) / 2
             final_ylim_top = mid + min_range_y/2
             final_ylim_bottom = mid - min_range_y/2
    # 'auto' no hace nada aquí, matplotlib lo manejará (pero puede ser afectado por outliers)

    if y_label and "Absoluto" in y_label : # Si es error absoluto, el límite inferior es 0
        if final_ylim_bottom is not None:
            final_ylim_bottom = max(0, final_ylim_bottom)
        else: # Para 'auto' y error absoluto
            final_ylim_bottom = 0


    # 4. Lógica de ploteo
    x_col_to_plot = x_col
    x_ticks_rotation = 0
    x_ticks_ha = 'center'

    is_x_numeric = pd.api.types.is_numeric_dtype(df_plot[x_col])

    if plot_type in ['boxplot', 'violinplot'] and is_x_numeric and df_plot[x_col].nunique() > param_bins:
        try:
            df_plot[x_col + '_binned'] = pd.cut(df_plot[x_col], bins=param_bins)
            x_col_to_plot = x_col + '_binned'
            x_ticks_rotation = 45
            x_ticks_ha = 'right'
        except Exception as e:
            print(f"  INFO (create_error_plot): No se pudo hacer binning para {x_col} en '{title}' (error: {e}). Usando {x_col} original.")
            # Si el binning falla, se usará x_col original, y si aún hay muchos valores únicos,
            # boxplot/violinplot podrían no ser ideales. lineplot_ci es más robusto en ese caso.
            if plot_type in ['boxplot', 'violinplot']: # Forzar a lineplot si el binning falla y es numérico
                 print(f"  INFO (create_error_plot): Forzando a 'lineplot_ci' para '{title}' debido a fallo de binning en x numérica.")
                 plot_type = 'lineplot_ci'


    if plot_type == 'boxplot':
        sns.boxplot(x=x_col_to_plot, y='error_abs_processed', data=df_plot, hue=hue_col,
                    showfliers=show_fliers_boxplot, palette=palette, ax=ax, whis=boxplot_whis_percentiles)
    elif plot_type == 'violinplot':
        sns.violinplot(x=x_col_to_plot, y='error_abs_processed', data=df_plot, hue=hue_col,
                       palette=palette, ax=ax, inner='quartile', cut=0) # cut=0 para no extender más allá de los datos
    elif plot_type == 'lineplot_ci':
        # lineplot es mejor si x_col_to_plot es numérico o categórico ordenado.
        # Si x_col_to_plot es binned (categórico), podría necesitar convertir los labels a puntos medios para un lineplot "real"
        # Por ahora, seaborn lo maneja bien si x es numérico. Si es categórico (binned), lo trata como tal.
        sns.lineplot(x=x_col_to_plot, y='error_abs_processed', data=df_plot, hue=hue_col,
                     estimator=lineplot_estimator, errorbar=('ci', lineplot_ci) if isinstance(lineplot_ci, int) else lineplot_ci,
                     palette=palette, ax=ax, legend='auto', marker='o', markersize=5)
    else:
        print(f"ADVERTENCIA (create_error_plot): Tipo de gráfico '{plot_type}' no reconocido para '{title}'. Usando scatterplot.")
        sns.scatterplot(x=x_col_to_plot, y='error_abs_processed', data=df_plot, hue=hue_col,
                        palette=palette, ax=ax, alpha=0.5, edgecolor=None, legend='auto')

    # 5. Configurar estética
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(x_label if x_label else x_col, fontsize=label_fontsize)
    ax.set_ylabel(y_label if y_label else 'Error Absoluto Procesado', fontsize=label_fontsize)

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if x_ticks_rotation > 0:
        plt.xticks(rotation=x_ticks_rotation, ha=x_ticks_ha)

    ax.grid(True, linestyle='--', alpha=0.6)

    if hue_col: # Mover la leyenda si existe
        ax.legend(title=hue_col, fontsize=tick_fontsize, title_fontsize=label_fontsize-1, loc='best')


    # Aplicar los límites Y calculados
    if final_ylim_bottom is not None and final_ylim_top is not None:
        ax.set_ylim(final_ylim_bottom, final_ylim_top)

    plt.tight_layout()

    # 6. Guardar gráfico
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, output_filename)
        plt.savefig(filepath, dpi=dpi)
        print(f"Gráfico guardado: {filepath}")
    except Exception as e:
        print(f"Error guardando gráfico {output_filename}: {e}")
    plt.close()

if __name__ == '__main__':
    print("--- Módulo plotting_utils.py ---")
    print("Este módulo contiene funciones para generar gráficos.")
    print("Ejemplo de cómo se podría usar (requiere un DataFrame 'dummy_df'):")

    # Crear un DataFrame dummy para probar la función
    # Debe hacerse solo si este script se ejecuta directamente para prueba
    # y no cuando es importado.

    # n_points = 200
    # dummy_data = {
    #     'snr_db': np.random.choice([-5, 0, 5, 10, 15, 20], size=n_points),
    #     'rt60': np.random.choice([0.3, 0.5, 0.7], size=n_points),
    #     'tdoa_error_raw': np.random.randn(n_points) * 0.001 + (np.random.rand(n_points) - 0.5) * 0.0005, # errores pequeños
    #     'doa_error_raw': np.random.randn(n_points) * 10 + (np.random.rand(n_points) * 360 - 180), # errores grandes y variados
    #     'method': np.random.choice(['A', 'B', 'C'], size=n_points)
    # }
    # # Introducir algunos outliers y NaNs
    # dummy_data['tdoa_error_raw'][0:5] = np.random.randn(5) * 0.1 # Outliers grandes
    # dummy_data['doa_error_raw'][5:10] = np.random.randn(5) * 720
    # dummy_data['tdoa_error_raw'][10] = np.nan
    # dummy_df = pd.DataFrame(dummy_data)

    # print("\nGenerando gráficos de ejemplo con datos dummy (si se descomenta el código)...")

    # Ejemplo 1: Boxplot TDOA error vs SNR, con hue por método
    # create_error_plot(
    #     df_full=dummy_df, x_col='snr_db', y_col='tdoa_error_raw',
    #     title='Ejemplo: Error TDOA vs SNR por Método (Boxplot)',
    #     output_filename='dummy_tdoa_err_vs_snr_boxplot.png',
    #     plot_type='boxplot', hue_col='method',
    #     is_angular_error=False,
    #     ylim_strategy='percentile', ylim_params={'low': 1, 'high': 99, 'min_range_abs': 0.0001},
    #     x_label='SNR (dB)', y_label='Error Absoluto TDOA (s)',
    #     palette='viridis'
    # )

    # Ejemplo 2: Violinplot DOA error vs RT60
    # create_error_plot(
    #     df_full=dummy_df, x_col='rt60', y_col='doa_error_raw',
    #     title='Ejemplo: Error DOA vs RT60 (Violinplot)',
    #     output_filename='dummy_doa_err_vs_rt60_violin.png',
    #     plot_type='violinplot',
    #     is_angular_error=True, # Importante para errores DOA
    #     ylim_strategy='percentile', ylim_params={'low': 5, 'high': 95, 'min_range_abs': 10},
    #     x_label='RT60 (s)', y_label='Error Absoluto DOA (grados)',
    #     palette='magma'
    # )

    # Ejemplo 3: Lineplot con CI para TDOA error vs SNR (numérico)
    # create_error_plot(
    #     df_full=dummy_df, x_col='snr_db', y_col='tdoa_error_raw',
    #     title='Ejemplo: Error TDOA vs SNR (Lineplot con CI)',
    #     output_filename='dummy_tdoa_err_vs_snr_lineplot.png',
    #     plot_type='lineplot_ci', hue_col='method',
    #     is_angular_error=False,
    #     ylim_strategy='auto', # Dejar que matplotlib/seaborn decida, o ajustar
    #     x_label='SNR (dB)', y_label='Error Absoluto TDOA (s)',
    #     lineplot_estimator='median', lineplot_ci=95
    # )
    pass
