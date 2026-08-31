from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data_estaciones'
NDVI_PATH = ROOT / 'Etapa 3' / 'ndvi_semanal_estaciones_2021_2025.csv'


def build_weekly_station(df: pd.DataFrame, station: str) -> pd.DataFrame:
    """Agrega los datos horarios a nivel semanal usando la mediana para cada columna."""
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['date']).copy()

    # Semana ISO con inicio lunes, igual que el NDVI semanal
    d['semana'] = d['date'].dt.to_period('W-MON').dt.start_time

    # Mantener columnas numéricas relevantes; excluir la fecha y el año si existen
    exclude = {'date'}
    numeric_cols = [col for col in d.columns if col not in exclude and pd.api.types.is_numeric_dtype(d[col])]

    weekly = (
        d.groupby('semana', as_index=False)[numeric_cols]
        .median(numeric_only=True)
        .sort_values('semana')
        .reset_index(drop=True)
    )

    weekly['estacion'] = station
    weekly['anio'] = weekly['semana'].dt.year
    weekly = weekly[['estacion', 'anio', 'semana'] + [c for c in numeric_cols if c != 'anio']]
    return weekly


def main() -> None:
    ne2 = pd.read_csv(DATA_DIR / 'BD_NE2.csv')
    ne3 = pd.read_csv(DATA_DIR / 'BD_NE3.csv')
    ndvi = pd.read_csv(NDVI_PATH)

    ndvi['semana'] = pd.to_datetime(ndvi['semana'], errors='coerce')
    ndvi = ndvi.dropna(subset=['semana']).copy()

    for station_name, raw_df in [('NE2', ne2), ('NE3', ne3)]:
        weekly = build_weekly_station(raw_df, station_name)

        ndvi_station = ndvi[ndvi['estacion'] == station_name][['semana', 'ndvi_mediana']].copy()
        weekly = weekly.merge(ndvi_station, on='semana', how='left')

        # Archivo con filas nulas preservadas
        out_all = weekly.sort_values('semana').reset_index(drop=True)
        out_all_path = DATA_DIR / f'{station_name}_semanal_mediana_con_ndvi.csv'
        out_all.to_csv(out_all_path, index=False)

        # Archivo eliminando filas con cualquier valor nulo
        out_no_null = out_all.dropna(how='any').reset_index(drop=True)
        out_no_null_path = DATA_DIR / f'{station_name}_semanal_mediana_con_ndvi_sin_nulos.csv'
        out_no_null.to_csv(out_no_null_path, index=False)

        print(f'Generado: {out_all_path.name} ({len(out_all)} filas)')
        print(f'Generado: {out_no_null_path.name} ({len(out_no_null)} filas)')


if __name__ == '__main__':
    main()
