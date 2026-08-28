# Datos

## Fuente

El enlace histórico del equipo está en [\`FUENTE_DATOS.txt\`](FUENTE_DATOS.txt). El informe también enlaza un dataset ya imputado, pero para entrenar o auditar se necesita la base limpia **antes** de completar los valores faltantes.

El repositorio no descarga automáticamente archivos desde Google Drive porque:

- la carpeta puede requerir permisos;
- los identificadores de archivo pueden cambiar;
- el tamaño y los términos de redistribución no están documentados aquí;
- entrenar sobre un dataset ya imputado invalidaría la auditoría.

## Contrato mínimo del CSV

Cada fila representa una lectura de una estación en un instante. Columnas obligatorias:

| Columna | Tipo esperado | Uso |
|---|---|---|
| \`ID\` | texto o categoría | Identificador de estación |
| \`time\` | fecha/hora interpretable por pandas | Índice temporal |
| 15 variables | numérico o vacío | Entradas multivariadas |

Variables y unidades descritas en el informe:

| Variable | Descripción | Unidad reportada |
|---|---|---|
| CO | Monóxido de carbono | ppm |
| NO | Óxido nítrico | ppb |
| NO2 | Dióxido de nitrógeno | ppb |
| NOX | Óxidos de nitrógeno totales | ppb |
| O3 | Ozono | ppb |
| PM10 | Partículas menores a 10 μm | μg/m³ |
| PM2.5 | Partículas menores a 2.5 μm | μg/m³ |
| PRS | Presión atmosférica | mmHg |
| RAINF | Precipitación | mm |
| RH | Humedad relativa | % |
| SO2 | Dióxido de azufre | ppb |
| SR | Radiación solar | kW/m² |
| TOUT | Temperatura exterior | °C |
| WSR | Velocidad del viento | km/h |
| WDR | Dirección del viento | grados |

El script promedia filas duplicadas de una estación-hora, reindexa a frecuencia horaria y representa ausencias como \`NaN\`. No aplica reglas de limpieza física a valores espurios: esas reglas deben ejecutarse antes o añadirse como un experimento documentado.

## Preparación

1. Obtén autorización y descarga la base limpia original.
2. Guárdala como \`data/df_simanew_cleaned.csv\`.
3. Valida el contrato:

\`\`\`bash
uv run python sima.py inspect --input data/df_simanew_cleaned.csv
\`\`\`

4. Revisa fechas, estaciones y porcentajes faltantes antes de entrenar.

Los archivos \`data/*.csv\` están ignorados por Git para evitar publicar datos grandes o restringidos por accidente.
