# trafico — flujo vehicular en un tramo con semáforo

Simulación microscópica (por vehículo) de autos, bicicletas, autobuses y cualquier otro tipo de
vehículo definido en la configuración, que recorren un tramo recto de longitud y número de carriles
configurables, con un semáforo al final. Corre solo en CPU
(NumPy, sin CUDA/GPU), reparte réplicas Monte Carlo entre procesos y al final grafica con matplotlib
la **capacidad de movilidad de pasajeros** de cada tipo de vehículo a lo largo del tiempo.

## Instalación y uso (uv)

```bash
uv sync                               # crea .venv con numpy y matplotlib
uv run trafico                        # config.toml de la raíz -> resultados/<ID>_corrida/
uv run trafico regular_traffic        # config.toml de la raíz -> resultados/<ID>_regular_traffic/
uv run trafico resultados/<ID>_regular_traffic   # config.toml de esa carpeta (réplica)
```

El archivo de configuración siempre se llama `config.toml`. El comando acepta un único argumento
opcional:

- **Sin argumento:** se usa el `config.toml` de la raíz del proyecto (junto a `pyproject.toml`),
  sin importar desde qué directorio se ejecute.
- **Una carpeta existente:** se lee su `config.toml`. Puede ser una carpeta de resultados (para
  replicar) o una carpeta de escenario propia, como `escenarios/lluvia/`.
- **Cualquier otro texto:** es el **nombre** de la corrida, el sufijo de su carpeta de resultados,
  y se usa el `config.toml` de la raíz. Solo se conservan letras, dígitos, `.`, `_` y `-`. Un texto
  con `/` que no exista como carpeta es un error, no un nombre.

Todos los parámetros se leen de `config.toml`; su referencia completa está en
[`docs/configuracion.html`](docs/configuracion.html). Las claves que falten toman su valor por
defecto, y una clave desconocida o de tipo incorrecto detiene la corrida con un mensaje claro.

| Sección | Claves (valor por defecto) |
|---|---|
| `[road]` | `length` (200 m; el semáforo está al final), `max_line_speed` (límite en km/h de cada carril; sin límite por defecto) |
| `[traffic_light]` | `red` (30 s), `green` (30 s), `start_phase` (`"red"` \| `"green"`) |
| `[demand]` | `<clave>_rate` por tipo en veh/min, con 0 = no participa (`car_rate` 15, `bike_rate` 4, `bus_rate` 1); `slow_lane` (`"right"` \| `"random"`), carril fijo de los tipos que no cambian de carril y no tienen `lane` |
| `[execution]` | `run` (10 s de proceso), `time_scale` (10), `sample` (1 s), `replicas` (8), `workers` (0 = automático), `seed` |
| `[output]` | `dir` (`"resultados"`), `name` (sufijo de la carpeta si no se da en el comando), `csv`, `show`, `progress` |
| `[vehicles.<clave>]` | `speed_kmh` (velocidad máxima del vehículo), `length`, `gap_run`, `gap_stop`, `pax_min`, `pax_max`, `pax_mean`, `pax_std`; opcionales `name`, `lane_change`, `lane`, `exclusive`, `abreast` y la parada (`stop_position`, `stop_time_mean`, `stop_time_std`) |
| `[behavior]` | tiempos de reacción y de cambio de carril, factor de velocidad en la maniobra, umbrales |

El `config.toml` incluido documenta cada clave y trae todos los valores del modelo explícitos.

### Agregar un tipo de vehículo

Los tipos de vehículo se definen en el archivo de configuración. Cada sección
`[vehicles.<clave>]` es un tipo, y su tasa de llegada va en `[demand]` como `<clave>_rate`:

```toml
[demand]
motorbike_rate = 5.0

[vehicles.motorbike]
name = "moto"          # opcional: etiqueta en gráfica, resumen y CSV (por defecto, la clave)
lane_change = true     # opcional: puede cambiar de carril (por defecto, true)
speed_kmh = 40.0
length = 2.0
gap_run = 2.0
gap_stop = 1.0
pax_min = 1
pax_max = 2
pax_mean = 1.2
pax_std = 0.2
```

- **Tipos nuevos:** deben definir los ocho parámetros numéricos y su tasa. Si falta alguno, la
  corrida se detiene indicando cuál.
- **Tipos incorporados** (`car`, `bike`, `bus`): siempre existen y toman valores por defecto para
  lo que no se indique. Para excluirlos de un escenario, pon su tasa en 0.
- **Carril fijo por tipo:** un tipo con `lane_change = false` puede fijar su carril con `lane`
  (0 = carril derecho, 1 = el siguiente hacia la izquierda, …). Por ejemplo, para separar bicis y
  autobuses:

  ```toml
  [vehicles.bike]
  lane = 0
  exclusive = true   # opcional: carril solo para bicis
  [vehicles.bus]
  lane = 1
  exclusive = true   # opcional: carril solo para autobuses
  ```

  Sin `lane`, se aplica `[demand] slow_lane`. El carril debe existir (menor que el número de carriles).
  Sin `exclusive`, el carril fijo se comparte: autos y motos también pueden entrar o cambiarse a él.
  Con `exclusive = true`, el carril queda reservado para los tipos que lo tienen como `lane`, y la
  gráfica lo marca como "solo bici", "solo autobús", etc.
- **Límite:** se admiten hasta 8 tipos en total, uno por color de la paleta de la gráfica. El color
  de cada tipo depende de su posición: primero los incorporados y después los nuevos, en el orden
  del archivo.

**Carriles.** El número de carriles es la longitud de las listas por carril: `[road]
max_line_speed` y `[behavior] congestion_factor`. Si ambas son listas, deben tener el mismo largo; un
número suelto aplica a todos los carriles, y sin ninguna lista se usan 2 carriles. La antigua clave
`[road] lanes` ya no se usa. Solo se acepta, con un aviso, para poder replicar copias guardadas
antes de este cambio.

**Tiempo.** La resolución es de 1/10 s: cada paso avanza 0.1 s simulados. `run = R` simula
`R × time_scale` segundos; con los valores por defecto, `run = 10` son 100 s simulados (1000 pasos).
El programa corre tan rápido como puede y reporta el tiempo real usado.

## Resultados y replicación

Cada corrida crea su propia carpeta, `<output.dir>/<AAAAMMDD-HHMMSS>_<nombre>/`:

```
resultados/20260924-090500_regular_traffic/
├── config.toml               # copia de la configuración usada
├── movilidad_pasajeros.png   # gráfica (lleva el identificador de la corrida y la semilla)
├── distribucion_pasajeros.png  # pasajeros por vehículo de cada tipo: observada vs. esperada
├── espacio_tiempo_300-420s.png # con [output] animation = true o trafico-ver: trayectorias por carril
├── movimiento_300-420s.webm    # … y video visto desde arriba (WebM; MP4 o GIF con [animation] format)
├── resumen.txt               # el resumen impreso en la terminal
└── series.csv                # series (media y σ); si output.csv = true
```

La copia de la configuración basta para replicar la corrida:

```bash
uv run trafico resultados/20260924-090500_regular_traffic
```

## Ver el movimiento de los vehículos

Para ajustar el modelo sirve ver cómo se mueven los vehículos. `trafico-ver` vuelve a simular una
réplica de una corrida (idéntica, con la semilla de su copia de la configuración) y guarda en la
misma carpeta un diagrama espacio-tiempo por carril y un video visto desde arriba:

```bash
uv run trafico-ver                                    # la corrida más reciente
uv run trafico-ver resultados/<ID>_insurgentes --inicio 300 --duracion 60 --velocidad 2
```

