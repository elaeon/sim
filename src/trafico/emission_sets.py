"""Conjuntos con nombre de coeficientes de emisión del modelo de Int Panis, Broekx y Liu (2006).

Cada conjunto da, por tipo de vehículo, los coeficientes con el formato de `VehicleSpec.emissions`:
(contaminante, [f1..f6] para a ≥ −0.5 m/s², [f1..f6] para a < −0.5 m/s²). Un tipo los elige en
`[vehicles.<clave>.emissions] source = "<conjunto>"` (con `source_type` para tomar la entrada de otro tipo), y
los coeficientes escritos a mano en esa sección reemplazan a los del conjunto, contaminante por contaminante.

- `int_panis_2006`: los originales, para una flota europea de principios de los 2000. Auto a gasolina (CO2, NOx,
  VOC) verificados en arXiv 2411.15238 (tabla 7) y 1912.05956; PM de autobús en arXiv 2008.02405. Su CO2 usa el
  mismo juego al frenar, así que el término a² hace que frenar fuerte emita más que ir a velocidad constante.
- `sedema_cdmx_2018`: recalibrados con la flota de la CDMX medida en la verificación vehicular de SEDEMA
  (jul.–dic. 2018): el CO2 de Int Panis escalado por la masa del tipo, con el juego de frenado igual al ralentí
  (al frenar se corta el combustible), y NOx y VOC como CO2 × la razón contaminante/CO2 medida según la potencia
  específica (VSP), ajustados por mínimos cuadrados en 0–80 km/h y a de −0.5 a 2.5 m/s². Copiados de
  `multi-dashboard/data/sedema_cdmx/emisiones_sim.toml` (generado por `exploracion/sedema_cdmx/factores_sim.py`).
  Limitaciones: es la media de la flota (la cargan los altos emisores); arriba de ≈ 7 kW/t (p. ej. al acelerar
  después de un tope) la razón es extrapolada, constante; el HC por infrarrojo subestima el VOC; sin CO ni PM, y
  el diésel (autobús, carga) no se recalibra porque SEDEMA solo mide su opacidad. Tipos: `car` (particular a
  gasolina, 1250 kg), `taxi` (gasolina, 1150 kg), `carga_ligera` (pickups a gasolina, 1700 kg) y `colectivo`
  (vans y microbuses a gas LP, 2300 kg).
"""

from __future__ import annotations

Coefs = tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...]


def _idle(f1: float) -> tuple[float, ...]:
    """Juego de frenado constante: el valor en ralentí."""
    return (f1, 0.0, 0.0, 0.0, 0.0, 0.0)


_IP_CAR_CO2 = (5.53e-1, 1.61e-1, -2.89e-3, 2.66e-1, 5.11e-1, 1.83e-1)
_IP_BUS_PM = (2.23e-4, 3.47e-4, -2.38e-5, 2.08e-3, 1.76e-3, 2.23e-4)

PETROL_CAR_EMISSIONS: Coefs = (
    ("co2", _IP_CAR_CO2, _IP_CAR_CO2),
    ("nox", (6.19e-4, 8.00e-5, -4.03e-6, -4.13e-4, 3.80e-4, 1.77e-4), _idle(2.17e-4)),
    ("voc", (4.47e-3, 7.32e-7, -2.87e-8, -3.41e-6, 4.94e-6, 1.66e-6), _idle(2.63e-3)),
)  # fmt: skip
BUS_EMISSIONS: Coefs = (("pm", _IP_BUS_PM, _IP_BUS_PM),)


def _sedema(co2: tuple[float, ...], nox: tuple[float, ...], voc: tuple[float, ...]) -> Coefs:
    return tuple((pol, c, _idle(c[0])) for pol, c in (("co2", co2), ("nox", nox), ("voc", voc)))


SEDEMA_CDMX_2018: dict[str, Coefs] = {
    "car": _sedema(
        (0.553, 0.161, -0.00289, 0.266, 0.511, 0.183),
        (0.0001252, 7.058e-05, -2.645e-07, 0.0003921, 0.0001883, 0.0001746),
        (7.956e-05, 2.953e-05, -4.904e-07, 7.725e-05, 8.427e-05, 3.859e-05),
    ),
    "taxi": _sedema(
        (0.5088, 0.1481, -0.002659, 0.2447, 0.4701, 0.1684),
        (0.0001639, 9.375e-05, -6.157e-07, 0.0004608, 0.0002525, 0.0002139),
        (8.125e-05, 3.044e-05, -5.326e-07, 7.163e-05, 8.236e-05, 3.799e-05),
    ),
    "carga_ligera": _sedema(
        (0.7521, 0.219, -0.00393, 0.3618, 0.695, 0.2489),
        (0.0006999, 0.000216, -1.678e-06, 0.0009969, 0.0008622, 0.0003985),
        (0.000363, 3.153e-05, 3.345e-07, 0.0001486, 0.0002701, 7.974e-05),
    ),
    "colectivo": _sedema(
        (1.018, 0.2962, -0.005318, 0.4894, 0.9402, 0.3367),
        (0.003369, 0.00109, -1.938e-05, 0.002236, 0.003218, 0.001281),
        (0.0008306, 0.0002252, -4.088e-06, 0.0002955, 0.0007498, 0.0002491),
    ),
}

EMISSION_SETS: dict[str, dict[str, Coefs]] = {
    "int_panis_2006": {"car": PETROL_CAR_EMISSIONS, "bus": BUS_EMISSIONS},
    "sedema_cdmx_2018": SEDEMA_CDMX_2018,
}

SET_LABELS = {"int_panis_2006": "Int Panis 2006", "sedema_cdmx_2018": "SEDEMA CDMX 2018"}
"""Cómo se nombra cada conjunto en la cabecera y el resumen."""
