# Niveles diarios por percentiles con GARCH

Calcula niveles de precio estadísticos (**P5, P25, P50, P75 y P95**) para la próxima sesión del **Nasdaq 100** a partir de la volatilidad pronosticada por un modelo **GARCH**, y comprueba **fuera de muestra** si esos niveles realmente funcionan.

## Qué hace

- Descarga datos diarios de `^NDX` desde 2015 y usa solo sesiones completas.
- Prueba 12 modelos GARCH (p, q ∈ {1, 2}; distribución normal, t y skew-t) y elige el de menor **AIC**.
- Calcula los niveles con los cuantiles exactos de la distribución del modelo.
- **Validación fuera de muestra (~3 años):**
  - **Llega:** % de sesiones en que el precio tocó el nivel.
  - **Rebota:** % de toques en que el precio cerró de vuelta adentro, con **prueba binomial** contra el 50% del azar (corrección de Bonferroni).
  - **Calibración:** % de cierres por debajo de cada nivel vs. el percentil teórico.
- Régimen de volatilidad, rango típico de la sesión y forecast de las próximas 5 sesiones.
- Gráficas: velas con niveles, validación, volatilidad y forecast.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python percentiles_garch.py
```

## Limitaciones

- Los niveles son probabilidades basadas en la volatilidad histórica, no garantías.
- Usa solo datos diarios (no intradía) y no descuenta festivos.
- Depende de la calidad de los datos de Yahoo Finance.

## Autor

**Santiago Peña Díaz** — Estudiante de Ingeniería de Sistemas (CUN, Colombia)

- GitHub: [Lmao320](https://github.com/Lmao320)
- Documentación completa: [Portafolio en Notion](https://freezing-hope-5b5.notion.site/Portafolio-de-Proyectos-Santiago-Pe-a-D-az-41f7f18e4ca2811793dcece91c3d00c0)
