# Jeidy Super Saltarín

Juego de plataformas con dos ediciones independientes: **2D (Pygame)** y **3D (Ursina)**. Conserva los modos para uno o dos jugadores, rival controlado por IA y exportación de telemetría CSV.

## Qué incluye

- Tres niveles que alternan los fondos de islas, noche y parque, con plataformas distintas en cada nivel.
- Personajes Jeidy, Amiga y PC a partir de las imágenes adjuntas, convertidas a PNG con transparencia.
- Ambiente musical en bucle y efectos diferentes para saltar, caminar, deslizarse, aterrizar, recoger objetos y recibir golpes.
- Botones en pantalla para iniciar, finalizar el nivel y continuar al siguiente.
- Archivos de lanzamiento de VS Code en `.vscode/launch.json`.

Los sonidos se generan y guardan como WAV dentro de `assets/audio`; las imágenes se guardan en `assets/`. No hace falta descargar recursos adicionales.

## Requisitos

Python 3.10–3.12 recomendado. La edición 3D requiere una computadora compatible con Ursina/Panda3D. La instalación de TensorFlow puede tardar y requiere una versión compatible de Python.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecutar

Desde la carpeta principal del juego, con el entorno virtual activo:

```bash
python version_2d/game_2d.py
```

o:

```bash
python version_3d/game_3d.py
```

En VS Code, abre esta carpeta, selecciona el intérprete de `.venv` y ejecuta **Jeidy Super Saltarín 2D** o **Jeidy Super Saltarín 3D** desde el panel Run and Debug.

## Controles

- Jugador 1: flechas; arriba salta, abajo se agacha y abajo + izquierda/derecha se desliza.
- Jugador 2: WASD; W salta, S se agacha y S + A/D se desliza.
- En 3D, `,` y `.` mueven al jugador 1 en profundidad; `Q` y `E`, al jugador 2.
- En el menú, `1`, `2` o `3` inician el modo indicado. `ESC` cierra el juego.
- Usa los botones **INICIAR JUEGO**, **FINALIZAR NIVEL** y **SIGUIENTE NIVEL** para jugar con el mouse.

Al terminar, la telemetría se guarda en `version_2d/telemetria/` o `version_3d/telemetria/`.

## Subir una copia propia a GitHub

1. Crea un repositorio vacío en GitHub y copia su URL.
2. Desde esta carpeta, ejecuta:

```bash
git init
git add .
git commit -m "Mejora Jeidy Super Saltarín: niveles, arte y sonidos"
git branch -M main
git remote add origin URL_DE_TU_REPOSITORIO
git push -u origin main
```

No subas `.venv/` ni archivos de telemetría generados; ya están excluidos en `.gitignore`.
