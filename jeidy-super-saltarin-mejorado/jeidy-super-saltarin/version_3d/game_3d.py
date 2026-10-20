# -*- coding: utf-8 -*-
"""
=============================================================================
 JEIDY SUPER SALTARÍN  -  VERSIÓN 3D (Ursina Engine)
=============================================================================
Mismas mecánicas que la versión 2D, trasladadas a un escenario tridimensional.

Modos (menú, teclas 1, 2 o 3): PC vs USUARIO | PC vs USUARIO vs USUARIO | USUARIO vs USUARIO

Controles
    Jugador 1 (flechas): <- -> caminar | ARRIBA súper salto | ABAJO agacharse
                         ABAJO + <- / -> deslizarse acostado
                         ',' y '.'  -> moverse en profundidad (eje Z, solo en 3D)
    Jugador 2 (WASD)   : A D caminar | W súper salto | S agacharse | S + A/D deslizarse
                         Q y E      -> moverse en profundidad (eje Z, solo en 3D)
    Recoger objetos    : automático al tocarlos (monedas y hongo)
    ESC                : salir (se exporta la telemetría con pandas)
=============================================================================
"""
from ursina import *   # Entity, Text, Ursina, camera, held_keys, color, destroy, Sky, ...

import math
import os
import random
import threading
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# =============================================================================
#  CEREBRO IA DEL AGENTE "PC"  (scikit-learn + Keras/TensorFlow)
# =============================================================================
class AIBrain:
    """
    Agente predictor ligero.
      * Árbol de decisión (scikit-learn): decide el MOVIMIENTO horizontal
        (0 = quieto, 1 = izquierda, 2 = derecha).
      * Red neuronal pequeña (Keras): decide si SALTAR (probabilidad 0..1).
      * Imitación: guarda muestras de cuándo saltan los jugadores humanos y
        reentrena la red en segundo plano para imitar sus saltos.

    Vector de características (4 valores):
        [dx, dy, en_suelo, peligro_adelante]
        dx, dy  -> distancia normalizada al objetivo (dy < 0 = objetivo ARRIBA)
        en_suelo -> 1.0 si la IA toca el suelo
        peligro_adelante -> 1.0 si hay un enemigo cerca en la dirección de marcha
    """

    def __init__(self):
        self.samples = []          # muestras humanas (features, saltó?)
        self.training = False
        self.model = None
        self.w = None              # pesos de la red (para inferencia rápida con numpy)
        self.Xs, self.ms, self.js = self._synthetic()
        self.tree = DecisionTreeClassifier(max_depth=6, random_state=0)
        self.tree.fit(self.Xs, self.ms)
        self._load_keras()

    # ---------- datos sintéticos de entrenamiento inicial ----------
    @staticmethod
    def _synthetic(n=4000):
        rng = np.random.default_rng(0)
        dx = rng.uniform(-1, 1, n)
        dy = rng.uniform(-1, 1, n)
        ground = rng.integers(0, 2, n).astype(float)
        haz = rng.integers(0, 2, n).astype(float)
        X = np.c_[dx, dy, ground, haz]
        move = np.where(dx > 0.012, 2, np.where(dx < -0.012, 1, 0))
        jump = (((dy < -0.06) & (ground == 1)) | ((haz == 1) & (ground == 1))).astype(int)
        return X, move, jump

    # ---------- red neuronal (Keras) ----------
    def _load_keras(self):
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            try:
                from tensorflow import keras
            except ImportError:
                import keras
            m = keras.Sequential([
                keras.layers.Input(shape=(4,)),
                keras.layers.Dense(12, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ])
            m.compile(optimizer="adam", loss="binary_crossentropy")
            m.fit(self.Xs, self.js, epochs=25, batch_size=64, verbose=0)
            self.model = m
            self._sync_weights()
            print("[IA] Árbol de decisión + red Keras listos.")
        except Exception as exc:  # el juego sigue funcionando sin Keras
            print("[IA] Keras/TensorFlow no disponible, uso regla de respaldo:", exc)

    def _sync_weights(self):
        self.w = tuple(np.asarray(a) for a in self.model.get_weights())

    # ---------- imitación de saltos humanos ----------
    def add_sample(self, features, jumped):
        self.samples.append((list(features), int(jumped)))
        if len(self.samples) > 2000:
            del self.samples[:200]

    def retrain_async(self):
        if self.model is None or self.training or len(self.samples) < 150:
            return
        self.training = True
        threading.Thread(target=self._retrain, daemon=True).start()

    def _retrain(self):
        try:
            Xh = np.array([s[0] for s in self.samples])
            yh = np.array([s[1] for s in self.samples])
            idx = np.random.default_rng().choice(len(self.Xs), size=1500, replace=False)
            X = np.vstack([self.Xs[idx], Xh])
            y = np.concatenate([self.js[idx], yh])
            self.model.fit(X, y, epochs=3, batch_size=64, verbose=0)
            self._sync_weights()
        except Exception as exc:
            print("[IA] Error reentrenando:", exc)
        finally:
            self.training = False

    # ---------- inferencia ----------
    def _jump_prob(self, f):
        if self.w is None:  # regla de respaldo
            return 1.0 if (f[2] > 0.5 and (f[1] < -0.06 or f[3] > 0.5)) else 0.0
        W1, b1, W2, b2 = self.w
        h = np.maximum(0.0, np.asarray(f, dtype=float) @ W1 + b1)
        z = float((h @ W2 + b2).ravel()[0])
        return 1.0 / (1.0 + math.exp(-z))

    def decide(self, f):
        """Devuelve (izquierda, derecha, salto, agacharse)."""
        move = int(self.tree.predict(np.array([f]))[0])
        jump = self._jump_prob(f) > 0.5
        return (move == 1, move == 2, jump, False)


# =============================================================================
#  CONSTANTES
# =============================================================================
WX, WY = 40.0, 12.0           # escalas de normalización para la IA (ancho y alto del mundo)
GRAV = -32.0
JUMP = 17.0                   # súper salto (≈4.5 unidades)
WALK, SLIDE, ZSPD = 7.0, 11.0, 5.0
MAXFALL = -30.0
ROUND_SECONDS = 90

# (x, y_superior, z, medio_ancho, medio_fondo) -> el primero es el suelo
PLATS = [(0, 0, 0, 22, 8),
         (-12, 3, 0, 3.5, 2.5), (0, 3, 2, 3.5, 2.5), (12, 3, -1, 3.5, 2.5),
         (-6, 6, 0, 3.5, 2.5), (6, 6, 0, 3.5, 2.5),
         (-14, 9, 2, 3, 2.5), (0, 9, 0, 3.5, 2.5), (14, 9, -2, 3, 2.5)]
LEVEL_PLATFORMS = [
    PLATS,
    [(0, 0, 0, 22, 8), (-13, 3, -1, 3.2, 2.5), (0, 3, 2, 3.5, 2.5),
     (13, 3, 0, 3.2, 2.5), (-7, 6, 1, 3.5, 2.5), (6, 6, -2, 3.5, 2.5),
     (-14, 9, 2, 3, 2.5), (0, 9, -1, 3.5, 2.5), (14, 9, 1, 3, 2.5)],
    [(0, 0, 0, 22, 8), (-12, 3, 1, 3.2, 2.5), (1, 3, -2, 3.5, 2.5),
     (13, 3, 0, 3.2, 2.5), (-7, 6, -1, 3.5, 2.5), (6, 6, 2, 3.5, 2.5),
     (-14, 9, -2, 3, 2.5), (0, 9, 1, 3.5, 2.5), (14, 9, -1, 3, 2.5)],
]
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIR = os.path.join(PROJECT_DIR, "assets")
BACKGROUND_FILES = [os.path.join(ASSET_DIR, "backgrounds", f"nivel_{i}.jpg") for i in range(1, 4)]
MENU_BACKGROUND_FILES = [
    os.path.join(ASSET_DIR, "backgrounds", f"menu_3d_{i}.png") for i in range(1, 4)
]
CHARACTER_FILES = {
    "p1": os.path.join(ASSET_DIR, "characters", "jeidy.png"),
    "p2": os.path.join(ASSET_DIR, "characters", "amiga.png"),
    "ai": os.path.join(ASSET_DIR, "characters", "pc.png"),
}

MODES = {
    1: ("PC vs USUARIO", ["p1", "ai"]),
    2: ("PC vs USUARIO vs USUARIO", ["p1", "p2", "ai"]),
    3: ("USUARIO vs USUARIO", ["p1", "p2"]),
}
# nombre, color, (izq, der, salto, abajo, z+, z-)
ROLE_INFO = {
    "p1": ("Jeidy (J1)", color.pink, ("left arrow", "right arrow", "up arrow", "down arrow", ".", ",")),
    "p2": ("Amiga (J2)", color.azure, ("a", "d", "w", "s", "e", "q")),
    "ai": ("PC-Jeidy", color.lime, None),
}


class SoundManager:
    """Reproduce ambiente y efectos desde WAV locales usando el audio de Ursina."""

    def __init__(self):
        self.effect_voices = {}
        self.voice_index = {}
        self.ambience = None
        sound_dir = os.path.join(ASSET_DIR, "audio")
        settings = {
            "salto": (.42, 1.08, .12),
            "pasos": (.18, 1.0, .04),
            "deslizar": (.3, .92, .08),
            "caida": (.42, .88, .08),
            "moneda": (.48, 1.18, .12),
            "premio": (.52, 1.06, .1),
            "golpe": (.48, .9, .1),
            "pisoton": (.42, .82, .06),
            "final": (.5, 1.0, .04),
        }
        ambience_path = os.path.join(sound_dir, "ambiente.wav")
        if os.path.isfile(ambience_path):
            try:
                self.ambience = Audio(ambience_path, autoplay=True, loop=True,
                                      volume=.24, pitch=1.0, group="music")
            except Exception as exc:
                print("[Audio] No se pudo iniciar la música ambiental:", exc)
        else:
            print(f"[Audio] No se encontró la música ambiental: {ambience_path}")

        for name, (volume, pitch, variation) in settings.items():
            path = os.path.join(sound_dir, f"{name}.wav")
            if not os.path.isfile(path):
                print(f"[Audio] No se encontró el efecto '{name}': {path}")
                continue
            voices = []
            for _ in range(3):
                try:
                    voices.append(
                        Audio(path, autoplay=False, loop=False, volume=volume,
                              pitch=pitch, group="sfx")
                    )
                except Exception as exc:
                    print(f"[Audio] No se pudo cargar el efecto '{name}': {exc}")
                    break
            if voices:
                self.effect_voices[name] = (voices, pitch, variation)
                self.voice_index[name] = 0

    def set_level(self, level):
        if self.ambience:
            self.ambience.pitch = (0.96, 1.0, 1.04)[(level - 1) % 3]

    def play(self, name):
        voice_set = self.effect_voices.get(name)
        if voice_set:
            voices, pitch, variation = voice_set
            index = self.voice_index[name]
            sound = voices[index]
            self.voice_index[name] = (index + 1) % len(voices)
            try:
                sound.stop(destroy=False)
                sound.pitch = pitch + random.uniform(-variation, variation)
                sound.play()
            except Exception as exc:
                print(f"[Audio] No se pudo reproducir '{name}': {exc}")


# =============================================================================
#  PERSONAJE 3D
# =============================================================================
class Jeidy3D:
    def __init__(self, name, col, spawn, kind, keys=None, sprite_path=None):
        self.name, self.col, self.spawn, self.kind, self.keys = name, col, spawn, kind, keys
        self.root = Entity()
        self.sprite_is_image = bool(sprite_path and os.path.isfile(sprite_path))
        if self.sprite_is_image:
            self.sprite = Entity(parent=self.root, model="quad", texture=load_texture(sprite_path),
                                 double_sided=True, scale=(.85, 1.8), position=(0, .9, -.45),
                                 color=color.white, collider=None, transparency=True, unlit=True)
        else:
            self.sprite = Entity(parent=self.root, model="cube", color=col, scale=(.8, 1.6, .5), y=.8)
        self.reset()

    def reset(self):
        self.x, self.y, self.z = self.spawn          # y = pies
        self.vx = self.vy = self.vz = 0.0
        self.prev_y = self.y
        self.on_ground = False
        self.big = False
        self.alive = True
        self.lives = 3
        self.invuln = 0.0
        self.state = "stand"                         # stand | crouch | slide
        self.facing = 1
        self.score = self.coins = self.jumps = self.stomps = 0
        self.jumped_now = False
        self.root.enabled = True
        self.apply_visual()

    # dimensiones según estado y tamaño (Hongo = escala x2)
    def dims(self):
        m = 2 if self.big else 1
        if self.state == "crouch":
            return m, 1.0, 0.6
        if self.state == "slide":
            return m, 1.4, 0.35
        return m, 1.0, 1.0

    @property
    def hx(self):
        m, sx, _ = self.dims()
        return .4 * sx * m

    @property
    def hz(self):
        return .3 * self.dims()[0]

    @property
    def h(self):
        m, _, sy = self.dims()
        return 1.75 * sy * m

    @property
    def cy(self):
        return self.y + self.h / 2

    def apply_visual(self):
        m, sx, sy = self.dims()
        self.root.scale = Vec3(sx * m, sy * m, m)     # escala del sprite 3D
        self.root.position = Vec3(self.x, self.y, self.z)
        self.root.rotation_y = 0
        if hasattr(self, "sprite"):
            self.sprite.rotation_z = -90 if self.state == "slide" else 0
            self.sprite.y = .42 if self.state == "slide" else (.62 if self.state == "crouch" else .9)
            if self.sprite_is_image:
                self.sprite.scale_x = (-.85 if self.facing < 0 else .85) * (1.4 if self.state == "slide" else 1)

    def grow(self):
        self.big = True

    def hit(self):
        if self.invuln > 0 or not self.alive:
            return
        if self.big:
            self.big = False
            self.invuln = 1.5
        else:
            self.lives -= 1
            self.invuln = 2.0
            if self.lives <= 0:
                self.alive = False
                self.root.enabled = False
            else:
                self.x, self.y, self.z = self.spawn
                self.vx = self.vy = self.vz = 0.0

    def update(self, inp, dt):
        left, right, up, down, zf, zb = inp
        self.prev_y = self.y
        self.jumped_now = False
        self.invuln = max(0.0, self.invuln - dt)
        d = (1 if right else 0) - (1 if left else 0)
        zd = (1 if zf else 0) - (1 if zb else 0)
        if d:
            self.facing = d

        if self.on_ground and down and (d or zd):
            self.state = "slide"
        elif self.on_ground and down:
            self.state = "crouch"
        else:
            self.state = "stand"

        if self.state == "slide":
            self.vx, self.vz = SLIDE * d, SLIDE * .6 * zd
        elif self.state == "crouch":
            self.vx = self.vz = 0.0
        else:
            self.vx, self.vz = WALK * d, ZSPD * zd

        if up and self.on_ground and self.state == "stand":        # SÚPER SALTO
            self.vy = JUMP
            self.on_ground = False
            self.jumps += 1
            self.jumped_now = True
        if down and not self.on_ground:
            self.vy += GRAV * .6 * dt                                # caída rápida

        self.vy = max(self.vy + GRAV * dt, MAXFALL)
        self.x = max(-21.0, min(21.0, self.x + self.vx * dt))
        self.z = max(-7.5, min(7.5, self.z + self.vz * dt))
        self.y += self.vy * dt

        self.on_ground = False
        if self.vy <= 0:
            for (px, top, pz, hw, hd) in PLATS:
                if (abs(self.x - px) < hw + self.hx and abs(self.z - pz) < hd + self.hz
                        and self.prev_y >= top - .05 and self.y <= top):
                    self.y, self.vy, self.on_ground = float(top), 0.0, True
                    break
        self.apply_visual()
        # parpadeo al ser golpeado
        self.root.visible = not (self.invuln > 0 and int(self.invuln * 10) % 2)

    def touches(self, cx, cy, cz, r=.5):
        return (abs(self.x - cx) < self.hx + r and abs(self.z - cz) < self.hz + r + .3
                and self.y - .2 < cy + r and self.y + self.h + .2 > cy - r)


class Enemigo3D:
    def __init__(self, minx, maxx, top, z, speed=2.2):
        self.minx, self.maxx, self.top, self.z, self.speed = minx, maxx, top, z, speed
        self.x = random.uniform(minx, maxx)
        self.dir = random.choice((-1, 1))
        self.alive, self.timer = True, 0.0
        self.e = Entity(model="sphere", color=color.rgb(.95, .27, .35),
                        scale=(1, .76, .82),
                        position=(self.x, top + .4, z))
        for eye_x in (-.22, .22):
            Entity(parent=self.e, model="sphere", color=color.white, scale=.24,
                   position=(eye_x, .18, -.39))
            Entity(parent=self.e, model="sphere", color=color.rgb(.12, .08, .18), scale=.11,
                   position=(eye_x, .18, -.59))
        for foot_x in (-.3, 0, .3):
            Entity(parent=self.e, model="sphere", color=color.rgb(1, .68, .28),
                   scale=(.22, .17, .24), position=(foot_x, -.34, -.12))

    def update(self, dt):
        if not self.alive:
            self.timer -= dt
            if self.timer <= 0:
                self.alive = True
                self.e.enabled = True
            return
        self.x += self.dir * self.speed * dt
        if self.x < self.minx or self.x > self.maxx:
            self.dir *= -1
        self.e.x = self.x

    def kill(self):
        self.alive, self.timer = False, 5.0
        self.e.enabled = False


# =============================================================================
#  JUEGO
# =============================================================================
class Juego3D:
    def __init__(self):
        print("[IA] Cargando scikit-learn + Keras (puede tardar unos segundos)...")
        self.brain = AIBrain()
        Sky()
        camera.position = (0, 7, -32)
        camera.rotation_x = 4
        camera.fov = 70
        self.menu_backdrop = Entity(
            parent=camera.ui, model="quad", texture=load_texture(MENU_BACKGROUND_FILES[0]),
            position=(0, 0, 1), scale=(camera.aspect_ratio, 1),
            collider=None, unlit=True)
        self.state = "menu"
        self.world, self.players, self.enemies, self.coins = [], [], [], []
        self.mushroom = None
        self.hud = []
        self.player_hud = []
        self.ai_state = {}
        self.telemetry = []
        self.round_no = 0
        self.mode = 1
        self.level = 1
        self.audio = SoundManager()
        self.walk_sound_timer = 0.0
        self.menu_title = Text(
            text="JEIDY SUPER SALTARÍN",
            origin=(0, 0), position=(0, .31), scale=2.05, color=color.white)
        self.menu_text = Text(
            text="AVENTURA 3D  |  NIVEL 1 DE 3\n\n"
                 "1   PC VS JEIDY\n"
                 "2   EQUIPO DE TRES\n"
                 "3   DUELO DE JUGADORES",
            origin=(0, 0), position=(0, .08), scale=1.05, color=color.white)
        self.menu_guide = Text(
            text="J1  Flechas: mover  |  Arriba: saltar  |  , .: profundidad\n"
                 "J2  A/D: mover  |  W: saltar  |  Q/E: profundidad",
            origin=(0, 0), position=(0, -.17), scale=.72, color=color.rgb(.85, .91, 1))
        self.over_text = None
        self.start_button = Button(text="INICIAR AVENTURA", color=color.azure, scale=(.32, .075),
                                   position=(-.19, -.34), on_click=self.start_from_button)
        self.quit_button = Button(text="SALIR", color=color.red, scale=(.2, .075),
                                  position=(.2, -.34), on_click=application.quit)
        self.finish_button = Button(text="FINALIZAR NIVEL", color=color.red, scale=(.24, .065),
                                    position=(.74, .45), on_click=self.finish, enabled=False)
        self.next_button = Button(text="SIGUIENTE NIVEL", color=color.green, scale=(.28, .075),
                                  position=(-.32, -.36), on_click=self.next_level, enabled=False)
        self.menu_button = Button(text="MENÚ", color=color.azure, scale=(.2, .075),
                                  position=(0, -.36), on_click=self.return_to_menu, enabled=False)
        self.over_quit_button = Button(text="FINALIZAR", color=color.red, scale=(.24, .075),
                                       position=(.29, -.36), on_click=application.quit, enabled=False)

    def start_from_button(self):
        self.start_round(self.mode)

    def next_level(self):
        self.level = self.level % 3 + 1
        self.start_round(self.mode)

    def return_to_menu(self):
        self.clear()
        self.state = "menu"
        self.menu_text.text = (
            f"AVENTURA 3D  |  NIVEL {self.level} DE 3\n\n"
            "1   PC VS JEIDY\n"
            "2   EQUIPO DE TRES\n"
            "3   DUELO DE JUGADORES"
        )
        self.menu_text.enabled = True
        self.menu_title.enabled = True
        self.menu_guide.enabled = True
        self.menu_backdrop.texture = load_texture(MENU_BACKGROUND_FILES[(self.level - 1) % 3])
        self.menu_backdrop.enabled = True
        self.audio.play("moneda")
        self.start_button.enabled = True
        self.quit_button.enabled = True
        self.finish_button.enabled = False
        self.next_button.enabled = False
        self.menu_button.enabled = False
        self.over_quit_button.enabled = False

    # ---------- construcción del escenario ----------
    def clear(self):
        for e in self.world + [p.root for p in self.players] + [e.e for e in self.enemies] + self.coins + self.hud:
            destroy(e)
        if self.mushroom:
            destroy(self.mushroom["e"])
        if self.over_text:
            destroy(self.over_text)
            self.over_text = None
        self.world, self.players, self.enemies, self.coins, self.hud = [], [], [], [], []
        self.player_hud = []
        self.mushroom = None

    def start_round(self, mode):
        global PLATS
        self.clear()
        self.menu_title.enabled = False
        self.menu_text.enabled = False
        self.menu_guide.enabled = False
        self.menu_backdrop.enabled = False
        self.start_button.enabled = False
        self.quit_button.enabled = False
        self.finish_button.enabled = True
        self.next_button.enabled = False
        self.menu_button.enabled = False
        self.over_quit_button.enabled = False
        self.mode, self.state = mode, "play"
        self.round_no += 1
        self.time_left, self.tel_t = float(ROUND_SECONDS), 0.0
        self.ai_state = {}
        self.walk_sound_timer = 0.0
        PLATS = LEVEL_PLATFORMS[(self.level - 1) % len(LEVEL_PLATFORMS)]
        background_path = BACKGROUND_FILES[(self.level - 1) % len(BACKGROUND_FILES)]
        if os.path.isfile(background_path):
            backdrop = Entity(model="quad", texture=load_texture(background_path),
                              position=(0, 6, 18), scale=(130, 80), double_sided=True,
                              collider=None, unlit=True)
            self.world.append(backdrop)
        self.audio.set_level(self.level)
        platform_colors = (
            (color.rgb(.38, .91, .44), color.rgb(.47, .28, .19), color.rgb(1, .75, .28)),
            (color.rgb(.75, .42, 1), color.rgb(.28, .14, .44), color.rgb(1, .8, .3)),
            (color.rgb(.24, .88, .77), color.rgb(.17, .42, .34), color.rgb(1, .62, .25)),
        )
        top_color, base_color, edge_color = platform_colors[(self.level - 1) % 3]
        for i, (x, top, z, hw, hd) in enumerate(PLATS):
            thickness = .75 if i == 0 else .42
            self.world.append(Entity(
                model="cube", texture="white_cube", color=base_color,
                position=(x, top - thickness / 2, z),
                scale=(hw * 2, thickness, hd * 2)))
            self.world.append(Entity(
                model="cube", texture="white_cube", color=top_color,
                position=(x, top - .035, z),
                scale=(hw * 2 + .12, .16, hd * 2 + .12)))
            self.world.append(Entity(
                model="cube", texture="white_cube", color=edge_color,
                position=(x, top - .25, z - hd),
                scale=(hw * 2, .12, .12)))
        roles = MODES[mode][1]
        xs = [-16, 16] if len(roles) == 2 else [-16, 0, 16]
        for x, role in zip(xs, roles):
            name, col, keys = ROLE_INFO[role]
            self.players.append(Jeidy3D(name, col, (x, 0.0, -3.0 if role != "ai" else 3.0),
                                        "ai" if role == "ai" else "human", keys, CHARACTER_FILES[role]))
        self.enemies = []
        for platform_index in (1, 3, 5, 7):
            px, top, pz, hw, _ = PLATS[platform_index]
            self.enemies.append(Enemigo3D(px - hw + .8, px + hw - .8, top, pz))
        for _ in range(8):
            self.spawn_coin()
        player_panel = Entity(parent=camera.ui, model="quad",
                              position=(-.58, .37, .1), scale=(.78, .27),
                              color=color.rgba(.05, .08, .19, .74))
        self.hud.append(player_panel)
        for i, p in enumerate(self.players):
            player_text = Text(text="", position=(-.96, .46 - i * .065),
                               origin=(-.5, .5), scale=.72, color=color.white)
            self.player_hud.append(player_text)
            self.hud.append(player_text)
        header_panel = Entity(parent=camera.ui, model="quad",
                              position=(.27, .43, .1), scale=(.68, .19),
                              color=color.rgba(.05, .08, .19, .7))
        self.hud.append(header_panel)
        self.timer_panel = Entity(parent=camera.ui, model="quad",
                                  position=(.83, .45, .1), scale=(.25, .15),
                                  color=color.rgba(.05, .08, .19, .74))
        self.hud.append(self.timer_panel)
        self.timer_text = Text(text="", position=(.83, .46), origin=(0, .5),
                               scale=1.35, color=color.white)
        self.hud.append(self.timer_text)
        self.hud.append(Text(text=MODES[mode][0], position=(.27, .46), origin=(0, .5),
                             scale=.68, color=color.white))
        self.hud.append(Text(text=f"NIVEL {self.level}/3", position=(.27, .405), origin=(0, .5),
                             scale=.7, color=color.rgb(1, .82, .39)))
        self.audio.play("premio")

    def spawn_coin(self):
        x, top, z, hw, hd = random.choice(PLATS)
        c = Entity(position=(
            x + random.uniform(-hw + .5, hw - .5),
            top + .9,
            z + random.uniform(-hd + .5, hd - .5)))
        Entity(parent=c, model="cylinder", color=color.rgb(1, .62, .12),
               scale=(.58, .1, .58), rotation_x=90)
        Entity(parent=c, model="cylinder", color=color.rgb(1, .88, .35),
               scale=(.39, .115, .39), rotation_x=90)
        Entity(parent=c, model="sphere", color=color.white.tint(.4),
               scale=(.1, .1, .1), position=(-.08, .1, -.08))
        self.coins.append(c)

    def spawn_mushroom(self):
        x, top, z, hw, hd = random.choice(PLATS[1:])
        e = Entity(position=(x, top + .6, z))
        Entity(parent=e, model="cylinder", color=color.white.tint(-.1), scale=(.3, .5, .3), y=-.2)
        Entity(parent=e, model="sphere", color=color.red, scale=(.9, .6, .9), y=.15)
        Entity(parent=e, model="sphere", color=color.white, scale=.2, position=(-.2, .3, -.3))
        Entity(parent=e, model="sphere", color=color.white, scale=.2, position=(.2, .3, -.3))
        self.mushroom = {"e": e, "pos": Vec3(x, top + .6, z)}

    # ---------- IA ----------
    def nearest_collectible(self, p):
        best, bd = None, 1e9
        for c in self.coins:
            d = math.dist((c.x, c.y, c.z), (p.x, p.cy, p.z))
            if d < bd:
                best, bd = (c.x, c.y, c.z), d
        if self.mushroom and not p.big:
            m = self.mushroom["pos"]
            d = math.dist((m.x, m.y, m.z), (p.x, p.cy, p.z)) * .5
            if d < bd:
                best = (m.x, m.y, m.z)
        return best

    def features(self, p, tx, ty):
        # dy se invierte para respetar la convención del modelo (negativo = objetivo ARRIBA)
        dx, dy = (tx - p.x) / WX, (p.cy - ty) / WY
        dirn = 1 if dx >= 0 else -1
        haz = 0.0
        for e in self.enemies:
            if e.alive and abs(e.top + .4 - p.cy) < 1.6 and abs(e.z - p.z) < 1.2 and 0 < (e.x - p.x) * dirn < 3:
                haz = 1.0
                break
        return [dx, dy, 1.0 if p.on_ground else 0.0, haz]

    def ai_target(self, p):
        humans = [q for q in self.players if q.kind == "human" and q.alive]
        if humans:
            h = min(humans, key=lambda q: math.dist((q.x, q.y, q.z), (p.x, p.y, p.z)))
            if math.dist((h.x, h.y, h.z), (p.x, p.y, p.z)) < 9:      # interceptar al jugador
                return h.x, h.y + h.h + 1.8, h.z
        t = self.nearest_collectible(p)
        return t if t else (0.0, p.cy, 0.0)

    def ai_input(self, p, dt):
        st = self.ai_state.setdefault(p, {"t": 0.0, "inp": (False,) * 6})
        st["t"] -= dt
        if st["t"] <= 0:
            st["t"] = .1                                       # "piensa" cada 0.1 s
            tx, ty, tz = self.ai_target(p)
            l, r, u, d = self.brain.decide(self.features(p, tx, ty))   # X y salto: árbol + red
            dz = tz - p.z                                      # profundidad: regla proporcional simple
            st["inp"] = (l, r, u, d, dz > .5, dz < -.5)
        return st["inp"]

    # ---------- bucle ----------
    def on_key(self, key):
        if key == "escape":
            application.quit()
        elif self.state == "menu" and key in ("1", "2", "3"):
            self.start_round(int(key))
        elif self.state == "over" and key == "r":
            self.return_to_menu()
        elif self.state == "over" and key == "n":
            self.next_level()

    def tick(self, dt):
        if self.state != "play":
            return
        self.time_left -= dt
        self.walk_sound_timer = max(0.0, self.walk_sound_timer - dt)
        for c in self.coins:
            c.rotation_y += 140 * dt

        for p in self.players:
            if not p.alive:
                continue
            was_ground, old_state = p.on_ground, p.state
            if p.kind == "human":
                inp = tuple(bool(held_keys[k]) for k in p.keys)
                tgt = self.nearest_collectible(p)
                f = self.features(p, tgt[0], tgt[1]) if tgt else None
                p.update(inp, dt)
                if f and was_ground and random.random() < .15:     # muestras para imitar saltos
                    self.brain.add_sample(f, p.jumped_now)
            else:
                inp = self.ai_input(p, dt)
                p.update(inp, dt)
            if p.jumped_now:
                self.audio.play("salto")
            elif not was_ground and p.on_ground:
                self.audio.play("caida")
            if p.state == "slide" and old_state != "slide":
                self.audio.play("deslizar")
            elif p.on_ground and (inp[0] or inp[1] or inp[4] or inp[5]) and self.walk_sound_timer <= 0:
                self.audio.play("pasos")
                self.walk_sound_timer = .42
        if int(self.time_left) % 20 == 0 and int(self.time_left) != getattr(self, "_last_rt", -1):
            self._last_rt = int(self.time_left)
            self.brain.retrain_async()

        for e in self.enemies:
            e.update(dt)
        self.collisions()

        if random.random() < dt / 2 and len(self.coins) < 12:
            self.spawn_coin()
        if self.mushroom is None and random.random() < dt / 10:
            self.spawn_mushroom()

        self.tel_t += dt
        if self.tel_t >= .25:                                    # telemetría (pandas)
            self.tel_t = 0.0
            for p in self.players:
                self.telemetry.append({
                    "ronda": self.round_no, "modo": MODES[self.mode][0], "t": round(ROUND_SECONDS - self.time_left, 2),
                    "jugador": p.name, "x": round(p.x, 2), "y": round(p.y, 2), "z": round(p.z, 2),
                    "saltos": p.jumps, "monedas": p.coins, "puntos": p.score, "vidas": p.lives,
                    "grande": int(p.big), "estado": p.state})

        for p, t in zip(self.players, self.player_hud):
            t.text = f"{p.name}  Pts:{p.score}  Mon:{p.coins}  Vidas:{max(p.lives, 0)}" + ("  [GRANDE]" if p.big else "")
        self.timer_text.text = f"{max(0, int(self.time_left))}s"

        if self.time_left <= 0 or sum(p.alive for p in self.players) <= 1:
            self.finish()

    def collisions(self):
        alive = [p for p in self.players if p.alive]
        for p in alive:
            for c in self.coins[:]:
                if p.touches(c.x, c.y, c.z):
                    self.coins.remove(c)
                    destroy(c)
                    p.coins += 1
                    p.score += 1
                    self.audio.play("moneda")
            if self.mushroom and p.touches(*self.mushroom["pos"], r=.6):
                destroy(self.mushroom["e"])
                self.mushroom = None
                p.grow()
                p.score += 10
                self.audio.play("premio")
            for e in self.enemies:
                if not e.alive or not p.touches(e.x, e.top + .4, e.z, r=.55):
                    continue
                if p.vy <= 0 and p.prev_y >= e.top + .5:
                    e.kill()
                    p.vy, p.score = 9.0, p.score + 3
                    p.stomps += 1
                    self.audio.play("pisoton")
                elif p.state == "slide":
                    e.kill()
                    p.score += 2
                    self.audio.play("deslizar")
                else:
                    p.hit()
                    self.audio.play("golpe")
        for a in alive:                                          # jugador vs jugador
            for b in alive:
                if a is b or not a.alive or not b.alive or b.invuln > 0:
                    continue
                if not (abs(a.x - b.x) < a.hx + b.hx and abs(a.z - b.z) < a.hz + b.hz
                        and a.y < b.y + b.h and a.y + a.h > b.y):
                    continue
                if a.vy <= 0 and a.prev_y >= b.y + b.h - .4:
                    b.hit()
                    a.vy, a.score = 9.0, a.score + 5
                    a.stomps += 1
                    self.audio.play("pisoton")
                elif a.state == "slide" and b.state != "slide":
                    b.hit()
                    a.score += 2
                    self.audio.play("deslizar")

    def finish(self):
        if self.state == "over":
            return
        self.state = "over"
        self.audio.play("final")
        self.finish_button.enabled = False
        top = max(p.score for p in self.players)
        winners = [p for p in self.players if p.score == top]
        msg = "¡EMPATE!" if len(winners) > 1 else f"¡GANA {winners[0].name.upper()}!"
        scores = "   |   ".join(f"{p.name}: {p.score}" for p in self.players)
        self.over_text = Text(text=f"{msg}\n\nNIVEL {self.level} COMPLETADO\n{scores}",
                              origin=(0, 0), position=(0, .1), scale=1.8, color=color.black)
        self.next_button.enabled = True
        self.menu_button.enabled = True
        self.over_quit_button.enabled = True

    # ---------- telemetría con pandas ----------
    def export_telemetry(self):
        if not self.telemetry:
            return
        df = pd.DataFrame(self.telemetry)
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telemetria")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"telemetria_3d_{datetime.now():%Y%m%d_%H%M%S}.csv")
        df.to_csv(path, index=False)
        resumen = df.groupby(["ronda", "jugador"]).agg(
            puntos=("puntos", "max"), saltos=("saltos", "max"), monedas=("monedas", "max"),
            x_prom=("x", "mean"), y_prom=("y", "mean"), z_prom=("z", "mean")).round(2)
        print("\n=== RESUMEN DE TELEMETRÍA ===")
        print(resumen)
        print(f"Telemetría guardada en: {path}")


# =============================================================================
#  ARRANQUE (Ursina llama a update() e input() definidas en este módulo)
# =============================================================================
app = Ursina(title="Jeidy Super Saltarín - 3D", borderless=False)
window.color = color.cyan.tint(-.1)
game = Juego3D()


def update():
    game.tick(min(time.dt, 1 / 30))


def input(key):
    game.on_key(key)


if __name__ == "__main__":
    try:
        app.run()
    finally:
        game.export_telemetry()
