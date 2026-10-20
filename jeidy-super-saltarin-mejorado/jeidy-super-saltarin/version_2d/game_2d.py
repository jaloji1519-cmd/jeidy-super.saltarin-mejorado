# -*- coding: utf-8 -*-
"""
=============================================================================
 JEIDY SUPER SALTARÍN  -  VERSIÓN 2D (Pygame)
=============================================================================
Modos de juego (se elige en el menú con las teclas 1, 2 o 3):
    1) PC vs USUARIO
    2) PC vs USUARIO vs USUARIO
    3) USUARIO vs USUARIO

Controles
    Jugador 1 (flechas): <- -> caminar | ARRIBA súper salto | ABAJO agacharse
                         ABAJO + <- / -> deslizarse acostado por el suelo
    Jugador 2 (WASD)   : A D caminar   | W súper salto       | S agacharse
                         S + A / D deslizarse acostado
    Recoger objetos    : automático al tocarlos (monedas y hongo)
    ESC                : salir (al cerrar se exporta la telemetría con pandas)

Reglas
    * Monedas = +1 punto.  Hongo = +10 y Jeidy duplica su tamaño (transform.scale);
      siendo grande resiste un golpe extra.
    * Saltar sobre un enemigo (+3) o sobre otro jugador (+5) le hace daño.
      Deslizarse contra un rival también lo golpea (+2).
    * Cada jugador tiene 3 vidas. La ronda dura 90 s o hasta que quede uno vivo.
=============================================================================
"""
import math
import os
import random
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import pygame
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
W, H = 1280, 720
FPS = 60
GRAVITY = 0.8
JUMP_V = -17.5            # súper salto (≈190 px de altura)
WALK = 5.0
SLIDE_SPEED = 9.0
MAX_FALL = 18.0
ROUND_SECONDS = 90
BASE_W, BASE_H = 34, 48   # tamaño base del sprite de Jeidy

GROUND = pygame.Rect(0, 660, W, 60)
PLATFORMS = [
    pygame.Rect(100, 530, 260, 16), pygame.Rect(500, 530, 280, 16), pygame.Rect(920, 530, 260, 16),
    pygame.Rect(300, 400, 260, 16), pygame.Rect(720, 400, 260, 16),
    pygame.Rect(40, 280, 240, 16), pygame.Rect(520, 270, 240, 16), pygame.Rect(1000, 280, 240, 16),
]
SURFACES = [GROUND] + PLATFORMS   # plataformas "de un solo sentido": se atraviesan desde abajo
LEVEL_PLATFORM_LAYOUTS = [
    PLATFORMS,
    [
        pygame.Rect(60, 535, 220, 16), pygame.Rect(390, 520, 250, 16), pygame.Rect(880, 535, 300, 16),
        pygame.Rect(180, 405, 240, 16), pygame.Rect(590, 390, 260, 16), pygame.Rect(930, 395, 250, 16),
        pygame.Rect(50, 275, 220, 16), pygame.Rect(520, 270, 250, 16),
    ],
    [
        pygame.Rect(80, 540, 250, 16), pygame.Rect(440, 520, 260, 16), pygame.Rect(900, 530, 300, 16),
        pygame.Rect(180, 410, 260, 16), pygame.Rect(640, 400, 260, 16), pygame.Rect(950, 390, 220, 16),
        pygame.Rect(50, 280, 220, 16), pygame.Rect(520, 270, 250, 16),
    ],
]
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIR = os.path.join(PROJECT_DIR, "assets")
BACKGROUND_FILES = [os.path.join(ASSET_DIR, "backgrounds", f"nivel_{i}.jpg") for i in range(1, 4)]


class SoundManager:
    """Efectos y ambiente opcionales: el juego continúa si el equipo no tiene audio."""

    def __init__(self):
        self.sounds = {}
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1)
            pygame.mixer.set_num_channels(16)
            volumes = {
                "ambiente": 0.16, "salto": 0.35, "pasos": 0.18, "deslizar": 0.3,
                "caida": 0.3, "moneda": 0.35, "premio": 0.38, "golpe": 0.38,
                "pisoton": 0.35, "final": 0.4,
            }
            for name, volume in volumes.items():
                path = os.path.join(ASSET_DIR, "audio", f"{name}.wav")
                if os.path.isfile(path):
                    sound = pygame.mixer.Sound(path)
                    sound.set_volume(volume)
                    self.sounds[name] = sound
        except pygame.error as exc:
            print("[Audio] Sonido desactivado:", exc)

    def play(self, name, loops=0):
        sound = self.sounds.get(name)
        if sound:
            sound.play(loops=loops)

    def start_ambience(self):
        sound = self.sounds.get("ambiente")
        if sound and not pygame.mixer.get_busy():
            sound.play(loops=-1)


P1_KEYS = (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN)
P2_KEYS = (pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_s)

MODES = {
    1: ("PC vs USUARIO", ["p1", "ai"]),
    2: ("PC vs USUARIO vs USUARIO", ["p1", "p2", "ai"]),
    3: ("USUARIO vs USUARIO", ["p1", "p2"]),
}
ROLE_INFO = {  # nombre, color, teclas
    "p1": ("Jeidy (J1)", (255, 105, 180), P1_KEYS, "jeidy.png"),
    "p2": ("Amiga (J2)", (70, 130, 230), P2_KEYS, "amiga.png"),
    "ai": ("PC-Jeidy", (60, 200, 90), None, "pc.png"),
}

MENU_MODE_BUTTONS = [pygame.Rect(390, 280 + i * 62, 500, 48) for i in range(3)]
MENU_START_BUTTON = pygame.Rect(410, 492, 210, 54)
MENU_EXIT_BUTTON = pygame.Rect(660, 492, 210, 54)
GAME_FINISH_BUTTON = pygame.Rect(1065, 676, 195, 36)
OVER_NEXT_BUTTON = pygame.Rect(365, 398, 220, 44)
OVER_MENU_BUTTON = pygame.Rect(600, 398, 150, 44)
OVER_EXIT_BUTTON = pygame.Rect(765, 398, 150, 44)


# =============================================================================
#  PERSONAJE
# =============================================================================
class Jeidy:
    def __init__(self, name, color, spawn, kind, keys=None, sprite_name=None):
        self.name, self.color, self.spawn, self.kind, self.keys = name, color, spawn, kind, keys
        self.base = self._make_base(sprite_name)
        self.cache = {}
        self.reset()

    def reset(self):
        self.x, self.y = self.spawn          # x = centro, y = pies
        self.vx = self.vy = 0.0
        self.prev_bottom = self.y
        self.on_ground = False
        self.big = False
        self.alive = True
        self.lives = 3
        self.invuln = 0
        self.state = "stand"                 # stand | crouch | slide
        self.facing = 1
        self.score = self.coins = self.jumps = self.stomps = 0
        self.jumped_now = False

    # ---------- sprite ----------
    def _make_base(self, sprite_name=None):
        if sprite_name:
            path = os.path.join(ASSET_DIR, "characters", sprite_name)
            if os.path.isfile(path):
                try:
                    return pygame.image.load(path).convert_alpha()
                except pygame.error as exc:
                    print(f"[Sprite] No se pudo cargar {sprite_name}: {exc}")
        s = pygame.Surface((BASE_W, BASE_H), pygame.SRCALPHA)
        dark = tuple(max(0, c - 80) for c in self.color)
        pygame.draw.rect(s, (40, 40, 60), (6, 36, 9, 12), border_radius=3)      # piernas
        pygame.draw.rect(s, (40, 40, 60), (19, 36, 9, 12), border_radius=3)
        pygame.draw.rect(s, self.color, (3, 19, 28, 21), border_radius=6)        # cuerpo
        pygame.draw.circle(s, (255, 224, 189), (17, 12), 11)                     # cabeza
        pygame.draw.ellipse(s, dark, (5, 0, 24, 11))                             # pelo
        pygame.draw.circle(s, dark, (29, 9), 4)                                  # colita
        pygame.draw.circle(s, (0, 0, 0), (12, 13), 2)                            # ojos
        pygame.draw.circle(s, (0, 0, 0), (22, 13), 2)
        pygame.draw.arc(s, (150, 50, 50), (11, 14, 12, 8), math.pi * 1.1, math.pi * 1.9, 2)
        return s

    @property
    def scale(self):
        return 2 if self.big else 1          # Hongo: el sprite duplica su escala

    @property
    def size(self):
        s = self.scale
        if self.state == "slide":
            return 50 * s, 22 * s
        if self.state == "crouch":
            return 34 * s, 28 * s
        return 34 * s, 48 * s

    @property
    def rect(self):
        w, h = self.size
        return pygame.Rect(int(self.x - w / 2), int(self.y - h), w, h)

    def image(self):
        w, h = self.size
        key = (w, h, self.facing, self.state)
        img = self.cache.get(key)
        if img is None:
            base = self.base
            if self.state == "slide":        # acostado
                base = pygame.transform.rotate(base, -90 if self.facing > 0 else 90)
            elif self.facing < 0:
                base = pygame.transform.flip(base, True, False)
            img = pygame.transform.scale(base, (w, h))   # transform.scale
            self.cache[key] = img
        return img

    # ---------- lógica ----------
    def grow(self):
        self.big = True

    def hit(self):
        if self.invuln > 0 or not self.alive:
            return
        if self.big:                          # el hongo absorbe un golpe
            self.big = False
            self.invuln = 90
        else:
            self.lives -= 1
            self.invuln = 120
            if self.lives <= 0:
                self.alive = False
            else:
                self.x, self.y = self.spawn
                self.vx = self.vy = 0.0

    def update(self, inp):
        left, right, up, down = inp
        self.prev_bottom = self.y
        self.jumped_now = False
        if self.invuln > 0:
            self.invuln -= 1
        direction = (1 if right else 0) - (1 if left else 0)
        if direction:
            self.facing = direction

        if self.on_ground and down and direction:
            self.state = "slide"
        elif self.on_ground and down:
            self.state = "crouch"
        else:
            self.state = "stand"

        if self.state == "slide":
            self.vx = SLIDE_SPEED * direction
        elif self.state == "crouch":
            self.vx = 0.0
        else:
            self.vx = WALK * direction

        if up and self.on_ground and self.state == "stand":      # SÚPER SALTO
            self.vy = JUMP_V
            self.on_ground = False
            self.jumps += 1
            self.jumped_now = True
        if down and not self.on_ground:
            self.vy += 0.6                                        # caída rápida

        self.vy = min(self.vy + GRAVITY, MAX_FALL)
        w, _ = self.size
        self.x = max(w / 2, min(W - w / 2, self.x + self.vx))
        self.y += self.vy

        self.on_ground = False
        if self.vy >= 0:
            r = self.rect
            for p in SURFACES:
                if r.right > p.left and r.left < p.right and self.prev_bottom <= p.top + 1 <= self.y + 1:
                    self.y, self.vy, self.on_ground = p.top, 0.0, True
                    break


class Enemy:
    def __init__(self, minx, maxx, ground_y, speed=1.6):
        self.minx, self.maxx, self.gy, self.speed = minx, maxx, ground_y, speed
        self.x = random.uniform(minx, maxx)
        self.dir = random.choice((-1, 1))
        self.alive, self.timer = True, 0

    @property
    def rect(self):
        return pygame.Rect(int(self.x - 18), self.gy - 30, 36, 30)

    def update(self):
        if not self.alive:
            self.timer -= 1
            if self.timer <= 0:
                self.alive = True
            return
        self.x += self.dir * self.speed
        if self.x < self.minx or self.x > self.maxx:
            self.dir *= -1


# =============================================================================
#  JUEGO
# =============================================================================
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Jeidy Super Saltarín - 2D")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.f_big = pygame.font.SysFont("arial", 56, bold=True)
        self.f_mid = pygame.font.SysFont("arial", 30, bold=True)
        self.f_small = pygame.font.SysFont("arial", 20, bold=True)
        self.backgrounds = []
        for path in BACKGROUND_FILES:
            try:
                image = pygame.image.load(path).convert()
                self.backgrounds.append(pygame.transform.smoothscale(image, (W, H)))
            except (pygame.error, OSError):
                self.backgrounds.append(None)
        self.sound = SoundManager()
        self.draw_center("Cargando IA (scikit-learn + Keras)...", H // 2)
        self.brain = AIBrain()
        self.running = True
        self.state = "menu"
        self.players, self.enemies, self.coins = [], [], []
        self.mushroom = None
        self.ai_state = {}
        self.telemetry = []        # filas para el DataFrame de pandas
        self.round_no = 0
        self.mode = 1
        self.level = 1

    # ---------- utilidades ----------
    def draw_center(self, text, y, font=None, color=(255, 255, 255)):
        font = font or self.f_mid
        self.screen.fill((20, 24, 48)) if text.startswith("Cargando") else None
        img = font.render(text, True, color)
        self.screen.blit(img, (W // 2 - img.get_width() // 2, y))
        if text.startswith("Cargando"):
            pygame.display.flip()

    def start_round(self, mode):
        global PLATFORMS, SURFACES
        PLATFORMS = [pygame.Rect(p) for p in LEVEL_PLATFORM_LAYOUTS[(self.level - 1) % len(LEVEL_PLATFORM_LAYOUTS)]]
        SURFACES = [GROUND] + PLATFORMS
        self.mode, self.state = mode, "play"
        self.round_no += 1
        roles = MODES[mode][1]
        xs = [200, 1080] if len(roles) == 2 else [200, 640, 1080]
        self.players = []
        for x, role in zip(xs, roles):
            name, color, keys, sprite_name = ROLE_INFO[role]
            self.players.append(Jeidy(name, color, (x, 660), "ai" if role == "ai" else "human", keys, sprite_name))
        self.enemies = [Enemy(260, 1000, 660)] + [
            Enemy(p.left + 18, p.right - 18, p.top) for p in (PLATFORMS[1], PLATFORMS[4], PLATFORMS[6])]
        self.coins, self.mushroom, self.ai_state = [], None, {}
        for _ in range(8):
            self.spawn_coin()
        self.time_left, self.frame = float(ROUND_SECONDS), 0
        self.sound.start_ambience()
        self.sound.play("premio")

    def spawn_coin(self):
        s = random.choice(SURFACES)
        x = random.randint(s.left + 20, s.right - 20)
        self.coins.append(pygame.Rect(x - 10, s.top - 40, 20, 20))

    def spawn_mushroom(self):
        s = random.choice(PLATFORMS)
        x = random.randint(s.left + 25, s.right - 25)
        self.mushroom = pygame.Rect(x - 15, s.top - 30, 30, 30)

    # ---------- IA ----------
    def nearest_collectible(self, p):
        best, bd = None, 1e9
        for c in self.coins:
            d = math.hypot(c.centerx - p.x, c.centery - p.rect.centery)
            if d < bd:
                best, bd = c.center, d
        if self.mushroom is not None and not p.big:
            m = self.mushroom
            d = math.hypot(m.centerx - p.x, m.centery - p.rect.centery) * 0.5
            if d < bd:
                best = m.center
        return best

    def features(self, p, tx, ty):
        cx, cy = p.x, p.rect.centery
        dx, dy = (tx - cx) / W, (ty - cy) / H
        dirn = 1 if dx >= 0 else -1
        haz = 0.0
        for e in self.enemies:
            if e.alive and abs(e.rect.centery - cy) < 60 and 0 < (e.rect.centerx - cx) * dirn < 110:
                haz = 1.0
                break
        return [dx, dy, 1.0 if p.on_ground else 0.0, haz]

    def ai_target(self, p):
        humans = [q for q in self.players if q.kind == "human" and q.alive]
        if humans:
            h = min(humans, key=lambda q: math.hypot(q.x - p.x, q.y - p.y))
            if math.hypot(h.x - p.x, h.y - p.y) < 300:        # interceptar al jugador
                return h.x, h.rect.top - 70
        t = self.nearest_collectible(p)
        return t if t else (W / 2, p.rect.centery)

    def ai_input(self, p):
        st = self.ai_state.setdefault(p, {"t": 0, "inp": (False, False, False, False)})
        st["t"] -= 1
        if st["t"] <= 0:
            st["t"] = 5                                         # "piensa" cada 5 frames
            tx, ty = self.ai_target(p)
            st["inp"] = self.brain.decide(self.features(p, tx, ty))
        return st["inp"]

    # ---------- bucle ----------
    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if self.state == "menu":
                    for i, rect in enumerate(MENU_MODE_BUTTONS, start=1):
                        if rect.collidepoint(ev.pos):
                            self.mode = i
                    if MENU_START_BUTTON.collidepoint(ev.pos):
                        self.start_round(self.mode)
                    elif MENU_EXIT_BUTTON.collidepoint(ev.pos):
                        self.running = False
                elif self.state == "play" and GAME_FINISH_BUTTON.collidepoint(ev.pos):
                    self.finish_round()
                elif self.state == "over":
                    if OVER_NEXT_BUTTON.collidepoint(ev.pos):
                        self.level = self.level % 3 + 1
                        self.start_round(self.mode)
                    elif OVER_MENU_BUTTON.collidepoint(ev.pos):
                        self.state = "menu"
                    elif OVER_EXIT_BUTTON.collidepoint(ev.pos):
                        self.running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self.running = False
                elif self.state == "menu" and ev.unicode in ("1", "2", "3"):
                    self.start_round(int(ev.unicode))
                elif self.state == "over" and ev.key == pygame.K_r:
                    self.state = "menu"
                elif self.state == "over" and ev.key == pygame.K_n:
                    self.level = self.level % 3 + 1
                    self.start_round(self.mode)

    def finish_round(self):
        if self.state == "play":
            self.state = "over"
            self.sound.play("final")

    def update(self):
        self.frame += 1
        self.time_left -= 1 / FPS
        keys = pygame.key.get_pressed()

        for p in self.players:
            if not p.alive:
                continue
            was_ground, old_state = p.on_ground, p.state
            if p.kind == "human":
                inp = tuple(bool(keys[k]) for k in p.keys)
                tgt = self.nearest_collectible(p)
                f = self.features(p, *tgt) if tgt else None
                p.update(inp)
                if f and was_ground and self.frame % 6 == 0:    # muestras para imitar saltos
                    self.brain.add_sample(f, p.jumped_now)
            else:
                inp = self.ai_input(p)
                p.update(inp)
            if p.jumped_now:
                self.sound.play("salto")
            elif not was_ground and p.on_ground:
                self.sound.play("caida")
            if p.state == "slide" and old_state != "slide":
                self.sound.play("deslizar")
            elif (inp[0] or inp[1]) and p.on_ground and self.frame % 24 == 0:
                self.sound.play("pasos")
        if self.frame % 1200 == 0:
            self.brain.retrain_async()

        for e in self.enemies:
            e.update()
        self.collisions()

        if self.frame % 120 == 0 and len(self.coins) < 12:
            self.spawn_coin()
        if self.mushroom is None and self.frame % 600 == 0:
            self.spawn_mushroom()

        if self.frame % 15 == 0:                                # telemetría (pandas)
            for p in self.players:
                self.telemetry.append({
                    "ronda": self.round_no, "modo": MODES[self.mode][0], "t": round(self.frame / FPS, 2),
                    "jugador": p.name, "x": round(p.x, 1), "y": round(p.y, 1), "saltos": p.jumps,
                    "monedas": p.coins, "puntos": p.score, "vidas": p.lives, "grande": int(p.big),
                    "estado": p.state})

        if self.time_left <= 0 or sum(p.alive for p in self.players) <= 1:
            self.finish_round()

    def collisions(self):
        alive = [p for p in self.players if p.alive]
        for p in alive:
            r = p.rect
            for c in self.coins[:]:                              # monedas (automático)
                if r.colliderect(c):
                    self.coins.remove(c)
                    p.coins += 1
                    p.score += 1
                    self.sound.play("moneda")
            if self.mushroom is not None and r.colliderect(self.mushroom):   # hongo
                self.mushroom = None
                p.grow()
                p.score += 10
                self.sound.play("premio")
            for e in self.enemies:                               # enemigos
                if not e.alive or not r.colliderect(e.rect):
                    continue
                if p.vy >= 0 and p.prev_bottom <= e.rect.top + 12:
                    e.alive, e.timer = False, 300
                    p.vy, p.score = -11, p.score + 3
                    p.stomps += 1
                    self.sound.play("pisoton")
                elif p.state == "slide":
                    e.alive, e.timer = False, 300
                    p.score += 2
                    self.sound.play("deslizar")
                else:
                    p.hit()
                    self.sound.play("golpe")
        for a in alive:                                          # jugador vs jugador
            for b in alive:
                if a is b or not a.alive or not b.alive or b.invuln > 0:
                    continue
                if not a.rect.colliderect(b.rect):
                    continue
                if a.vy >= 0 and a.prev_bottom <= b.rect.top + 12:
                    b.hit()
                    a.vy, a.score = -11, a.score + 5
                    a.stomps += 1
                    self.sound.play("pisoton")
                elif a.state == "slide" and b.state != "slide":
                    b.hit()
                    a.score += 2
                    self.sound.play("deslizar")

    # ---------- dibujo ----------
    def draw(self):
        s = self.screen
        background = self.backgrounds[(self.level - 1) % len(self.backgrounds)] if self.backgrounds else None
        if background:
            s.blit(background, (0, 0))
        else:
            s.fill((135, 206, 235))
            for cx, cy in ((200, 90), (700, 60), (1100, 120)):
                for dx, r in ((-30, 28), (0, 36), (30, 28)):
                    pygame.draw.circle(s, (255, 255, 255), (cx + dx, cy), r)
        for p in PLATFORMS:
            pygame.draw.rect(s, (139, 90, 43), p, border_radius=4)
            pygame.draw.rect(s, (80, 180, 70), (p.x, p.y, p.w, 6), border_radius=4)
        pygame.draw.rect(s, (110, 70, 35), GROUND)
        pygame.draw.rect(s, (80, 180, 70), (0, GROUND.y, W, 10))

        if self.state == "menu":
            self.draw_menu()
        else:
            self.draw_world()
            if self.state == "over":
                self.draw_over()
        pygame.display.flip()

    def draw_menu(self):
        s = self.screen
        pan = pygame.Surface((800, 520), pygame.SRCALPHA)
        pan.fill((20, 24, 48, 215))
        s.blit(pan, (240, 85))
        lines = [
            ("JEIDY SUPER SALTARÍN", self.f_big, (255, 220, 80)),
            (f"NIVEL {self.level} DE 3  ·  FONDO {self.level}", self.f_small, (200, 230, 255)),
            ("Elige un modo y pulsa INICIAR", self.f_small, (230, 230, 245)),
        ]
        y = 125
        for text, font, color in lines:
            img = font.render(text, True, color)
            s.blit(img, (W // 2 - img.get_width() // 2, y))
            y += img.get_height() + 16
        mode_names = [
            "1  ·  PC vs JUGADOR",
            "2  ·  PC vs DOS JUGADORES",
            "3  ·  DOS JUGADORES",
        ]
        for i, (rect, text) in enumerate(zip(MENU_MODE_BUTTONS, mode_names), start=1):
            selected = self.mode == i
            pygame.draw.rect(s, (55, 130, 180) if selected else (55, 65, 95), rect, border_radius=10)
            pygame.draw.rect(s, (255, 220, 80) if selected else (140, 155, 185), rect, 2, border_radius=10)
            label = self.f_small.render(text, True, (255, 255, 255))
            s.blit(label, label.get_rect(center=rect.center))
        self.draw_button(MENU_START_BUTTON, "INICIAR JUEGO", (40, 150, 95))
        self.draw_button(MENU_EXIT_BUTTON, "SALIR", (155, 65, 75))
        controls = self.f_small.render("J1: flechas  |  J2: WASD  |  ESC: salir", True, (220, 225, 240))
        s.blit(controls, controls.get_rect(center=(W // 2, 620)))

    def draw_button(self, rect, label, color):
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=10)
        text = self.f_small.render(label, True, (255, 255, 255))
        self.screen.blit(text, text.get_rect(center=rect.center))

    def draw_world(self):
        s = self.screen
        for c in self.coins:
            pygame.draw.circle(s, (255, 200, 0), c.center, 10)
            pygame.draw.circle(s, (255, 240, 120), c.center, 6)
        if self.mushroom:
            m = self.mushroom
            pygame.draw.rect(s, (250, 235, 205), (m.x + 8, m.y + 14, 14, 16))
            pygame.draw.ellipse(s, (220, 40, 40), (m.x, m.y, 30, 22))
            pygame.draw.circle(s, (255, 255, 255), (m.x + 9, m.y + 9), 4)
            pygame.draw.circle(s, (255, 255, 255), (m.x + 21, m.y + 9), 4)
        for e in self.enemies:
            if e.alive:
                r = e.rect
                pygame.draw.rect(s, (120, 40, 40), r, border_radius=10)
                pygame.draw.circle(s, (255, 255, 255), (r.x + 11, r.y + 11), 5)
                pygame.draw.circle(s, (255, 255, 255), (r.x + 25, r.y + 11), 5)
                pygame.draw.circle(s, (0, 0, 0), (r.x + 12, r.y + 12), 2)
                pygame.draw.circle(s, (0, 0, 0), (r.x + 26, r.y + 12), 2)
        for p in self.players:
            if p.alive and not (p.invuln > 0 and (p.invuln // 4) % 2):
                r = p.rect
                s.blit(p.image(), r)
                tag = self.f_small.render(p.name, True, (0, 0, 0))
                s.blit(tag, (r.centerx - tag.get_width() // 2, r.top - 24))
        # HUD
        for i, p in enumerate(self.players):
            x = 20 + i * 420
            pygame.draw.rect(s, (20, 24, 48), (x, 8, 400, 34), border_radius=8)
            pygame.draw.rect(s, p.color, (x, 8, 8, 34), border_radius=4)
            txt = f"{p.name}  Pts:{p.score}  Mon:{p.coins}  Vidas:{max(p.lives, 0)}" + ("  [GRANDE]" if p.big else "")
            s.blit(self.f_small.render(txt, True, (255, 255, 255)), (x + 16, 13))
        t = self.f_mid.render(f"{max(0, int(self.time_left))}s", True, (20, 24, 48))
        s.blit(t, (W // 2 - t.get_width() // 2, 46))
        label = self.f_small.render(MODES[self.mode][0], True, (20, 24, 48))
        s.blit(label, (W // 2 - label.get_width() // 2, 82))
        level_label = self.f_small.render(f"NIVEL {self.level}/3", True, (20, 24, 48))
        s.blit(level_label, (W // 2 - level_label.get_width() // 2, 108))
        self.draw_button(GAME_FINISH_BUTTON, "FINALIZAR NIVEL", (150, 55, 65))

    def draw_over(self):
        s = self.screen
        pan = pygame.Surface((720, 310), pygame.SRCALPHA)
        pan.fill((20, 24, 48, 225))
        s.blit(pan, (280, 165))
        top = max(p.score for p in self.players)
        winners = [p for p in self.players if p.score == top]
        msg = "¡EMPATE!" if len(winners) > 1 else f"¡GANA {winners[0].name.upper()}!"
        for text, font, color, y in ((msg, self.f_big, (255, 220, 80), 215),
                                     ("  |  ".join(f"{p.name}: {p.score}" for p in self.players), self.f_small, (255, 255, 255), 310),
                                     (f"NIVEL {self.level} COMPLETADO", self.f_small, (200, 230, 255), 350)):
            img = font.render(text, True, color)
            s.blit(img, (W // 2 - img.get_width() // 2, y))
        self.draw_button(OVER_NEXT_BUTTON, "SIGUIENTE NIVEL", (40, 150, 95))
        self.draw_button(OVER_MENU_BUTTON, "MENÚ", (65, 105, 165))
        self.draw_button(OVER_EXIT_BUTTON, "FINALIZAR", (155, 65, 75))

    # ---------- telemetría con pandas ----------
    def export_telemetry(self):
        if not self.telemetry:
            return
        df = pd.DataFrame(self.telemetry)
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telemetria")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"telemetria_2d_{datetime.now():%Y%m%d_%H%M%S}.csv")
        df.to_csv(path, index=False)
        resumen = df.groupby(["ronda", "jugador"]).agg(
            puntos=("puntos", "max"), saltos=("saltos", "max"), monedas=("monedas", "max"),
            x_prom=("x", "mean"), y_prom=("y", "mean")).round(1)
        print("\n=== RESUMEN DE TELEMETRÍA ===")
        print(resumen)
        print(f"Telemetría guardada en: {path}")

    def run(self):
        try:
            while self.running:
                self.handle_events()
                if self.state == "play":
                    self.update()
                self.draw()
                self.clock.tick(FPS)
        finally:
            self.export_telemetry()
            pygame.quit()


if __name__ == "__main__":
    Game().run()
