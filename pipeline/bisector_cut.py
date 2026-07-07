# bisector_cut.py   (Tarea 2.2 — fichero NUEVO, no modifica el pipeline existente)
# =============================================================================
# Corte geométrico de instrumentos fusionados ORIENTADO POR LA CONCAVIDAD.
#
# Motivación
# ----------
# `pipeline/concave_cut.py` (producción) traza la recta de separación
# PERPENDICULAR al eje largo de la bounding box.  Esa orientación es arbitraria:
# no tiene en cuenta la geometría real del punto de contacto y, cuando los dos
# instrumentos se cruzan en ángulo, la recta no separa limpiamente las piezas.
#
# El contorno de dos herramientas en contacto forma, en el punto cóncavo, una
# "V" (\/).  La recta de separación natural es la BISECTRIZ del ángulo de esa V
# (\|/): la dirección que reparte por igual el ángulo entre los dos segmentos
# del contorno que llegan al punto cóncavo.  Este módulo calcula esa bisectriz a
# partir de los vecinos ±k del punto cóncavo (los mismos que usa el test de
# concavidad del paper de Miró-Nicolau et al., 2024) y traza el corte a lo largo
# de ella, conservando la longitud del corte original.
#
# API pública
# -----------
#   bisector_direction(contour, idx, k)         → vector unitario de la bisectriz
#   compute_bisector_cuts(seg_binary, bboxes, best_points, cfg)
#                                                → dict[idx → geometría del corte]
#   apply_bisector_cuts(seg_binary, bboxes, best_points, cfg)
#                                                → máscara binaria con los cortes
#
# Para activarlo en producción bastaría con sustituir, en run_pipeline.py, la
# llamada a `apply_concave_cuts` por `apply_bisector_cuts` (misma firma).  No se
# hace aquí para respetar la restricción de no tocar ficheros existentes.
#
# Arquitectura: cero imports de matplotlib.  Devuelve datos.

from __future__ import annotations

import numpy as np
import cv2 as cv

from config import PreprocessConfig
from pipeline.concave_points import rdp_simplify


# --------------------------------------------------------------------------- #
#  Contorno simplificado de una bbox anómala                                  #
# --------------------------------------------------------------------------- #

def simplified_contour_for_bbox(
    seg_binary: np.ndarray,
    bbox: tuple,
    cfg: PreprocessConfig,
) -> np.ndarray | None:
    """Devuelve el contorno simplificado (RDP) de la bbox, en coords de imagen.

    Replica EXACTAMENTE el recorte y la simplificación de
    `pipeline.concave_points.detect_concave_points`, de modo que los índices de
    vértice coinciden con los que produjeron el punto cóncavo elegido.

    Args:
        seg_binary: máscara binaria uint8 (H, W); instrumentos = 255.
        bbox:       (center, size, angle, area).
        cfg:        PreprocessConfig (cp_rdp_epsilon, seg_min_contour_area).

    Returns:
        Array (M, 2) float64 de vértices en coordenadas de imagen completa,
        o None si el contorno no es extraíble.
    """
    h_img, w_img = seg_binary.shape
    center, size, angle = bbox[0], bbox[1], bbox[2]
    box_pts = np.int32(cv.boxPoints((center, size, angle)))

    ax, ay, aw, ah = cv.boundingRect(box_pts)
    x1, y1 = max(0, ax), max(0, ay)
    x2, y2 = min(w_img, ax + aw), min(h_img, ay + ah)
    if x2 <= x1 or y2 <= y1:
        return None

    roi_crop = seg_binary[y1:y2, x1:x2]
    bbox_mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
    cv.fillPoly(bbox_mask_full, [box_pts], 255)
    roi_bin = cv.bitwise_and(roi_crop, bbox_mask_full[y1:y2, x1:x2])

    cnts, _ = cv.findContours(roi_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    largest = max(cnts, key=cv.contourArea)
    if cv.contourArea(largest) < cfg.seg_min_contour_area:
        return None

    simplified = rdp_simplify(largest, epsilon=cfg.cp_rdp_epsilon)
    return simplified + np.array([[x1, y1]], dtype=np.float64)


# --------------------------------------------------------------------------- #
#  Bisectriz del ángulo del contorno en un punto                              #
# --------------------------------------------------------------------------- #

def bisector_direction(contour: np.ndarray, idx: int, k: int) -> np.ndarray:
    """Vector unitario de la bisectriz del ángulo del contorno en `idx`.

    Sean P = contour[idx] el punto cóncavo y A = contour[idx-k],
    B = contour[idx+k] sus vecinos a ±k pasos (los mismos del test de
    concavidad).  Los dos segmentos del contorno que forman la "V" son P→A y
    P→B.  La bisectriz INTERNA del ángulo ∠APB es la suma de los vectores
    unitarios desde P:

        bis = û + v̂ ,   con  u = A − P ,  v = B − P .

    Esa dirección apunta a lo largo del eje de simetría de la V (la buscada).
    El corte se traza a lo largo de ella (en ambos sentidos), de modo que la
    porción que entra en el cuerpo fusionado secciona el "cuello" que une las
    dos herramientas.

    Caso degenerado: si la V está casi alineada (ángulo ≈ 180°, û+v̂ ≈ 0), la
    bisectriz no está definida; se cae a la NORMAL de la cuerda A→B (la
    "tangente por el punto normal" que el enunciado admite como alternativa).

    Args:
        contour: (N, 2) array de vértices.
        idx:     índice del punto cóncavo.
        k:       paso de vecindad ±k (cfg.cp_k).

    Returns:
        Vector unitario (2,) float64.
    """
    contour = np.asarray(contour, dtype=np.float64)
    N = len(contour)
    P = contour[idx % N]
    A = contour[(idx - k) % N]
    B = contour[(idx + k) % N]

    u = A - P
    v = B - P
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        # vecinos coincidentes con P → usa la normal de la cuerda
        chord = B - A
        bis = np.array([-chord[1], chord[0]], dtype=np.float64)
    else:
        bis = u / nu + v / nv

    nb = np.linalg.norm(bis)
    if nb < 1e-6:
        # ángulo ≈ 180°: bisectriz indefinida → normal de la cuerda (tangente⊥)
        chord = B - A
        bis = np.array([-chord[1], chord[0]], dtype=np.float64)
        nb = np.linalg.norm(bis)
        if nb < 1e-9:
            return np.array([0.0, 1.0])
    return bis / nb


def _nearest_vertex(contour: np.ndarray, x: float, y: float) -> int:
    """Índice del vértice del contorno más cercano a (x, y)."""
    d = contour - np.array([[x, y]], dtype=np.float64)
    return int(np.argmin(np.einsum('ij,ij->i', d, d)))


# --------------------------------------------------------------------------- #
#  Geometría del corte por bisectriz                                          #
# --------------------------------------------------------------------------- #

def compute_bisector_cuts(
    seg_binary: np.ndarray,
    outlier_bboxes: list[tuple],
    best_points: dict[int, dict | None],
    cfg: PreprocessConfig,
) -> dict[int, dict]:
    """Calcula la recta de corte por bisectriz para cada bbox anómala.

    Para cada bbox con punto cóncavo elegido:
      1. Reconstruye el contorno simplificado de la bbox.
      2. Localiza el vértice más cercano al punto cóncavo (pt['x'], pt['y']).
      3. Calcula la bisectriz del ángulo del contorno en ese vértice.
      4. Define los extremos del corte: ± media diagonal del bbox (= 1 diagonal
         total), de modo que la recta cruza el blob por completo y secciona el
         cuello aunque sea ancho.

    Returns:
        dict[bbox_idx → {
            'p':          (x, y) punto cóncavo (centro del corte),
            'a':          (x, y) vecino −k del contorno,
            'b':          (x, y) vecino +k del contorno,
            'dir':        (dx, dy) vector unitario de la bisectriz,
            'angle_deg':  orientación de la bisectriz en grados,
            'p1', 'p2':   extremos del segmento de corte (x, y),
            'half_len':   semilongitud del corte (px),
        }].  Sólo incluye bboxes con punto cóncavo y contorno válidos.
    """
    k = cfg.cp_k
    cuts: dict[int, dict] = {}

    for bbox_idx, bbox in enumerate(outlier_bboxes):
        pt = best_points.get(bbox_idx)
        if pt is None:
            continue

        contour = simplified_contour_for_bbox(seg_binary, bbox, cfg)
        if contour is None or len(contour) < 2 * k + 1:
            continue

        idx = _nearest_vertex(contour, pt['x'], pt['y'])
        direction = bisector_direction(contour, idx, k)

        _center, size, _angle, _area = bbox
        w, h = size
        # El corte debe CRUZAR por completo el blob fusionado a lo largo de la
        # bisectriz.  El punto cóncavo P está sobre el borde del contorno, así
        # que media diagonal en CADA sentido garantiza que la recta alcanza el
        # lado opuesto del cuello sea cual sea la orientación.  El valor anterior
        # (long_side/8) sólo penetraba ~1/8 del cuerpo desde P: en cuellos
        # anchos el blob quedaba mellado pero NO seccionado, seguía conectado y
        # la re-segmentación devolvía la misma bbox única (sin separación).
        half_len = 0.5 * float(np.hypot(w, h))

        cx, cy = float(pt['x']), float(pt['y'])
        dx, dy = direction * half_len
        N = len(contour)

        cuts[bbox_idx] = {
            'p':         (cx, cy),
            'a':         tuple(contour[(idx - k) % N]),
            'b':         tuple(contour[(idx + k) % N]),
            'dir':       (float(direction[0]), float(direction[1])),
            'angle_deg': float(np.degrees(np.arctan2(direction[1], direction[0]))),
            'p1':        (cx - dx, cy - dy),
            'p2':        (cx + dx, cy + dy),
            'half_len':  float(half_len),
        }

    return cuts


def apply_bisector_cuts(
    seg_binary: np.ndarray,
    outlier_bboxes: list[tuple],
    best_points: dict[int, dict | None],
    cfg: PreprocessConfig,
) -> np.ndarray:
    """Copia de seg_binary con un corte negro a lo largo de la bisectriz.

    Misma firma que `pipeline.concave_cut.apply_concave_cuts`: es un sustituto
    directo (drop-in) que cambia ÚNICAMENTE la orientación de la recta de corte
    (bisectriz del contorno en lugar de perpendicular al eje de la bbox).

    Returns:
        máscara binaria uint8 (H, W) con los cortes dibujados; no modifica la
        entrada.
    """
    cut_mask = seg_binary.copy()
    line_width = cfg.concave_cut_line_width
    cuts = compute_bisector_cuts(seg_binary, outlier_bboxes, best_points, cfg)

    for cut in cuts.values():
        p1, p2 = cut['p1'], cut['p2']
        cv.line(
            cut_mask,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            color=0, thickness=line_width,
        )
    return cut_mask
