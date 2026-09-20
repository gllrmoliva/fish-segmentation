# WIP — DINO "erase & re-run" (notebook 11)

Nota de traspaso para continuar después. **Este archivo es temporal**: bórralo
cuando el notebook 11 y su documentación estén terminados (no forma parte de
`docs/`).

## Objetivo

Encontrar delfines más débiles (p. ej. sumergidos) que hoy pierden contra el
pico dominante del heatmap: iterar `detectar → borrar lo encontrado → volver a
detectar` sobre **un solo frame** de
`dataset/processed/20260813_213529_7f994a2f/video.mp4`.

## Por qué puede funcionar (mecanismo real)

1. El pool de referencias de `compute_anomaly_heatmap` son tokens del propio
   frame: un objeto grande/fuerte contamina la referencia "agua" y aplana el
   contraste del resto.
2. Los umbrales son percentiles relativos (99.5 semilla / 98 crecimiento): con
   un pico dominante, el siguiente pico no entra en el top 0.5%; al quitarlo,
   pasa a ser el nuevo top.

Límite honesto: el bucle **no inventa señal**. Un delfín sumergido cuyo contraste
no supere la textura del agua no aparecerá por más iteraciones; la ganancia
esperable es sobre "segundos picos". Además `run_zoom_detector` ya re-ejecuta
DINO en zooms anidados, así que parte de la mejora podría existir ya → el
baseline es obligatorio para medir.

### Trampa de la versión literal ("tokens vacíos")

- `extract_patch_tokens` L2-normaliza: un token a cero queda en vector cero →
  similitud 0 con todo → score 1.0 = **máxima** anomalía (lo contrario de
  borrar).
- Enmascarar tokens *después* del forward no elimina la influencia del objeto:
  la atención del ViT es global. Por eso A/C actúan en el pool y B re-ejecuta el
  forward sobre píxeles borrados.
- Los patches son de 16 px en la entrada escalada (~15 px del frame 4K): dilatar
  la máscara ≥1 patch o quedan patches mixtos que el bucle re-detecta.
- `cv2.inpaint` deja un relleno liso más "anómalo" que el agua → el bucle puede
  re-detectar el agujero. De ahí `leaked_detections` como diagnóstico clave.

## Alcance acordado (respondido por el usuario)

- Notebook 11 con **los 3 modos A/B/C** comparados (más el baseline).
- Máscara de borrado: **SAM3 cajas + fallback DINO** (si el área > `max_area_fraction`
  o el score < `min_score`, se usa bbox/círculo de la detección DINO).
- Medición: A/B contra baseline, candidatos nuevos *fuera* de las zonas
  borradas con score/solidity decentes, y validación visual en varios frames
  (no solo uno). Si el resultado es cero, el notebook igual entrega el A/B y el
  diagnóstico de leakage.

Cuatro variantes sobre el mismo frame:

- **Baseline**: `run_zoom_detector` sin nada (es la iteración 0 de los loops).
- **B — erase (inpaint / patch) + re-forward**: borrar los píxeles con máscara
  SAM3 (o bbox DINO) y volver a correr el detector completo.
- **A — suppress score**: sin tocar píxeles; excluir lo encontrado del pool de
  referencias y del percentil, y re-segmentar.
- **C — suppress tokens**: como A, pero además los tokens de los patches
  encontrados se sustituyen por la media de los supervivientes (nunca ceros:
  `extract_patch_tokens` L2-normaliza y un vector cero tendría score 1.0).

## Estado actual

### Hecho (código)

`src/fish_segmentation/dinov3.py`:

- `compute_anomaly_heatmap(..., suppress_mask=None)`: excluye patches del pool
  de referencias (modo mean y knn) y fija su score al mínimo del mapa después de
  `_local_contrast` (para que el -1/min no se propague por el blur). Valida
  `shape == grid_shape`.
- `_mask_to_patch_grid(mask, grid_shape)`: rasteriza una máscara nativa al grid
  de patches con `INTER_AREA > 0` (un patch cuenta si cualquier píxel cae).
- `replace_masked_tokens(tokens, patch_mask)`: sustituye tokens enmascarados por
  la media normalizada de los no enmascarados; no-op si el mask es todo/ninguno.
- `_run_dino_heatmap(..., suppress_mask=None, suppress_mode="score"|"tokens")`.
- `_tiled_dino_heatmap(..., suppress_mask, suppress_mode)`: recorta la máscara
  en cada tile (también sale del pool local) y la fija en el mapa cosido tras el
  local contrast. El reintento por OOM propaga los kwargs.
- `_compute_dino_heatmap(...)`, `_dino_pass(...)`, `run_dino_view(...)`,
  `run_zoom_detector(..., suppress_mask=None, suppress_mode=...)`: propagan la
  supresión. `suppress_mask` va en coords del frame ORIGINAL (con barras
  letterbox); cada vista la recorta. Condicional: si el mask es `None` **no** se
  pasan los kwargs nuevos (mantiene idénticas las firmas de los dobles de test
  existentes).
- `segment_anomalies(..., valid_mask=None)`: los percentiles se calculan solo
  sobre píxeles válidos; los inválidos se sustituyen por `-inf` en el mapa de
  trabajo para que jamás entren a la máscara (aunque un percentil colapse).
- Nuevas utilidades públicas: `mask_from_detections(detections, size,
  dilate_px=0, shape="bbox"|"radius", radius_scale=1.5)`,
  `suppress_regions(score, mask, fill_value=None)`, `erase_regions(image, mask,
  mode="inpaint"|"patch", inpaint_radius=3, feather_px=5, patch_offset=1.25)`.
- Nuevos loops: `run_dino_erase_loop(...)` y `run_dino_suppress_loop(...)`.
  Ambos devuelven una lista de records con:
  `iteration`, `image`, `detections`, `new_detections` (sin `iteration`),
  `leaked_detections` (crudo dentro de lo ya borrado → artefacto de erase),
  `mask`, `suppressed_mask` (acumulado ANTES de la iteración),
  `accumulated_mask`, y en erase `erased_image`. Iteración 0 = baseline.
  `mask_fn(detections, image) -> bool mask` permite meter la máscara de SAM3.
- `plot_erase_records(image, records)`: un panel por iteración (borrado previo
  en rojo, borrado nuevo en amarillo, candidatos nuevos en lima, leaked en rojo
  ✕) + panel resumen coloreado por iteración.

`src/fish_segmentation/sam3_utils.py`:

- `segment_boxes(processor, image, boxes, label=True)`: wrappea
  `Sam3Processor` (modelo de imagen) con una caja por vez, `reset_all_prompts`
  entre cajas; devuelve `{'mask': bool HxW|None, 'score': float}` del mejor mask.
  **Duck-typed, sin importar sam3** (mantiene el fichero CPU-friendly).
- `_bbox_to_cxcywh(box)`: `[x0,y0,x1,y1]` normalizado → `[cx,cy,w,h]` clampado.
- `detection_erase_mask(processor, image, detections, max_area_fraction=0.25,
  min_score=0.5, fallback_shape="bbox", fallback_radius_scale=1.5)`: caja por
  detección → SAM3; si no hay mask, score bajo, o área > 25% del frame → fallback
  a bbox/círculo DINO. Devuelve bool mask a resolución de imagen. Importa
  `mask_from_detections` de forma perezosa (evita ciclo de imports).

### Tests

- `tests/test_dinov3.py`: añadidos ~10 tests (suppress_mask en heatmap, token
  replace, mask_from_detections, erase_regions inpaint/patch, valid_mask de
  segment_anomalies, `_dino_pass` reenvía `valid_mask`, loops con
  `run_zoom_detector` monkeypatcheado incluyendo detección de leakage y parada).
- `tests/test_sam3_utils.py`: añadidos tests de `segment_boxes` (mejor mask,
  reset, salida vacía), `_bbox_to_cxcywh`, `detection_erase_mask` (fallback por
  área/score y detección sin bbox).

### Comandos

```bash
uv run pytest -q tests/                                  # suite completa (lenta ~2 min solo por imports)
uv run pytest -q tests/test_dinov3.py tests/test_sam3_utils.py
```

La suite completa pasaba 34/34 antes de tocar nada. Tras los cambios, la última
corrida de los dos ficheros dio **35 passed, 2 failed**; los dos fallos eran
bugs reales ya corregidos (`_paste_water_patch` no recortaba al bbox antes de
mezclar; `cv2.circle` no acepta arrays bool) y un test mal construido
(place el token "más fuerte" en el borde del grid, que `border_margin_pct`
suprime). **Pendiente: re-ejecutar la suite completa tras los últimos arreglos.**

## Bloqueo actual — RESUELTO (test reescrito, pendiente de confirmar en suite)

`tests/test_dinov3.py::test_compute_anomaly_heatmap_suppress_mask_pins_and_pools`
fallaba en la línea ~309 porque el `runner_up` estaba en el patch (4,4),
adyacente en diagonal al patch suprimido (3,3): el upsample bicúbico sangraba
(8x8 → 128x128) y el centro del patch suprimido llegaba a 0.145 (umbral 0.1).

Medición exacta (script en background, ya terminado):

```
plain strongest center mean 0.9314   runner center 0.3421
rerun strongest center max   0.145    runner center mean 0.9306
```

Arreglo aplicado: mover el `runner_up` al patch (6,6) (index 54), lejos del
suprimido; el sangrado ya no llega a su centro. El bloqueo es intrínseco a
bicubic, no un bug del pipeline.

Pendiente: confirmar con `uv run pytest -q tests/` (quedó corriendo en
background al escribir esto).

## Siguiente trabajo

1. **Cerrar el test** anterior + `uv run pytest -q tests/` en verde.
2. **Notebook `notebooks/11_dino_erase_rerun.ipynb`** (driver fino, sin
   definiciones; `load_env()` + `ROOT = find_repo_root()` en la primera celda):
   - Config: `FRAME_INDEX`, `DINO_LONG_SIDE=1024`, `DINO_RESOLUTION_SCALE=4.0`,
     `ZONE_MERGE_GAP=0.08`, `MIN_ZONE_FRACTION=0.25`, `MIN_CONFIRMATIONS=2`,
     `MAX_ITERATIONS=2`, `ERASE_DILATE_PX≈12`, `MIN_SCORE=0.5`,
     `MIN_SOLIDITY=0.25`.
   - Frame: `load_sampled_frames(video_path, n_frames=FRAME_INDEX+1)[...]`
     (misma ruta processed con fallback a `data/test_media/dolphin_00.mp4`).
   - Modelos: `load_backbone("facebook/dinov3-vitl16-pretrain-lvd1689m",
     device=device)`, `build_sam3_image_model()` + `Sam3Processor(...)`, y
     `mask_fn = partial(detection_erase_mask, processor=sam3_processor, ...)`.
   - Corridas: `run_dino_erase_loop(erase_mode="inpaint", mask_fn=mask_fn, ...)`,
     `erase_mode="patch"`, `run_dino_suppress_loop(suppress_mode="score")` y
     `("tokens")`, todas con `plot_erase_records(frame, records)`.
   - Comparativa: por modo, nº de candidatos nuevos/leaked por iteración, score
     medio, y crops de validación a ojo. Guardar un JSON resumen (solo metadata
     de detecciones) en `notebooks/outputs/` (gitignored).
   - Cleanup: `del` de modelos + `torch.cuda.empty_cache()`.
3. **Docs**:
   - `docs/notebooks/11_dino_erase_rerun.md` + entrada en `docs/index.md`
     (lista de lectura y support files).
   - `docs/src/fish_segmentation/dinov3.md`: `suppress_mask`/`valid_mask`, loops,
     utilidades de borrado, `plot_erase_records` y gotchas nuevos.
   - `docs/src/fish_segmentation/sam3_utils.md`: `segment_boxes`,
     `detection_erase_mask` (fila en la tabla + gotcha de area/score).
4. Borrar este WIP.

## Decisiones/lecciones para no repetir errores

- **Tokens a cero = score máximo**, no borrado. Media de los supervivientes y
  re-normalizar.
- La supresión debe quitar el patch del **pool de referencias** además de fijar
  su score; si no, el objeto sigue contaminando el "agua".
- Dilatar la máscara de borrado (~1 patch ViT) para no dejar patches mixtos en
  el borde.
- `leaked_detections` es LA métrica: candidato dentro de zona borrada =
  artefacto (agujero de inpaint o costura del patch pegado), no delfín.
- `cv2.inpaint`/`cv2.circle` necesitan uint8, no bool.
- En tests con heatmaps sintéticos, el percentil inferior puede colapsar a 0 y
  floodear el frame; usar fondo con ruido pequeño y picos de área suficiente.
- `_dino_pass`/`run_zoom_detector` pasan los kwargs de supresión solo si hay
  máscara: no rompas esa propiedad o los fakes de los tests existentes fallan.
