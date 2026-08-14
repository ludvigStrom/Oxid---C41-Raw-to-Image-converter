# Camera IDT profiles (camera → ACEScg)

Place JSON files here to add **Input Device Transform** presets for Use ACEScg. Each file defines a 3×3 matrix that converts **linear camera RGB** to **ACEScg**.

## JSON format

```json
{
  "name": "Sony A7R II",
  "matrix": [
    [0.754, 0.021, -0.010],
    [0.134, 1.005, 0.005],
    [0.112, -0.027, 1.005]
  ]
}
```

- **name**: Label shown in the GUI dropdown.
- **matrix**: 3×3 row-major (row i × [R,G,B] → channel i). Values are floats.

## Where to get matrices

- [AMPAS aces-dev](https://github.com/ampas/aces-dev) – vendor IDTs in CTL; the 3×3 linear step can be extracted.
- [ACES documentation](https://docs.acescentral.com/) – Input Transforms and IDT specs.
- Camera vendors sometimes publish ACES IDTs; convert from their format to the JSON above.

## Notes

- Our pipeline uses **linear** sensor/camera RGB (after demosaic, no log curve). Use IDTs that expect linear camera space (e.g. linear SGamut for Sony), not log-encoded.
- "Identity" (no transform) is always available in the dropdown; use it when no camera-specific IDT is needed.
