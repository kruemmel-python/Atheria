# DEMO - Executable Program Forge

Dieser Ordner enthaelt ein vollstaendiges Generator-Tool, das ein neues ausfuehrbares Kind-Programm inklusive Integritaetsdaten erzeugt.

Ausfuehrliche Beschreibung und Ergebnisprotokoll:

- `DEMO/Dokumentation.md`

## Dateien

- `DEMO/forge_executable.py`
- `DEMO/__init__.py`

## Hauptfunktionen

- Profil-Templates: `survival`, `diagnostic`, `stress-test`
- Artefakt-Hashing: SHA-256 pro erzeugter Datei
- Signierung: HMAC-SHA256 (Schluessel aus Datei/ENV oder auto-generiert)
- Optionales Harness: automatischer Lauf + Validierung (`.py`, `.pyz`, Zustand, Integritaet)
- Integration in ATHERIA-Lineage:
  - `SelfReproductionEngine` kann automatisch Artefakte unter `DEMO/lineage/<offspring>` erzeugen,
  - inklusive Integritaetsbundle und optionaler Harness-Validierung.

## Standard-Aufruf

```bash
python DEMO/forge_executable.py --name atheria_child --profile survival --output-dir DEMO/generated --run-harness
```

## Profile

```bash
python DEMO/forge_executable.py --name child_survival --profile survival --output-dir DEMO/generated_survival --run-harness
python DEMO/forge_executable.py --name child_diag --profile diagnostic --output-dir DEMO/generated_diagnostic --run-harness
python DEMO/forge_executable.py --name child_stress --profile stress-test --output-dir DEMO/generated_stress --run-harness
```

## Signierung

Eigener Schluessel:

```bash
python DEMO/forge_executable.py --name child_signed --signing-key DEMO/my_signing.key
```

Schluessel aus Environment:

```bash
# Beispiel: base64-key
set ATHERIA_DEMO_SIGNING_KEY=base64:QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=
python DEMO/forge_executable.py --name child_signed_env
```

## Harness (optional)

```bash
python DEMO/forge_executable.py --name child_test --run-harness --harness-iterations 2 --harness-interval 0.0
```

## Tests

```bash
python -m unittest -v tests/test_demo_forge.py
```

Kompletter Projektlauf inkl. Forge:

```bash
python -m unittest -v tests/test_digital_life_claims.py tests/test_claim_hardening.py tests/test_demo_forge.py tests/test_collective_extensions.py
```

## Native EXE (optional)

```bash
python DEMO/forge_executable.py --name child_native --build-exe
```

Hinweis: `--build-exe` benoetigt `pyinstaller`.
