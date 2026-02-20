# ATHERIA

ATHERIA ist ein bio-physikalisches Runtime-Framework mit:

- osmotischer Diffusion,
- thermodynamischer Phasenlogik,
- nicht-lokaler Feldinferenz,
- Selbstheilung, Evolution und Selbstreproduktion,
- Population Layer mit Inter-Core-Funktionen:
  - HGT-Symbiose (`SymbiosisLayer`),
  - dynamischem Ressourcenmarkt (`AtherCreditMarket`),
  - kollektivem Inter-Core-Dreaming (`GlobalMorphicNode`).

Technische Gesamtdoku: `Dokumentation.md`  
Ausfuehrlicher Ablauf: `Anleitung.md`

## Schnellstart

### Voraussetzungen

- Python 3.10+
- `numpy`
- `torch`

### Manuelles Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install numpy torch
```

### CLI

```bash
python main.py --help
python main.py demo --duration 3
python main.py meditation --duration 60 --snapshot morphic_snapshot.json
python main.py ceremonial --preheat 10 --duration 60 --snapshot morphic_snapshot.json
```

## 0->100 Automationslauf (PowerShell)

Fuer Windows ist eine End-to-End Pipeline enthalten:

- Datei: `run_atheria_0_to_100.ps1`
- Fuehrt Setup, Build-Checks, Demo, Meditation, Ceremonial und alle Tests in Folge aus.

Voller Lauf:

```powershell
.\run_atheria_0_to_100.ps1
```

Schneller Lauf (ohne lange Meditationsschritte):

```powershell
.\run_atheria_0_to_100.ps1 -SkipMeditation -SkipCeremonial -DemoDuration 1
```

## Tests

Einzeln:

```bash
python -m unittest -v tests/test_digital_life_claims.py
python -m unittest -v tests/test_claim_hardening.py
python -m unittest -v tests/test_demo_forge.py
python -m unittest -v tests/test_collective_extensions.py
```

Kompletter Nachweislauf:

```bash
python -m unittest -v tests/test_digital_life_claims.py tests/test_claim_hardening.py tests/test_demo_forge.py tests/test_collective_extensions.py
```

## DEMO Program Forge

Der Ordner `DEMO` enthaelt einen Generator fuer ausfuehrbare Kind-Programme (Profile, Hashing/Signatur, Harness).

```bash
python DEMO/forge_executable.py --name demo_child --profile survival --output-dir DEMO/generated --run-harness
```

Details: `DEMO/README.md`, `DEMO/Dokumentation.md`
