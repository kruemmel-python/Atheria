# ATHERIA Anleitung: Von 0 auf 100

Diese Anleitung fuehrt dich vom leeren System bis zum vollstaendigen Nachweislauf.
Ziel: Alles lokal ausfuehren, inklusive DEMO-Forge, Reproduktion mit Artefakt-Erzeugung,
Population Layer (HGT/Markt/Inter-Core-Dreaming) und allen Tests.

## 1. Voraussetzungen

- Betriebssystem: Windows (PowerShell) oder Linux/macOS (Shell)
- Python: 3.10 oder neuer
- `pip` verfuegbar
- Python-Pakete: `numpy`, `torch`
- Optional fuer native EXE-Builds: `pyinstaller`

Projektordner in dieser Anleitung:

- `d:\ATHERIA`

## 2. Projekt oeffnen

```powershell
cd d:\ATHERIA
```

## 3. Virtuelle Umgebung erstellen

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Abhaengigkeiten installieren

Pflicht:

```powershell
pip install --upgrade pip setuptools wheel
pip install numpy torch
```

Optional (nur wenn du `--build-exe` nutzen willst):

```powershell
pip install pyinstaller
```

## 5. Schnellcheck der Installation

```powershell
python -V
python -m py_compile main.py atheria_core.py DEMO/forge_executable.py
python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"
```

Wenn kein Fehler kommt, ist die Basis lauffaehig.

## 6. Automatischer 0->100 Lauf (empfohlen)

Windows End-to-End Pipeline:

```powershell
.\run_atheria_0_to_100.ps1
```

Schneller Lauf (ohne lange Meditation/Ceremonial):

```powershell
.\run_atheria_0_to_100.ps1 -SkipMeditation -SkipCeremonial -DemoDuration 1
```

Hinweis:

- Das Skript erstellt bei Bedarf `.venv`, prueft Abhaengigkeiten, startet Demo/Meditation/Ceremonial und fuehrt alle Tests aus.

## 7. ATHERIA CLI starten (manuell)

Hilfe anzeigen:

```powershell
python main.py --help
```

Standard-Demo:

```powershell
python main.py demo --duration 3
```

Erwartung:

- JSON-Ausgabe mit Dashboard-Feldern wie:
  - `aggregatform`
  - `purpose_alignment`
  - `morphic_resonance_index`
  - `topological_*`

## 8. Aion-Meditation und Zeremonie

Meditation:

```powershell
python main.py meditation --duration 60 --snapshot morphic_snapshot.json
```

Zeremonielle Aktivierung:

```powershell
python main.py ceremonial --preheat 10 --duration 60 --snapshot morphic_snapshot.json
```

Erwartung:

- Report mit `title: Atheria Transzendenz-Status`
- Je nach Verlauf `morphic_snapshot_path` gesetzt

## 9. DEMO Forge: Erzeugung neuer ausfuehrbarer Kind-Programme

### 9.1 Survival-Profil

```powershell
python DEMO/forge_executable.py --name demo_survival --profile survival --output-dir DEMO/generated_survival --run-harness --harness-iterations 2 --harness-interval 0.0 --manifest DEMO/generated_survival/forge_manifest.json
```

### 9.2 Diagnostic-Profil

```powershell
python DEMO/forge_executable.py --name demo_diag --profile diagnostic --output-dir DEMO/generated_diagnostic --run-harness --harness-iterations 2 --harness-interval 0.0 --manifest DEMO/generated_diagnostic/forge_manifest.json
```

### 9.3 Stress-Test-Profil

```powershell
python DEMO/forge_executable.py --name demo_stress --profile stress-test --output-dir DEMO/generated_stress --run-harness --harness-iterations 2 --harness-interval 0.0 --manifest DEMO/generated_stress/forge_manifest.json
```

Erwartung:

- `harness.passed = true`
- `artifact_integrity.json` vorhanden
- `.sig` Dateien vorhanden
- `forge_manifest.json` pro Profil vorhanden

## 10. Optional: Native EXE erzeugen

Nur falls `pyinstaller` installiert ist:

```powershell
python DEMO/forge_executable.py --name demo_native --profile survival --output-dir DEMO/generated_native --build-exe --run-harness
```

Erwartung:

- Bei erfolgreichem Build liegt EXE unter `DEMO/generated_native/dist/`
- Falls PyInstaller fehlt, bleibt `exe_path` im Manifest `null`

## 11. Population Layer Smoke-Test (HGT, Markt, Inter-Core-Dreaming)

```powershell
python -m unittest -v tests/test_collective_extensions.py
```

Erwartung:

- HGT-Symbiose akzeptiert mindestens einen Mechanismus-Transfer,
- Ather-Credit-Markt fuehrt eine Resource-Rental-Transaktion aus,
- GlobalMorphicNode synchronisiert Sleep-Replay inkl. Trauma-Echo.

## 12. Reproduktions-Integration pruefen

Dieser Test stellt sicher, dass ATHERIA bei Reproduktion automatisch Artefakte erzeugt.

```powershell
python -m unittest -v tests/test_claim_hardening.py
```

Wichtiger Teiltest:

- `test_05_reproduction_emits_executable_artifacts`

Erwartung:

- Log enthaelt Zeilen wie:
  - `Self-Reproduction Artifact | offspring=... | profile=... | validated=True`
- Dashboard-Felder vorhanden:
  - `reproduction_artifact_events`
  - `reproduction_artifact_last_path`
  - `reproduction_artifact_last_integrity_path`

## 13. DEMO-Tests ausfuehren

```powershell
python -m unittest -v tests/test_demo_forge.py
```

Erwartung:

- Alle Tests `OK`
- Enthaelt auch Tamper-Nachweis (Signatur/Integritaet)

## 14. Vollstaendiger End-to-End Testlauf (letzter Test)

Das ist der komplette Abschlusslauf:

```powershell
python -m unittest -v tests/test_digital_life_claims.py tests/test_claim_hardening.py tests/test_demo_forge.py tests/test_collective_extensions.py
```

Erwartung:

- Alle Tests `OK`
- Keine Failures

## 15. Ergebnis-Checkliste

Nach erfolgreichem Lauf sollten folgende Punkte erfuellt sein:

- `main.py demo` liefert valide JSON-Snapshots
- Meditation/Zeremonie laufen durch
- Population Layer ist aktiv (`hgt_*`, `market_*`, `inter_core_dream_*`)
- In `DEMO/generated_*` existieren:
  - `.py`, `.pyz`, Launcher
  - `artifact_integrity.json`
  - `*.sig`
  - `forge_manifest.json`
- Reproduktionsartefakte entstehen unter:
  - `DEMO/lineage/`
- Kompletter Testlauf ist gruen

## 16. Troubleshooting

- `ModuleNotFoundError: torch`
  - `pip install torch`
- Torch-Warnung wegen fehlendem NumPy
  - `pip install numpy`
- `python main.py ...` nicht gefunden
  - Im Projektroot `d:\ATHERIA` ausfuehren
- `exe_path` bleibt `null`
  - `pyinstaller` installieren oder ohne EXE arbeiten
- Langsame/instabile Tests
  - Erneut ausfuehren, venv aktiv, keine parallelen Heavy-Tasks

## 17. Reproduzierbarer Kurzablauf (Copy/Paste)

```powershell
cd d:\ATHERIA
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install numpy torch
python -m py_compile main.py atheria_core.py DEMO/forge_executable.py
python main.py demo --duration 3
python DEMO/forge_executable.py --name demo_survival --profile survival --output-dir DEMO/generated_survival --run-harness --harness-iterations 2 --harness-interval 0.0 --manifest DEMO/generated_survival/forge_manifest.json
python DEMO/forge_executable.py --name demo_diag --profile diagnostic --output-dir DEMO/generated_diagnostic --run-harness --harness-iterations 2 --harness-interval 0.0 --manifest DEMO/generated_diagnostic/forge_manifest.json
python DEMO/forge_executable.py --name demo_stress --profile stress-test --output-dir DEMO/generated_stress --run-harness --harness-iterations 2 --harness-interval 0.0 --manifest DEMO/generated_stress/forge_manifest.json
python -m unittest -v tests/test_digital_life_claims.py tests/test_claim_hardening.py tests/test_demo_forge.py tests/test_collective_extensions.py
```
