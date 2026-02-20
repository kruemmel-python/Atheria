# DEMO Dokumentation

## 1. Zweck

Der Ordner `DEMO` implementiert einen vollstaendigen Programm-Generator:

- Ein Forge-Programm erzeugt ein neues ausfuehrbares Kind-Programm.
- Das Kind wird in mehreren Formaten ausgeliefert (`.py`, `.pyz`, Launcher, optional `.exe`).
- Alle erzeugten Artefakte werden gehasht und signiert.
- Optional prueft ein Harness das Kind automatisch.
- Die Forge-Logik ist als API importierbar und wird von `atheria_core.py` bei Reproduktions-Events genutzt.

## 2. Komponenten

- `DEMO/forge_executable.py`
  - CLI + API zum Erzeugen der Kind-Artefakte
  - Profile: `survival`, `diagnostic`, `stress-test`
  - Integritaet: SHA-256 + HMAC-SHA256
  - Harness: automatische Lauf- und Integritaetspruefung
- `DEMO/__init__.py`
  - Exporte fuer `forge(...)` und `validate_generated_offspring(...)`
- `DEMO/generated_*`
  - Beispielausgaben aus realen Laeufen

## 3. Vollstaendige Feature-Implementierung

### 3.1 Template-Varianten (Profile)

Die Generator-Profile sind nicht nur Label, sondern veraendern Laufzeitverhalten des erzeugten Kind-Programms:

- `survival`
  - stabilitaetsorientierte Puls-Entwicklung
  - checkpoint-basierte Erholung bei Ausreissern
  - Gesundheitsmetrik (`health`) + Recovery-Zaehler
- `diagnostic`
  - diagnostisches JSONL-Logging pro Schritt
  - Plattform-, Python-, PID- und Latenz-Metriken
  - erhoehte Beobachtbarkeit statt maximaler Last
- `stress-test`
  - CPU-Burst pro Schritt (`_stress_burst`)
  - Lastmetriken (`ops`, `ops_per_sec`, `checksum`)
  - aggressivere Dynamik zur Belastungspruefung

### 3.2 Signierung und Hashing

Fuer jedes erzeugte Artefakt:

- SHA-256 Hash
- HMAC-SHA256 Signatur (wenn Signierung aktiv)
- Detached Signature-Datei (`*.sig`)

Zusatzdatei:

- `artifact_integrity.json` mit Algorithmen, Zeitstempel und Metadaten aller Artefakte.

Schluesselquellen:

- `--signing-key <path>` (Datei)
- `--signing-key-env <VAR>`
- automatische Schluesselerzeugung (`forge_signing.key`) falls nichts gesetzt ist

### 3.3 Optionales Test-Harness

Mit `--run-harness` prueft das Tool automatisch:

- Start von Kind-Programm als `.py`
- Start von Kind-Programm als `.pyz`
- Gueltigkeit der State-Datei (`app`, `profile`, `step`, `pulse`)
- Integritaet aller Artefakte gegen Hash/Signatur
- optional Launcher-Ausfuehrung (`--harness-run-launchers`)

Ergebnis liegt im Manifest unter `harness`.

### 3.4 Integration in ATHERIA-Reports

Bei `SelfReproductionEngine.force_reproduction()` wird jetzt automatisch ein Executable-Artefakt fuer den Nachkommen erzeugt:

- Profilwahl dynamisch (`survival`/`diagnostic`/`stress-test`) basierend auf Systemzustand
- Ausgabe in `DEMO/lineage/<offspring-id>`
- Signierung + Integritaet + Harness
- JSONL-Verlauf: `DEMO/lineage/lineage_artifacts.jsonl`

Neue Dashboard-Felder in ATHERIA:

- `reproduction_artifact_events`
- `reproduction_artifact_last_profile`
- `reproduction_artifact_last_path`
- `reproduction_artifact_last_integrity_path`
- `reproduction_artifact_last_validated`
- `reproduction_artifact_last_signature`

## 4. Bedienung

## 4.1 Standard-Lauf

```bash
python DEMO/forge_executable.py --name atheria_child --profile survival --output-dir DEMO/generated --run-harness
```

## 4.2 Alle Profile

```bash
python DEMO/forge_executable.py --name demo_survival --profile survival --output-dir DEMO/generated_survival --run-harness
python DEMO/forge_executable.py --name demo_diag --profile diagnostic --output-dir DEMO/generated_diagnostic --run-harness
python DEMO/forge_executable.py --name demo_stress --profile stress-test --output-dir DEMO/generated_stress --run-harness
```

## 4.3 Signierung konfigurieren

```bash
python DEMO/forge_executable.py --name signed_child --signing-key DEMO/my_signing.key
```

oder per Environment:

```bash
set ATHERIA_DEMO_SIGNING_KEY=base64:QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=
python DEMO/forge_executable.py --name signed_child_env
```

## 4.4 Harness streng machen

```bash
python DEMO/forge_executable.py --name child --run-harness --strict-harness
```

## 4.5 Optional native EXE

```bash
python DEMO/forge_executable.py --name child_native --build-exe
```

Hinweis: braucht `pyinstaller`.

## 5. Ergebnisprotokoll (reale Laeufe)

Die folgenden Ergebnisse stammen aus tatsaechlich ausgefuehrten Commands im Workspace.

### 5.1 Survival-Profil

Pfad:

- `DEMO/generated_survival/forge_manifest.json`

Kernergebnis:

- `profile`: `survival`
- `harness.passed`: `true`
- `signature_algorithm`: `hmac-sha256`
- Artefakte inkl. `.sig` wurden erzeugt

State-Beispiel (harness):

- `profile`: `survival`
- `metrics.health`: vorhanden
- `metrics.recoveries`: vorhanden

### 5.2 Diagnostic-Profil

Pfad:

- `DEMO/generated_diagnostic/forge_manifest.json`

Kernergebnis:

- `profile`: `diagnostic`
- `harness.passed`: `true`
- diagnostische Metriken vorhanden (`diag_log`, `write_latency_ms`)
- Integritaetscheck erfolgreich

### 5.3 Stress-Test-Profil

Pfad:

- `DEMO/generated_stress/forge_manifest.json`

Kernergebnis:

- `profile`: `stress-test`
- `harness.passed`: `true`
- Lastmetriken vorhanden (`ops`, `ops_per_sec`, `checksum`)
- Integritaetscheck erfolgreich

### 5.4 ATHERIA-Reproduktionsintegration

Verifiziert in Hardening-Test:

- Test: `tests/test_claim_hardening.py::test_05_reproduction_emits_executable_artifacts`
- Beobachtete Log-Zeile:
  - `Self-Reproduction Artifact | offspring=... | profile=stress-test | validated=True`
- Assertions bestaetigen:
  - Artifact-Event-Zaehler > 0
  - gueltiger Artifact-Pfad vorhanden
  - gueltiger Integritaets-Pfad vorhanden

## 6. Validierung durch Tests

Direkte DEMO-Tests:

```bash
python -m unittest -v tests/test_demo_forge.py
```

Erwartung:

- Profilgenerierung + Harness fuer alle 3 Profile erfolgreich
- Manipulationsnachweis: Signatur-/Integritaetsfehler wird erkannt

Integrationstest mit ATHERIA:

```bash
python -m unittest -v tests/test_claim_hardening.py
```

## 7. Grenzen

- Native `.exe` ist optional und von `pyinstaller` abhaengig.
- Signierung ist absichtlich HMAC-basiert (symmetrisch), nicht PKI-basiert.
- Die Profile sind funktionale Laufzeitprofile, keine domaintrainierten Modelle.

## 8. Fazit

Alle vier angeforderten Punkte sind vollstaendig implementiert und ausfuehrbar:

1. Profile (`survival`, `diagnostic`, `stress-test`)  
2. Hashing + Signierung der Artefakte  
3. Optionales Harness zur automatischen Validierung  
4. Direkte Integration in ATHERIA-Reproduktionsreports
