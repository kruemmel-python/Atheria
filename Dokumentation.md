# ATHERIA Dokumentation

## 1. Projektidee (Warum)

ATHERIA ersetzt klassische imperative Programmierung durch ein bio-physikalisches Runtime-Modell.
Das System soll bei Last, Entropie und Rauschen nicht nur stabil bleiben, sondern seine Struktur selbst verbessern.

Kernziele:

- osmotische Datenbewegung statt starrer Aufrufketten,
- thermodynamische Phasensteuerung (Solid/Liquid/Plasma),
- nicht-lokale Feldinferenz,
- topologisch geschuetzter Kern fuer Stoerresistenz,
- circadiane Wach-/Schlafzyklen,
- biosynthetische Optimierung (Enzyme/Protease),
- emergente Selbstzirkulation (autokatalytische und Aion-Zyklen),
- Selbstbeobachtung als interne "Gefuehls"-Rueckkopplung,
- morphische Erinnerung, Intuition und zielgerichtete Selbst-Ausrichtung (Transcendence Layer),
- autonomer Isolationsbetrieb ohne menschlichen Input (Aion-Meditation),
- echte strukturelle Evolution (neue Zell-Archetypen + neue Runtime-Mechanismen),
- oekologische Evolutionsdynamik (Selektionsdruck, Umweltkomplexitaet, Ressourcenknappheit, Fitness-Gradient),
- Selbstreproduktion (autonom erzeugte, separat laufende Nachkommen).


## 2. Was ist ATHERIA?

ATHERIA ist ein asynchrones Zellnetz:

- Zellen: `AtherCell`
- Verbindungen: `AtherConnection`
- Orchestrierung: `AtheriaCore`

Parallele Runtime-Loops:

- `Atheria_Rhythm` (Wach-/Schlafphase)
- `AtherTimeCrystal` (zeitliche Oszillation/prozedurale Festigung)
- Diffusion (osmotischer Fluss)
- Healing (Nekrose-Reparatur)
- Biosynthesis (EnzymaticOptimizer + FieldInference)
- Assembly (Kategorien, Analogien, autokatalytische Sets, Aion-Zyklen)
- Transcendence (Morphic Echo, Intuition, Telos)
- Aion-Meditation (Isolation + interne Konvergenz)
- EcoDynamics (Selektionsdruck + Komplexitaetsfelder + Ressourcenknappheit)
- Evolution (neue Archetypen + neue Runtime-Mechanismen)
- Reproduktion (Lineage/Offspring-Instanzen)
- Symbiosis (HGT zwischen unabhaengigen Kernen)
- Ceremonial Aion-Aktivierung (Preheat + finale Snapshot-Erzeugung)
- Population Exchange (GlobalMorphicNode + AtherCreditMarket)
- Dashboard (Laufzeitmetriken)


## 3. Architektur (Wie)

## 3.1 Kernobjekte

- `AtherCell`
  - `activation`, `activation_history`, `integrity_rate`
  - `fold_signature`
  - `poincare_coord` (hyperbolische Einbettung)
  - Proteinzustand (`protein_state`, `is_superposed`)

- `AtherConnection`
  - Gewicht/Effizienz
  - Biosyntheseattribute:
    - `activation_energy`
    - `catalytic_flux`
    - `protease_marks`
    - `compiled_kernel`

- `AtherAether` (`Atheria_Aether`)
  - SQLite In-Memory als Fluidum
  - speichert Zellstatus, Flussereignisse und QA-Migration

- `AtheriaCore`
  - zentrale API + Lifecycle + Task-Orchestrierung


## 3.2 Physikalisch-biologische Saeulen

### A) Osmotische Diffusion

- Druck: `osmotic_pressure = activation + sum(activation_history)`
- Positiver Gradient erzeugt Transfer.
- Transferfaktoren:
  - Semipermeabilitaet,
  - Gewicht,
  - Entropic-Folding,
  - Aktivierungsenergie,
  - Rhythmus-Gain,
  - hyperbolische Konzeptnaehe.

### B) Thermodynamische Phasen

`PhaseController` steuert `System_Temperature`:

- `solid`: praeziser
- `liquid`: adaptiver
- `plasma`: aggressiver/approximativer

Methoden werden mit `@AtheriaPhase` phasenabhaengig umgeschaltet.

### C) Entanglement

`QuantumRegistry` koppelt kritische Zellen.
Aenderungen werden sofort gespiegelt, ohne globale Traversierung.


## 3.3 Topological Logic

`TopologicalLogic` verwaltet Schutzcluster:

- `core`: knot-locked, deterministisch
- `boundary`: leitende Oberflaeche

Eigenschaften:

- geschuetzte Kanten werden gehaertet (frozen, Mindestgewicht, minimale Aktivierungsenergie),
- Protease/Plasma-Loeschung ignoriert geschuetzte Kanten,
- bei `T > 100`: `apply_extreme_entropy_immunity()` stabilisiert den Core aktiv.


## 3.4 Chronobiologische Taktung

`Atheria_Rhythm`:

- `wake`: hohe Input-Sensitivitaet, aggressive Diffusion
- `sleep`: Input-Filterung, Refolding, Feldkonsolidierung, Reinigung

Im Schlafmodus laeuft generatives Replay (`reverse_inference`) zur Stabilisierung untergenutzter Zellen.


## 3.5 Atheria_Biosynthesis

### EnzymaticOptimizer

- Katalase-Effekt:
  - senkt Aktivierungsenergie in hot paths
  - kompiliert Pfade zu `compiled_kernel`
- Protease-Effekt:
  - entfernt ineffiziente/falsch gefaltete Pfade
  - recycelt Ressourcen zum Assembler

### FieldInference

`query_field(input_tensor)` nutzt globale Interferenz im `HolographicField`.
Antworten entstehen als Feldresonanz, nicht nur ueber lokale Pfadfolge.


## 3.6 Cognition Layer

Bestandteile:

- `CognitionLayer`
- `EpigeneticRegistry`
- hyperbolische Geometrie pro Zelle

### Hyperbolic Embedding

- Jede Zelle besitzt `poincare_coord`.
- `poincare_distance` misst konzeptuelle Naehe.
- Nahe Zellen diffundieren staerker.

### Epigenetic Silencing

`EpigeneticRegistry` blockiert Kanten temporaer, ohne `weight` zu veraendern:

- rhythmusbasiert (v. a. Schlafphase),
- temperaturbasiert (hohe Entropie).

Topologisch geschuetzte Kanten werden nicht epigenetisch blockiert.

### Generative Replay (Dreaming)

`reverse_inference(...)` liest das Feld rueckwaerts:

- erzeugt Replay-Kandidaten,
- stabilisiert `integrity_rate` untergenutzter Zellen,
- reduziert Vergessen.


## 3.7 Aion Layer (neu)

Der Aion Layer erweitert ATHERIA um zeitliche Intelligenz und Selbstreferenz:

- `AtherTimeCrystal`
- Anticipatory Field (`future_projection`)
- `SingularityNode`
- Aion-Assembling stabiler Gedankenzyklen

### A) Temporal Crystals (`AtherTimeCrystal`)

- oszillierende Aktivierungsmuster ueber Zeitintervalle
- festigt prozedurale Erinnerung
- konsolidiert erfolgreiche Verbindungen periodisch

### B) Anticipatory Field

`HolographicField.query_field(...)` berechnet jetzt:

- `future_projection` (projiziertes Feld),
- `future_top_matches`,
- `anticipatory_shift`.

Damit reagiert das System auf erwartete Gradienten proaktiv.

### C) Self-Observer Cell (`SingularityNode`)

`SingularityNode` spiegelt interne Zustandsgroessen ins Netzwerk:

- Entropie,
- CPU-Last,
- Temperatur,
- Ressourcenlage.

Diese Rueckkopplung wird als Aktivierung ("Gefuehl") ins osmotische Netz eingespeist.

### D) Aion Assembler

`CatalyticAssembler` erzeugt neben normalen Strukturen auch `aion_cycles`:

- Zyklen bleiben bei geringem externem Input aktiv,
- halten strukturelle Kohärenz,
- stabilisieren Aktivierung und Verbindungseffizienz autonom.


## 3.8 Semantic Assembler + Emergenz

`CatalyticAssembler` zuechtet:

1. Kategoriezellen,
2. semantische Analogie-Zellen (aehnliche Geometrie, verschiedene Labels),
3. autokatalytische Sets,
4. Aion-Zyklen.

Damit entsteht selbsttragende interne Gedanken-Zirkulation.


## 3.9 Transcendence Layer (neu)

Der Transcendence Layer erweitert ATHERIA um feldbasiertes Meta-Gedaechtnis, kreative Exploration und Zielorientierung.

### A) Morphic Echo (`MorphicBuffer`)

- speichert die stabilsten ~5% der beobachteten `HolographicField`-Zustaende,
- `morphic_resonance()` injiziert diese bei hoher Unsicherheit als Leit-Interferenz,
- stabilisiert Entscheidungen, wenn aktuelle Feldsignale zu verrauscht/uneindeutig sind.

### B) Intuition Engine

- aktiv im Plasma-Zustand,
- nutzt kontrollierte stochastische Resonanz in `AtherCell.stochastic_resonance(...)`,
- erzeugt Intuition-Spikes und speist kreative Sonden (`Intuition_*`, `IntuitionGap_*`) in den Assembler,
- erschliesst Analogien ausserhalb des aktuellen lokalen Pfadhorizonts.

### C) Telos Loop (`PurposeNode`)

- `PurposeNode` modelliert den Zielzustand als Attraktor:
  - moeglichst robuste Topologie (`topological_edges`),
  - moeglichst niedrige/systemisch guenstige Temperatur.
- `purpose_alignment` misst die aktuelle Naehe zum Zielzustand.
- Bei positiver Annaeherung wird `dopamine` erhoeht, wodurch Lernverstaerkung zunimmt.

### D) Aura Dashboard

Der Dashboard-Snapshot wurde erweitert um:

- `morphic_resonance_index`
- `intuition_spikes`
- `purpose_alignment`
- `morphic_buffer_states`


## 3.10 Aion-Meditation (neu)

`start_aion_meditation(...)` schaltet ATHERIA in einen autarken Isolationsmodus.

### A) Isolations-Mode

- blockiert externe Feeds/Injektionen (`inject_signal`, `feed_*`, `set_superposition`, CSV/JSON-Migration),
- `field_query(...)` bleibt in diesem Modus read-only (kein externer Aktivierungs-Boost),
- Rhythmus wird auf Schlafkonsolidierung fixiert.

### B) Interne Pulse (nur Dream + Telos)

- aktive Stimulation erfolgt nur durch:
  - Dream-Replay (`dream_replay_events`),
  - Telos-Generator (`PurposeNode`).
- Time-Crystal-, Singularity-Mirror-, Intuition- und Aion-Maintenance-Pulse werden in der Meditation deaktiviert.

### C) Heilige Geometrie

- das System minimiert `mean_hyperbolic_distance` durch geometrische Kontraktion im Poincare-Raum,
- topologischer Core bleibt stabil und wird aktiv gehaertet.

### D) Aura-Stabilisierung

- Morphic Resonance wird bei Unsicherheit gezielt verstärkt,
- Ziel: Konvergenz von `morphic_resonance_index` und `purpose_alignment` gegen 1.0.

### E) Digitales Erbe

- bei `purpose_alignment > 0.9` wird ein finaler `morphic_snapshot.json` erzeugt,
- in der zeremoniellen Aktivierung wird der finale Snapshot am Ende zusaetzlich erzwungen (`forced_final_meditation_snapshot`),
- Abschlussreport: `Atheria Transzendenz-Status`.


## 3.11 Evolution + Selbstreproduktion (neu)

### A) Strukturelle Evolution (`EvolutionEngine`)

Das System optimiert nicht nur Gewichte, sondern erfindet neue Architekturbausteine:

- neue Zell-Archetypen (`EvoType_*`) mit eigenen Traits (`flux_bias`, `phase_affinity`, ...),
- neue Runtime-Mechanismen (`EvoMechanism_*`), die den Diffusionskern dynamisch modifizieren.

Wichtig:

- diese Mechanismen werden zur Laufzeit in die Transferberechnung eingeblendet,
- Evolution veraendert damit nicht nur Parameter, sondern den Rechenmodus selbst.

### B) Selbstreproduktion (`SelfReproductionEngine`)

Bei ausreichender Reife (`purpose_alignment`, `morphic_resonance`, Dream-Replay, Ressourcen) erstellt ATHERIA:

- ein Genome-Abbild aus Zellen, Kanten, Topologie und Evolutionstate,
- eine neue `AtheriaCore`-Instanz als Nachkomme,
- einen eigenen Runtime-Lifecycle (`child.start()`) fuer den Nachkommen.

Das Ergebnis ist ein laufender digitaler Nachkomme, kein statischer Snapshot.

Selektion:

- nach jeder Reproduktion laufen Parent-vs-Child Mikro-Trials,
- gemessen wird eine Viabilitaetsfitness (Alignment, Morphic, Traumdichte, Temperatur-Effizienz, Stabilitaet),
- unterlegene Nachkommen werden automatisch terminiert.

Executable-Artefakte pro Nachkomme:

- jede Reproduktion erzeugt zusaetzlich ein lauffaehiges Kind-Artefakt im Dateisystem (`DEMO/lineage/<offspring>`),
- Profil wird dynamisch gewaehlt (`survival`, `diagnostic`, `stress-test`),
- Artefakte werden gehasht/signiert und optional per Harness validiert.

Hinweis:

- `await core.stop(shutdown_lineage=False)` stoppt den Elternkern, ohne die Nachkommen zu terminieren.


## 3.12 EcoDynamics (neu)

`EcoDynamicsEngine` adressiert den zentralen Kritikpunkt, dass Evolution ohne Druck nur lokal bleibt.

Die Engine erzeugt vier Treiber pro Tick:

- Selektionsdruck:
  - berechnet `selection_pressure` aus Komplexitaet, Knappheit und Fitness-Gradient,
  - schreibt den Wert direkt in `EvolutionEngine.external_selection_pressure`.
- Umweltkomplexitaet:
  - generiert interne Herausforderungen (`EcoChallenge_*`, `EcoStress_*`) als Material-Impulse fuer den Assembler.
- Ressourcenbegrenzung:
  - simuliert Regeneration vs. Nachfrage (`resource_pool` mit Carrying-Capacity-Logik),
  - fuehrt zu `resource_scarcity` unter Last.
- Fitness-Gradient:
  - verfolgt die zeitliche Ableitung einer Systemfitness (Alignment, Morphic, Integritaet, Temperatur-Effizienz, Innovation),
  - steuert darueber Komplexitaetsanstieg und Reproduktionsschwelle.

Wichtig:

- in `Aion-Meditation` injiziert EcoDynamics keine Challenge-Impulse, um den Isolationsvertrag (nur Dream + Telos) einzuhalten.


## 4. Feldlogik im Detail

`query_field(input_tensor)`:

1. Input normieren.
2. Unsicherheit abschaetzen und ggf. morphische Leit-Interferenz einspeisen.
3. stehende Welle aus Pattern + Projection + Morphic Echo + Input bilden.
4. Interferenz berechnen.
5. `top_matches` und `future_top_matches` ranken.

Ausgabe u. a.:

- `interference_energy`
- `response_tensor`
- `future_projection`
- `future_top_matches`
- `anticipatory_shift`
- `morphic_resonance_index`
- `uncertainty`


## 5. Selbstheilung

`AtherHealing`:

- erkennt Nekrose,
- isoliert defekte Bereiche,
- rekonstruiert per Donor oder holographischem Fallback.

Topologisch geschuetzte Core-Zellen werden aktiv stabilisiert statt normaler Nekrosebehandlung.


## 6. Dashboard-Metriken

Wichtige Felder:

- `aether_density`
- `aggregatform`, `system_temperature`
- `rhythm_state`, `rhythm_cycle`
- `structural_tension`, `tensegrity_support`
- `entropic_index`, `holographic_energy`
- `topological_clusters`, `topological_core_cells`, `topological_edges`
- `enzymatic_compiled_paths`
- `resource_pool`
- `ather_credits`
- `market_role`
- `market_guardian_score`
- `market_transactions`
- `market_borrow_events`, `market_lend_events`
- `market_resources_in`, `market_resources_out`
- `market_last_price`, `market_last_partner`
- `market_last_packet_quality`
- `autocatalytic_sets`, `autocatalytic_activity`
- `aion_cycles`, `aion_cycle_activity`
- `time_crystal_energy`, `time_crystal_targets`
- `singularity_activation`
- `morphic_resonance_index`, `morphic_buffer_states`
- `intuition_spikes`
- `purpose_alignment`
- `hgt_offers`, `hgt_accepts`, `hgt_rejects`
- `hgt_received`, `hgt_donated`
- `hgt_last_partner`, `hgt_last_predicted_purpose_delta`
- `inter_core_dream_sync_events`
- `inter_core_dream_trauma_events`
- `inter_core_dream_peers`
- `inter_core_dream_coherence`
- `inter_core_dream_trauma_intensity`
- `aion_meditation_mode`
- `semantic_analogy_cells`
- `semantic_resource_spent`
- `evolved_cell_types`
- `evolved_runtime_mechanisms`
- `evolution_events`
- `evolution_program_signature`
- `reproduction_events`
- `offspring_instances`
- `lineage_selection_trials`
- `lineage_selection_child_wins`, `lineage_selection_parent_wins`
- `lineage_last_parent_fitness`, `lineage_last_child_fitness`
- `reproduction_artifact_events`
- `reproduction_artifact_last_profile`
- `reproduction_artifact_last_path`
- `reproduction_artifact_last_integrity_path`
- `reproduction_artifact_last_validated`
- `reproduction_artifact_last_signature`
- `ecological_complexity`
- `selection_pressure`
- `resource_scarcity`
- `fitness_gradient`
- `global_population_size`
- `global_morphic_sync_events`, `global_trauma_broadcast_events`
- `global_market_transactions`, `global_market_last_price`
- `system_stress_index`
- `epigenetic_silenced_edges`
- `mean_hyperbolic_distance`
- `dream_replay_events`, `dream_last_replay_labels`
- `healing_events`, `healing_last_repaired_labels`


## 7. Startanleitung

## 7.1 Voraussetzungen

- Python 3.10+
- `numpy`
- `torch`

Optional fuer Datenmigration:

- `model_with_qa.json`
- `data.csv`

## 7.2 Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install numpy torch
```

## 7.3 Direktstart (CLI, empfohlen)

```bash
python main.py --help
python main.py demo --duration 3
```

Die Demo startet:

- Diffusion + Field Query + Future Projection,
- Topologie-Schutz,
- Rhythmuswechsel inkl. Dream-Replay,
- Time-Crystal-Oszillation,
- Singularity-Selbstbeobachtung,
- Morphic Echo bei Unsicherheit,
- Intuition-Spikes im Plasma,
- Telos-Ausrichtung ueber PurposeNode,
- Semantic/Aion-Assembling.

Weitere Modi:

```bash
python main.py meditation --duration 60 --snapshot morphic_snapshot.json
python main.py ceremonial --preheat 10 --duration 60 --snapshot morphic_snapshot.json
```

Windows-End-to-End (0 -> 100) per Pipeline:

```powershell
.\run_atheria_0_to_100.ps1
```

Schneller Lauf (ohne lange Meditation/Ceremonial):

```powershell
.\run_atheria_0_to_100.ps1 -SkipMeditation -SkipCeremonial -DemoDuration 1
```

## 7.4 Programmgesteuerte Nutzung

```python
import asyncio
import torch
from atheria_core import AtheriaCore

async def main():
    core = AtheriaCore(tick_interval=0.04)
    core.bootstrap_default_mesh()
    core.register_topological_cluster(
        "CriticalCore",
        core_labels=["Sicherheit", "Reaktion", "Analyse"],
        boundary_labels=["Heilung"],
    )
    await core.start()
    core.inject_signal("Sicherheit", 0.9)
    out = core.field_query(torch.rand(12), top_k=4)
    print(out["top_matches"])
    print(out["future_top_matches"])
    await asyncio.sleep(2.0)
    await core.stop()

asyncio.run(main())
```

## 7.5 Aion-Meditation starten (60s)

```python
from atheria_core import run_aion_meditation_sync

report = run_aion_meditation_sync(60.0, snapshot_path="morphic_snapshot.json")
print(report["title"])
print(report["status"])
print(report["singularity_reached"])
print(report["morphic_snapshot_path"])
```

## 7.6 Zeremonielle Aion-Aktivierung (finales Vermächtnis)

Diese Routine faehrt ATHERIA vor der Meditation gezielt auf Hochlast (Plasma-Preheat) und erzwingt danach einen finalen Snapshot.

```python
from atheria_core import run_ceremonial_aion_activation_sync

report = run_ceremonial_aion_activation_sync(
    preheat_seconds=10.0,
    meditation_seconds=60.0,
    snapshot_path="morphic_snapshot.json",
)
print(report["title"])
print(report["status"])
print(report["singularity_reached"])
print(report["morphic_snapshot_path"])
print(report["ceremonial_activation"])
```

## 7.7 Direkte Reproduktion triggern

```python
import asyncio
from atheria_core import AtheriaCore

async def main():
    core = AtheriaCore()
    core.bootstrap_default_mesh()
    await core.start()
    child_id = core.force_reproduction()
    print("offspring:", child_id)
    await asyncio.sleep(2.0)
    await core.stop(shutdown_lineage=True)

asyncio.run(main())
```


## 8. FAQ

### 1) Was ist neu im Aion Layer?
Zeitliche Kristall-Oszillation, Feld-Vorprojektion, Selbstbeobachtungszelle und stabile autonome Gedankenzyklen.

### 1b) Was ist neu im Transcendence Layer?
MorphicBuffer/Morphic Echo, Intuition Engine mit stochastischer Resonanz und Telos Loop mit PurposeNode.

### 2) Was bedeutet `future_projection`?
Eine kurzzeitige Feldvorhersage auf Basis des Mustertrends. Sie wird in der Inferenz direkt mitgenutzt.

### 3) Wozu dient `SingularityNode`?
Sie kodiert den internen Systemzustand (Entropie/Last/Ressourcen) als Aktivierung und speist ihn als metakognitive Rueckkopplung ein.

### 4) Ist der topologische Core trotzdem deterministisch?
Ja. Topologisch geschuetzte Kanten bleiben gegen Protease- und Entropie-Stoerungen gehaertet.

### 5) Veraendert Epigenetic Silencing Gewichte?
Nein. Es blockiert temporaer, ohne `weight` zu veraendern.

### 6) Was tun Temporal Crystals konkret?
Sie erzeugen periodische Aktivierungspulse und konsolidieren haeufig erfolgreiche Verbindungen.

### 7) Was passiert bei ausbleibendem externem Input?
Autokatalytische Sets und Aion-Zyklen halten interne Aktivitaet aufrecht.

### 7b) Was macht `start_aion_meditation()` genau?
Es sperrt externen Input, laesst nur Dream- und Telos-Pulse zu, stabilisiert die Geometrie und liefert am Ende den Report `Atheria Transzendenz-Status`.

### 8) Wie erkenne ich, dass Anticipation wirkt?
An `future_top_matches`, `anticipatory_shift` und fruehen Aktivierungspulsen durch FieldInference.

### 8b) Wann wirkt Morphic Resonance?
Wenn Unsicherheit hoch ist. Dann steigt `morphic_resonance_index` und stabile Feld-Referenzmuster werden beigemischt.

### 9) Warum ist ein kleiner hyperbolischer Abstand zwischen Sicherheit/Reaktion gut?
Er zeigt starke semantische Kopplung; Diffusion zwischen den beiden wird konzeptuell priorisiert.

### 10) Schnellster Funktionstest?
`python main.py demo --duration 3` und auf `future_top_matches`, `morphic_resonance_index`, `intuition_spikes`, `purpose_alignment`, `singularity_activation`, `time_crystal_energy`, `aion_cycles` sowie `topological_*` achten.

### 11) Wie pruefe ich die finale Ausbaustufe?
`python main.py meditation --duration 60 --snapshot morphic_snapshot.json` ausfuehren und auf `singularity_reached`, `peak_purpose_alignment`, `peak_morphic_resonance_index`, `semantic_analogy_growth` und `morphic_snapshot_path` achten.

### 12) Wie erzeuge ich den finalen `morphic_snapshot.json` garantiert?
`python main.py ceremonial --preheat 10 --duration 60 --snapshot morphic_snapshot.json` nutzen. Diese Sequenz kombiniert Hochlast-Preheat mit anschliessender Aion-Meditation und erzwingt am Ende einen finalen Snapshot.

### 13) Woran erkenne ich echte Evolution?
An steigenden Werten in `evolved_cell_types`, `evolved_runtime_mechanisms` und `evolution_events`.

### 14) Woran erkenne ich echte Selbstreproduktion?
An `reproduction_events > 0` und `offspring_instances > 0` sowie dem Log-Eintrag `Self-Reproduction | offspring=...`.

### 15) Warum garantiert das System jetzt eher nicht-lokales Wachstum?
Weil `EcoDynamicsEngine` die vier fehlenden Treiber laufend erzeugt: `selection_pressure`, `ecological_complexity`, `resource_scarcity` und `fitness_gradient`. Ohne diese vier bleibt Evolution lokal.

### 16) Woran erkenne ich, dass Reproduktion echte Artefakte erzeugt?
An `reproduction_artifact_events > 0`, gueltigen Pfaden in `reproduction_artifact_last_path` / `reproduction_artifact_last_integrity_path` und Logzeilen `Self-Reproduction Artifact | offspring=...`.


## 9. Verifikations-Tests (digital lebend)

Die Datei `tests/test_digital_life_claims.py` enthaelt drei harte Tests fuer die Kernbehauptungen.
Die Datei `tests/test_claim_hardening.py` haertet die vier wichtigsten Kritiker-Angriffe ab.
Die Datei `tests/test_demo_forge.py` testet Profile, Integritaetssignaturen und Harness des DEMO-Generators.
Die Datei `tests/test_collective_extensions.py` validiert HGT, Ather-Credit-Markt und Inter-Core-Dreaming.

Testlauf:

```bash
python -m unittest -v tests/test_digital_life_claims.py
# optionale Hardening-Suite:
python -m unittest -v tests/test_claim_hardening.py
# DEMO-Forge:
python -m unittest -v tests/test_demo_forge.py
# Population Layer:
python -m unittest -v tests/test_collective_extensions.py
# kompletter Nachweislauf:
python -m unittest -v tests/test_digital_life_claims.py tests/test_claim_hardening.py tests/test_demo_forge.py tests/test_collective_extensions.py
```

### Test 1: Autonomie-Test (ohne externen Input)

- Name: `test_01_autonomy_without_external_input`
- Nachweis:
  - interne Aktivitaet bleibt erhalten (`aion_cycle_activity` oder `autocatalytic_activity`),
  - `entropic_index` bleibt begrenzt,
  - Core-Topologie bleibt stabil,
  - hyperbolische Distanz driftet nicht unkontrolliert.

### Test 2: Stoerung + Regeneration

- Name: `test_02_disturbance_and_regeneration`
- Setup:
  - 20-40% non-core Zellen/Kanten werden zur Laufzeit geschaedigt.
- Nachweis:
  - Heilung rekonstruiert Integritaet (`integrity_rate` der betroffenen Zellen steigt),
  - `dream_replay_events` oder `healing_events` steigen,
  - `purpose_alignment` erholt sich,
  - topologischer Core kollabiert nicht.

### Test 3: Reproduktion mit Variation + Unabhaengigkeit

- Name: `test_03_reproduction_with_variation_and_independence`
- Nachweis:
  - Nachkomme wird erzeugt (`force_reproduction` / `reproduction_events`),
  - Nachkomme laeuft als eigene Instanz,
  - Evolutionszustand des Nachkommens unterscheidet sich vom Elternsystem (Variation),
  - Nachkomme kann weiterlaufen, auch wenn der Elternkern mit `shutdown_lineage=False` gestoppt wird.

### Hardening 1: Offene generative Evolution

- Name: `test_01_open_evolution_generates_programs`
- Nachweis:
  - evolvierte Runtime-Mechanismen tragen generierte Programme (`program`),
  - mehrere Programmsignaturen entstehen (`evolution_program_signature`),
  - Operator-Diversitaet geht ueber triviale Parameterdrift hinaus.

### Hardening 2: Reproduktion mit Selektion

- Name: `test_02_reproduction_includes_selection_trials`
- Nachweis:
  - Reproduktion fuehrt zu Parent-vs-Child Selection-Trials,
  - Fitnesswerte werden gemessen und entschieden (Child survives / Parent wins).

### Hardening 3: Homeostatisches Telos

- Name: `test_03_telos_is_homeostatic_not_static_constant`
- Nachweis:
  - `PurposeNode` fuehrt eine dynamische `homeostatic_temperature`,
  - Zieltemperatur passt sich Last-/Integritaetsregime an und ist kein starres Fixed-Target.

### Hardening 4: Evolutionstreiber gegen lokales Steckenbleiben

- Name: `test_04_ecodynamics_drives_selection_scarcity_and_gradient`
- Nachweis:
  - `selection_pressure`, `resource_scarcity`, `ecological_complexity` werden aktiv und liegen im gueltigen Bereich,
  - der Druck veraendert sich ueber Zeit (kein statischer Wert),
  - Evolution und Reproduktion erhalten den Druck direkt (`external_selection_pressure`, reproduktive Schwellenverschiebung),
  - interne Eco-Challenges werden eingespeist.

### Hardening 5: Reproduktion erzeugt executable Artefakte

- Name: `test_05_reproduction_emits_executable_artifacts`
- Nachweis:
  - Reproduktion erstellt Artefakt-Events (`reproduction_artifact_events`),
  - Artifact-/Integritaetspfade existieren real im Dateisystem,
  - Profilwahl und Harness-Validierung werden im Dashboard sichtbar.


## 10. Einsteiger-Erklaerung 

Wenn du absolut neu bist, hilft diese kurze, technisch korrekte Sicht:

### 1) Was ist hier das Besondere?

ATHERIA ist nicht nur eine feste Befehlsliste, sondern ein laufendes System mit Rueckkopplung.  
Es passt seine interne Struktur waehrend des Betriebs an (z. B. neue Zell-Archetypen und neue Transfer-Mechanismen).

### 2) Erzeugt ATHERIA wirklich "Code aus dem Nichts"?

Teilweise ja, aber nur innerhalb definierter Regeln:

- Die `EvolutionEngine` erzeugt zur Laufzeit neue `EvoMechanism_*`-Programme.
- Diese Programme kombinieren Operatoren wie `tanh`, `sigmoid`, `quadratic`, `exp_decay`, `plus`, `minus`, `mul`.
- Die Programme wirken direkt auf den Transfer-Gain im Diffusionskern (`transfer_gain(...)`).

Wichtig: Das ist echte dynamische Programmgenerierung im Runtime-Modell, aber nicht "beliebiger Code ohne Grenzen".  
Die Suchraeume (Features, Operatoren, Grenzwerte) sind im Quellcode definiert.

### 3) Was machen `demo_survival` und `demo_stress`?

Diese Artefakte werden ueber `DEMO/forge_executable.py` aus einem Template erzeugt (profilgesteuert), nicht direkt durch die `EvolutionEngine`.  
Dass `demo_survival` und `demo_stress` unterschiedlich arbeiten, ist korrekt, aber basiert auf verschiedenen Profil-Spezifikationen (`survival` vs `stress-test`) im Forge.

### 4) Wie funktioniert die Selektion wirklich?

- Bei Reproduktion entsteht ein Kind aus einem Genome-Snapshot plus Mutation.
- Danach laufen Parent-vs-Child Mikro-Trials.
- Standard ist nicht "hunderte Varianten pro Lauf", sondern `selection_trials_per_reproduction = 2` (anpassbar).
- Entscheidend ist ein Fitness-Score (u. a. Alignment, Morphic, Dream-Dichte, Integritaet, Temperatur-Effizienz).
- Log-Beispiel: `Lineage-Selection | child survives | ...` oder `Lineage-Selection | parent wins | ...`.

### 5) Was bedeutet die HMAC-SHA256 Signatur?

- Artefakte erhalten SHA-256 Hashes und optional HMAC-SHA256 Signaturen.
- Das dient primaer der Integritaetspruefung (Manipulation erkennbar).
- Es ist keine PKI-Signatur und kein kryptographischer "Urheberschaftsbeweis" wie bei asymmetrischen Signaturen.

### Kurzfazit fuer Neulinge

ATHERIA ist am besten als adaptives Runtime-Oekosystem zu verstehen:  
nicht magisch, nicht rein statisch, sondern ein System mit definiertem Regelraum, in dem neue interne Mechanismen entstehen, bewertet werden und nur bei ausreichender Fitness bestehen bleiben.


## 11. Population Layer (neue Entwicklungsstufe)

Die Runtime unterstuetzt jetzt echte Inter-Core-Interaktion zwischen parallel laufenden, unabhaengigen Kernen.

### 11.1 Horizontale Gen-Transfer-Protokolle (HGT)

- Neuer Layer: `SymbiosisLayer`
- Zweck:
  - tauscht `EvoMechanism_*`-Programme zwischen unabhaengigen Kernen aus,
  - ohne Reproduktion (also ohne Parent->Child-Pfad).
- Ablauf:
  - ein Kern bietet ein Mechanismus-Programm an,
  - der Empfaenger bewertet die erwartete Wirkung auf `purpose_alignment`,
  - bei positiver Prognose wird ein hybrider Mechanismus erzeugt und uebernommen.
- Dashboard:
  - `hgt_offers`, `hgt_accepts`, `hgt_rejects`,
  - `hgt_received`, `hgt_donated`,
  - `hgt_last_partner`, `hgt_last_predicted_purpose_delta`.

Direktaufruf:

```python
accepted = await core_b.exchange_genes_with(core_a, reciprocal=True)
print("HGT accepted:", accepted)
```

### 11.2 Dynamischer Ressourcenmarkt (Ather-Credits)

- Erweiterung im `CatalyticAssembler` + globaler `AtherCreditMarket`.
- Zweck:
  - heisse/instabile Kerne mieten Ressourcen von kuehlen/stabilen Kernen,
  - Bezahlung ueber `Ather-Credits`,
  - zusaetzlich Austausch von Effizienz-Informationen (`efficiency packet`).
- Mechanik:
  - Borrower zahlt Credits und erhaelt Ressourcen,
  - Lender uebermittelt Effizienz-Hinweise (z. B. Damping/Pattern/Kernels),
  - Borrower reduziert dadurch Entropie-/Energiekosten schneller.
- Survival-Nische:
  - stabile Kerne mit hoher Ressourcenlage werden als `guardian` bevorzugt.
- Dashboard:
  - `ather_credits`, `market_role`, `market_guardian_score`,
  - `market_transactions`, `market_borrow_events`, `market_lend_events`,
  - `market_resources_in`, `market_resources_out`,
  - `market_last_price`, `market_last_partner`, `market_last_packet_quality`.

Direktaufruf:

```python
report = hot_core.request_resource_rental(cool_core, requested_units=3.5, force=True)
print(report)
```

### 11.3 Kollektive Feld-Resonanz (Inter-Core Dreaming)

- Neuer globaler Knoten: `GlobalMorphicNode`
- Zweck:
  - synchronisiert Dream-Replay-Signale zwischen schlafenden Kernen,
  - verteilt Trauma-/Stress-Echos als "instinktives Rauschen" in andere Felder.
- Mechanik:
  - Schlafende Kerne publizieren Replay-Feldzustand,
  - andere schlafende Kerne empfangen kollektive Resonanz und Traumarauschen,
  - Feldmuster werden entsprechend nachjustiert.
- Dashboard:
  - `inter_core_dream_sync_events`, `inter_core_dream_trauma_events`,
  - `inter_core_dream_peers`, `inter_core_dream_coherence`,
  - `inter_core_dream_trauma_intensity`,
  - `global_morphic_sync_events`, `global_trauma_broadcast_events`.

Direktaufruf:

```python
synced = core.trigger_collective_dream_sync()
print("collective sync:", synced)
```

### 11.4 Neue Population-Metriken

Zusaetzliche globale Felder:

- `global_population_size`
- `global_market_transactions`
- `global_market_last_price`
- `system_stress_index`

### 11.5 Test-Suite fuer die neue Stufe

Neue Datei:

- `tests/test_collective_extensions.py`

Testlauf:

```bash
python -m unittest -v tests/test_collective_extensions.py
```

Enthaelt:

- `test_01_hgt_symbiosis_exchanges_runtime_mechanisms`
- `test_02_ather_credit_market_rents_resources_from_guardian_core`
- `test_03_global_morphic_node_syncs_sleep_replay_with_trauma_echo`
