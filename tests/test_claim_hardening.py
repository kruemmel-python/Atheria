import asyncio
import random
import unittest
from pathlib import Path

import torch

from atheria_core import AtheriaCore


class TestClaimHardening(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        random.seed(7)
        torch.manual_seed(7)

    async def test_01_open_evolution_generates_programs(self) -> None:
        core = AtheriaCore(tick_interval=0.04)
        core.bootstrap_default_mesh()
        core.aion_meditation_mode = True
        try:
            for _ in range(60):
                core.evolution.step()
            mechanisms = core.evolution.runtime_mechanisms
            self.assertGreaterEqual(len(mechanisms), 2)

            signatures = set()
            operators = set()
            for mech in mechanisms.values():
                program = mech.get("program", [])
                self.assertTrue(program, "Generated runtime mechanism must contain a program.")
                signatures.add(str(mech.get("program_signature", "")))
                for term in program:
                    operators.add(str(term.get("op", "")))

            self.assertGreaterEqual(len(signatures), 2, "Expected multiple structurally distinct generated programs.")
            self.assertGreaterEqual(len(operators), 3, "Expected operator diversity in generated programs.")
        finally:
            core.aion_meditation_mode = False

    async def test_02_reproduction_includes_selection_trials(self) -> None:
        core = AtheriaCore(tick_interval=0.04)
        core.bootstrap_default_mesh()
        core.reproduction.selection_trials_per_reproduction = 1
        core.reproduction.selection_trial_seconds = 0.25
        core.reproduction.selection_margin = 0.0

        await core.start()
        try:
            core.modulators.force_plasma(core.phase_controller, intensity=1.2)
            await asyncio.sleep(1.2)
            child_id = core.force_reproduction()
            self.assertIsNotNone(child_id)

            deadline = asyncio.get_running_loop().time() + 6.0
            while core.reproduction.selection_total_trials < 1 and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.1)

            self.assertGreaterEqual(core.reproduction.selection_total_trials, 1)
            self.assertGreaterEqual(
                core.reproduction.selection_child_wins + core.reproduction.selection_parent_wins,
                1,
            )
            self.assertNotAlmostEqual(
                core.reproduction.last_parent_fitness,
                core.reproduction.last_child_fitness,
                delta=1e-8,
            )
        finally:
            await core.stop(shutdown_lineage=True)

    async def test_03_telos_is_homeostatic_not_static_constant(self) -> None:
        core = AtheriaCore(tick_interval=0.04)
        core.bootstrap_default_mesh()
        await core.start()
        try:
            await asyncio.sleep(1.0)
            purpose = core.transcendence.telos.ensure_purpose_node()
            initial_target = float(getattr(purpose, "homeostatic_temperature", 34.0))

            core.modulators.force_plasma(core.phase_controller, intensity=1.6)
            await asyncio.sleep(1.5)
            high_target = float(getattr(purpose, "homeostatic_temperature", 34.0))

            core.modulators.stabilize(core.phase_controller, intensity=1.4)
            await asyncio.sleep(1.2)
            settled_target = float(getattr(purpose, "homeostatic_temperature", 34.0))

            self.assertGreater(abs(high_target - initial_target), 0.3)
            self.assertGreater(abs(settled_target - 34.0), 0.15)
            self.assertGreaterEqual(core.transcendence.last_purpose_alignment, 0.0)
            self.assertLessEqual(core.transcendence.last_purpose_alignment, 1.0)
        finally:
            await core.stop(shutdown_lineage=True)

    async def test_04_ecodynamics_drives_selection_scarcity_and_gradient(self) -> None:
        core = AtheriaCore(tick_interval=0.04)
        core.bootstrap_default_mesh()
        core.assembler.resource_pool = 1.0

        await core.start()
        try:
            await asyncio.sleep(1.0)
            baseline = core.dashboard_snapshot()

            core.modulators.force_plasma(core.phase_controller, intensity=1.25)
            for i in range(6):
                core.feed_raw_material(category=f"Stress_{i}", relevance=1.1)
            await asyncio.sleep(2.2)
            post = core.dashboard_snapshot()

            self.assertGreaterEqual(post["selection_pressure"], 0.0)
            self.assertLessEqual(post["selection_pressure"], 1.0)
            self.assertGreaterEqual(post["resource_scarcity"], 0.0)
            self.assertLessEqual(post["resource_scarcity"], 1.0)
            self.assertGreaterEqual(post["ecological_complexity"], 0.1)
            self.assertLessEqual(post["ecological_complexity"], 1.0)

            self.assertTrue(
                abs(post["resource_scarcity"] - baseline["resource_scarcity"]) > 0.01
                or abs(post["selection_pressure"] - baseline["selection_pressure"]) > 0.01
                or abs(post["ecological_complexity"] - baseline["ecological_complexity"]) > 0.005
                or abs(post["fitness_gradient"] - baseline["fitness_gradient"]) > 0.005
            )
            self.assertNotEqual(core.reproduction.reproduction_threshold_offset, 0.0)
            self.assertAlmostEqual(core.evolution.external_selection_pressure, post["selection_pressure"], places=3)

            self.assertTrue(
                any(key.startswith("EcoChallenge_") for key in core.assembler.concentrations.keys())
                or any(label.startswith("EcoChallenge_") for label in core.cells.keys())
            )
        finally:
            await core.stop(shutdown_lineage=True)

    async def test_05_reproduction_emits_executable_artifacts(self) -> None:
        core = AtheriaCore(tick_interval=0.04)
        core.bootstrap_default_mesh()
        core.reproduction.selection_trials_per_reproduction = 1
        core.reproduction.selection_trial_seconds = 0.2
        core.reproduction.artifact_run_harness = True

        await core.start()
        try:
            core.modulators.force_plasma(core.phase_controller, intensity=1.25)
            await asyncio.sleep(1.0)
            child_id = core.force_reproduction()
            self.assertIsNotNone(child_id)

            deadline = asyncio.get_running_loop().time() + 12.0
            while core.reproduction.artifact_events < 1 and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.1)

            self.assertGreaterEqual(core.reproduction.artifact_events, 1)
            snap = core.dashboard_snapshot()
            self.assertGreaterEqual(snap["reproduction_artifact_events"], 1)
            self.assertIn(snap["reproduction_artifact_last_profile"], ["survival", "diagnostic", "stress-test"])
            self.assertTrue(bool(snap["reproduction_artifact_last_path"]))
            self.assertTrue(Path(str(snap["reproduction_artifact_last_path"])).exists())
            self.assertTrue(bool(snap["reproduction_artifact_last_integrity_path"]))
            self.assertTrue(Path(str(snap["reproduction_artifact_last_integrity_path"])).exists())
            self.assertIn("reproduction_artifact_last_validated", snap)
        finally:
            await core.stop(shutdown_lineage=True)


if __name__ == "__main__":
    unittest.main()
