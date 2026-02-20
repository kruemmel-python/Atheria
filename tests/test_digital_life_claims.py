import asyncio
import random
import unittest

import torch

from atheria_core import AtheriaCore


class TestDigitalLifeClaims(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        random.seed(42)
        torch.manual_seed(42)

    async def test_01_autonomy_without_external_input(self) -> None:
        core = AtheriaCore(tick_interval=0.04)
        core.bootstrap_default_mesh()
        core.rhythm.wake_duration = 0.45
        core.rhythm.sleep_duration = 0.45
        core.rhythm.interval = 0.12

        await core.start()
        try:
            await asyncio.sleep(1.4)
            baseline = core.dashboard_snapshot()
            await asyncio.sleep(3.0)
            final = core.dashboard_snapshot()

            self.assertTrue(
                final["aion_cycle_activity"] > 0.0 or final["autocatalytic_activity"] > 0.0,
                msg=f"Expected internal autonomous activity, got {final}",
            )
            self.assertLess(
                final["entropic_index"],
                2.0,
                msg=f"Entropic index drifted too high: {final['entropic_index']}",
            )
            self.assertGreaterEqual(final["topological_core_cells"], baseline["topological_core_cells"])
            self.assertGreaterEqual(final["topological_edges"], baseline["topological_edges"])
            self.assertLessEqual(
                final["mean_hyperbolic_distance"],
                baseline["mean_hyperbolic_distance"] * 1.5 + 0.35,
            )
        finally:
            await core.stop()

    async def test_02_disturbance_and_regeneration(self) -> None:
        core = AtheriaCore(tick_interval=0.04)
        core.bootstrap_default_mesh()
        core.rhythm.wake_duration = 0.45
        core.rhythm.sleep_duration = 0.45
        core.rhythm.interval = 0.12

        await core.start()
        try:
            await asyncio.sleep(1.5)
            pre = core.dashboard_snapshot()

            core_labels = set()
            for cluster in core.topological_logic.clusters.values():
                core_labels.update(cluster["core"])
            excluded = core_labels | {core.aion.singularity_label, core.transcendence.telos.purpose_label}

            non_core_cells = [cell for cell in core.cells.values() if cell.label not in excluded]
            self.assertTrue(non_core_cells, "Need non-core cells for disturbance test.")
            damage_count = max(1, int(len(non_core_cells) * 0.4))
            damaged = random.sample(non_core_cells, k=min(damage_count, len(non_core_cells)))

            for cell in damaged:
                cell.integrity_rate = 0.05
                cell.error_counter = 4
                cell.set_activation(0.0)

            cuttable_edges = []
            for src in non_core_cells:
                for target_label in list(src.connections.keys()):
                    if core.topological_logic.is_edge_protected(src.label, target_label):
                        continue
                    cuttable_edges.append((src, target_label))
            if cuttable_edges:
                cut_count = max(1, int(len(cuttable_edges) * 0.3))
                for src, target_label in random.sample(cuttable_edges, k=min(cut_count, len(cuttable_edges))):
                    src.remove_connection(target_label)

            core.modulators.force_plasma(core.phase_controller, intensity=1.4)
            await asyncio.sleep(4.0)
            post = core.dashboard_snapshot()

            recovered = sum(1 for cell in damaged if cell.integrity_rate > 0.5)
            self.assertGreaterEqual(
                recovered,
                max(1, len(damaged) // 2),
                msg=f"Damaged cells did not recover sufficiently: recovered={recovered}, damaged={len(damaged)}",
            )
            self.assertTrue(
                post["dream_replay_events"] > pre["dream_replay_events"]
                or post.get("healing_events", 0) > pre.get("healing_events", 0),
                msg=f"Expected replay or healing activity after damage. pre={pre} post={post}",
            )
            self.assertGreaterEqual(post["topological_core_cells"], pre["topological_core_cells"])
            self.assertGreaterEqual(post["topological_edges"], pre["topological_edges"])
            self.assertGreaterEqual(post["purpose_alignment"], pre["purpose_alignment"] * 0.72)
            self.assertGreaterEqual(post["structural_tension"], 0.0)
            self.assertLessEqual(post["structural_tension"], 1.0)
        finally:
            await core.stop()

    async def test_03_reproduction_with_variation_and_independence(self) -> None:
        core = AtheriaCore(tick_interval=0.04)
        core.bootstrap_default_mesh()

        await core.start()
        child = None
        try:
            core.modulators.force_plasma(core.phase_controller, intensity=1.3)
            await asyncio.sleep(1.6)

            # Ensure at least one non-baseline blueprint and one mechanism exist for variation checks.
            if len(core.evolution.cell_type_blueprints) <= 1:
                core.evolution._invent_cell_type(pressure=0.9)  # test setup
            if len(core.evolution.runtime_mechanisms) == 0:
                core.evolution._invent_runtime_mechanism(pressure=0.9)  # test setup

            parent_state = core.evolution.export_state()
            child_id = core.force_reproduction()
            self.assertIsNotNone(child_id)
            await asyncio.sleep(1.0)

            self.assertGreaterEqual(core.reproduction.reproduction_events, 1)
            self.assertIn(child_id, core.reproduction.offspring_cores)
            child = core.reproduction.offspring_cores[child_id]
            self.assertTrue(child.running)

            child_state = child.evolution.export_state()
            self.assertNotEqual(
                parent_state,
                child_state,
                msg="Offspring evolution state should differ from parent (variation).",
            )

            await core.stop(shutdown_lineage=False)
            await asyncio.sleep(0.5)
            self.assertTrue(child.running, "Offspring should continue independently after parent stop.")
        finally:
            if core.running:
                await core.stop(shutdown_lineage=True)
            if child is not None and child.running:
                await child.stop()


if __name__ == "__main__":
    unittest.main()
