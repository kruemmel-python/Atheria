import asyncio
import random
import unittest

import torch

from atheria_core import AtheriaCore, GLOBAL_MORPHIC_NODE, RhythmState


class TestCollectiveExtensions(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        random.seed(123)
        torch.manual_seed(123)

    async def test_01_hgt_symbiosis_exchanges_runtime_mechanisms(self) -> None:
        donor = AtheriaCore(tick_interval=0.04)
        receiver = AtheriaCore(tick_interval=0.04)
        donor.bootstrap_default_mesh()
        receiver.bootstrap_default_mesh()

        donor.aion_meditation_mode = True
        try:
            for _ in range(45):
                donor.evolution.step()
        finally:
            donor.aion_meditation_mode = False

        receiver.symbiosis.acceptance_margin = -0.002

        await donor.start()
        await receiver.start()
        try:
            accepted = False
            for _ in range(6):
                accepted = await receiver.exchange_genes_with(donor, reciprocal=False)
                if accepted:
                    break
                donor.evolution._invent_runtime_mechanism(pressure=0.95)
                await asyncio.sleep(0.05)

            self.assertTrue(accepted, "Expected at least one accepted HGT exchange.")
            self.assertGreaterEqual(receiver.symbiosis.hgt_accepts, 1)
            self.assertGreaterEqual(len(receiver.evolution.runtime_mechanisms), 1)
            self.assertTrue(
                any(
                    str(meta.get("source_core_id", "")) == donor.core_id
                    for meta in receiver.evolution.runtime_mechanisms.values()
                )
            )
        finally:
            await receiver.stop(shutdown_lineage=True)
            await donor.stop(shutdown_lineage=True)

    async def test_02_ather_credit_market_rents_resources_from_guardian_core(self) -> None:
        guardian = AtheriaCore(tick_interval=0.04)
        borrower = AtheriaCore(tick_interval=0.04)
        guardian.bootstrap_default_mesh()
        borrower.bootstrap_default_mesh()

        await guardian.start()
        await borrower.start()
        try:
            guardian.assembler.resource_pool = 48.0
            guardian.phase_controller.system_temperature = 33.0
            guardian.ecology.resource_scarcity = 0.02
            guardian.transcendence.last_purpose_alignment = 0.78

            borrower.assembler.resource_pool = 1.2
            borrower.assembler.credit_balance = 120.0
            borrower.phase_controller.system_temperature = 96.0
            borrower.ecology.resource_scarcity = 0.84
            borrower.transcendence.last_purpose_alignment = 0.22

            borrower_before_pool = borrower.assembler.resource_pool
            borrower_before_credit = borrower.assembler.credit_balance
            guardian_before_pool = guardian.assembler.resource_pool
            guardian_before_credit = guardian.assembler.credit_balance

            report = borrower.request_resource_rental(guardian, requested_units=4.0, force=True)
            self.assertIsNotNone(report, "Expected a resource-market transaction.")

            self.assertGreater(borrower.assembler.resource_pool, borrower_before_pool)
            self.assertLess(guardian.assembler.resource_pool, guardian_before_pool)
            self.assertLess(borrower.assembler.credit_balance, borrower_before_credit)
            self.assertGreater(guardian.assembler.credit_balance, guardian_before_credit)
            self.assertGreaterEqual(borrower.assembler.market_transactions, 1)
            self.assertGreaterEqual(guardian.assembler.market_transactions, 1)
            self.assertGreater(borrower.assembler.market_last_packet_quality, 0.0)
        finally:
            await borrower.stop(shutdown_lineage=True)
            await guardian.stop(shutdown_lineage=True)

    async def test_03_global_morphic_node_syncs_sleep_replay_with_trauma_echo(self) -> None:
        trauma_core = AtheriaCore(tick_interval=0.04)
        learner_core = AtheriaCore(tick_interval=0.04)
        trauma_core.bootstrap_default_mesh()
        learner_core.bootstrap_default_mesh()

        await trauma_core.start()
        await learner_core.start()
        try:
            trauma_core.rhythm.state = RhythmState.SLEEP
            learner_core.rhythm.state = RhythmState.SLEEP
            trauma_core.rhythm.wake_duration = 10.0
            trauma_core.rhythm.sleep_duration = 10.0
            learner_core.rhythm.wake_duration = 10.0
            learner_core.rhythm.sleep_duration = 10.0

            trauma_core.phase_controller.system_temperature = 112.0
            trauma_core.ecology.resource_scarcity = 0.88
            for idx in range(4):
                trauma_core.phase_controller.spike_local_entropy(f"trauma_{idx}", magnitude=26.0)
            for cell in trauma_core.cells.values():
                cell.error_counter += 3
                cell.integrity_rate = max(0.15, cell.integrity_rate * 0.7)

            GLOBAL_MORPHIC_NODE.publish_sleep_dream(
                core=trauma_core,
                replay_labels=["TraumaReplay"],
                replay_strength=0.95,
            )
            GLOBAL_MORPHIC_NODE.publish_trauma_if_relevant(trauma_core)

            before_pattern = learner_core.holographic_field.pattern.detach().clone()
            sync_happened = learner_core.trigger_collective_dream_sync()
            after_pattern = learner_core.holographic_field.pattern.detach().clone()

            delta = float(torch.norm(after_pattern - before_pattern, p=2))
            self.assertTrue(sync_happened, "Expected collective dream sync event.")
            self.assertGreater(learner_core.rhythm.inter_core_dream_sync_events, 0)
            self.assertGreater(delta, 1e-6)
            self.assertGreaterEqual(learner_core.rhythm.last_inter_core_peer_count, 1)
            self.assertGreater(
                learner_core.rhythm.last_inter_core_trauma_intensity,
                0.0,
                "Expected trauma echo to influence learner core.",
            )
        finally:
            await learner_core.stop(shutdown_lineage=True)
            await trauma_core.stop(shutdown_lineage=True)


if __name__ == "__main__":
    unittest.main()

