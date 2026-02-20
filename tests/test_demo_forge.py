import tempfile
import unittest
from pathlib import Path

from DEMO.forge_executable import PROFILE_CHOICES, forge, validate_generated_offspring


class TestDemoForge(unittest.TestCase):
    def test_01_profiles_generate_and_validate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="atheria_demo_forge_") as td:
            root = Path(td)
            for profile in PROFILE_CHOICES:
                out = root / profile
                result = forge(
                    name=f"child_{profile}",
                    output_dir=out,
                    profile=profile,
                    run_harness=True,
                    harness_iterations=2,
                    harness_interval=0.0,
                    harness_timeout_seconds=20.0,
                    sign_artifacts=True,
                    build_exe=False,
                )
                self.assertEqual(result.profile, profile)
                self.assertTrue(Path(result.script_path).exists())
                self.assertTrue(Path(result.pyz_path).exists())
                self.assertTrue(Path(result.integrity_path).exists())
                self.assertIsNotNone(result.harness)
                self.assertTrue(result.harness.passed, msg=f"Harness failed for {profile}: {result.harness.details}")

    def test_02_signing_detects_tamper(self) -> None:
        with tempfile.TemporaryDirectory(prefix="atheria_demo_tamper_") as td:
            out = Path(td) / "tamper_case"
            result = forge(
                name="tamper_child",
                output_dir=out,
                profile="diagnostic",
                run_harness=False,
                sign_artifacts=True,
                build_exe=False,
            )
            key_hex = Path(result.signing_key_path).read_text(encoding="utf-8").strip()
            key = bytes.fromhex(key_hex)

            script = Path(result.script_path)
            script.write_text(script.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

            harness = validate_generated_offspring(
                result,
                iterations=1,
                interval=0.0,
                timeout_seconds=20.0,
                run_launchers=False,
                signing_key=key,
            )
            self.assertFalse(harness.checks["integrity_ok"])
            self.assertFalse(harness.passed)


if __name__ == "__main__":
    unittest.main()

