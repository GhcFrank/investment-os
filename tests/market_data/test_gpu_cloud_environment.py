import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import env_utils


class GPUCloudEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_file = Path(self.temp_dir.name) / ".env"
        self.env_file.write_text(
            "VAST_API_KEY=dotenv-test-value\n",
            encoding="utf-8",
        )
        self.original_cwd = Path.cwd()

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()

    def _load_from_cwd(self, cwd: Path) -> str:
        os.chdir(cwd)
        return env_utils.get_project_environment_value("VAST_API_KEY")

    def test_project_env_path_is_independent_of_root_or_src_cwd(self):
        project_root = Path(__file__).resolve().parents[2]
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(env_utils, "PROJECT_ENV_FILE", self.env_file),
        ):
            from_root = self._load_from_cwd(project_root)
            os.environ.pop("VAST_API_KEY", None)
            from_src = self._load_from_cwd(project_root / "src")
        self.assertEqual(from_root, "dotenv-test-value")
        self.assertEqual(from_src, "dotenv-test-value")

    def test_explicit_environment_value_has_priority_over_dotenv(self):
        with (
            patch.dict(
                os.environ,
                {"VAST_API_KEY": "explicit-test-value"},
                clear=True,
            ),
            patch.object(env_utils, "PROJECT_ENV_FILE", self.env_file),
        ):
            value = env_utils.get_project_environment_value("VAST_API_KEY")
        self.assertEqual(value, "explicit-test-value")

    def test_missing_value_returns_empty_without_printing_or_guessing(self):
        missing_env_file = Path(self.temp_dir.name) / "missing.env"
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(env_utils, "PROJECT_ENV_FILE", missing_env_file),
        ):
            value = env_utils.get_project_environment_value("VAST_API_KEY")
        self.assertEqual(value, "")


if __name__ == "__main__":
    unittest.main()
