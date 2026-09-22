"""Server selection must not silently send an operator to the other platform."""

import copy
import shlex
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agilex_control.sites import load_registry, resolve_site


class SitesTest(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry()

    def test_ip_and_id_select_same_server_and_preserve_domain_username(self):
        for site_id in self.registry["sites"]:
            profile = self.registry["sites"][site_id]
            by_id = resolve_site(self.registry, site_id)
            self.assertEqual(by_id, resolve_site(self.registry, profile["ssh"]["host"]))
            self.assertEqual(shlex.split(by_id["ssh_command"]), by_id["ssh_argv"])
            self.assertEqual(
                by_id["ssh_argv"][-2:], [profile["ssh"]["user"], profile["ssh"]["host"]]
            )
        self.assertEqual(
            resolve_site(self.registry, "panfeng38")["ssh"]["user"],
            "SENSETIME\\panfeng1",
        )

    def test_missing_unknown_or_ambiguous_target_never_falls_back(self):
        for selector in [None, "", "  ", "10.169.21.39", "left"]:
            with self.assertRaises(ValueError):
                resolve_site(self.registry, selector)
        duplicate = copy.deepcopy(self.registry)
        duplicate["sites"]["duplicate"] = duplicate["sites"]["panfeng38"].copy()
        with self.assertRaises(ValueError):
            resolve_site(duplicate, "10.169.21.38")

    def test_new_server_does_not_inherit_old_runtime_or_calibration(self):
        registry = copy.deepcopy(self.registry)
        registry["sites"]["panfeng38"]["primitive_deployment"][
            "supported_primitives"
        ] = ["observe"]
        # Exercise an uncalibrated fixture even after the real site is calibrated.
        registry["sites"]["panfeng38"]["calibration"] = None
        old = resolve_site(registry, "agilex56")
        new = resolve_site(registry, "panfeng38")
        self.assertEqual(
            new["primitive_deployment"]["supported_primitives"],
            ["observe"],
        )
        self.assertNotEqual(
            new["primitive_deployment"]["runtime_dir"],
            old["primitive_deployment"]["runtime_dir"],
        )
        self.assertNotEqual(new["primitive_deployment"]["config"], "config/site.json")
        self.assertIsNone(new["calibration"])
        self.assertEqual(old["primitive_deployment"]["config"], "config/site.json")
        # Execution paths must not inherit another site's configuration.
        # Audit references may legitimately cite a shared cross-site comparison.
        execution_paths = [
            new["primitive_deployment"][key]
            for key in ("project", "python", "config", "runtime_dir")
        ]
        execution_paths += [new["native_project"], new["native_python"]]
        for path in execution_paths:
            self.assertNotIn("/home/agilex/", path)
            self.assertNotIn("10.169.21.56", path)

    def test_resolving_and_editing_one_profile_does_not_change_registry(self):
        original = copy.deepcopy(self.registry)
        selected = resolve_site(self.registry, "panfeng38")
        selected["ssh"]["host"] = "changed"
        self.assertEqual(self.registry, original)

    def test_third_server_is_discovered_from_registry_without_code_changes(self):
        registry = copy.deepcopy(self.registry)
        registry["sites"]["future_lab"] = {
            "ssh": {"host": "192.0.2.10", "user": "lab_user", "port": 2222},
            "expected_hostname": "future-lab",
            "reference": "test-only",
            "native_project": "/srv/robot",
            "primitive_deployment": None,
            "validation": {
                "date": "test-only",
                "level": "unverified",
                "detail": "fixture",
            },
            "calibration": None,
        }
        selected = resolve_site(registry, "192.0.2.10")
        self.assertEqual(selected["site_id"], "future_lab")
        self.assertEqual(
            selected["ssh_argv"], ["ssh", "-p", "2222", "-l", "lab_user", "192.0.2.10"]
        )
        self.assertEqual(
            resolve_site(registry, "panfeng38"),
            resolve_site(self.registry, "panfeng38"),
        )


if __name__ == "__main__":
    unittest.main()
