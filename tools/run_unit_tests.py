"""Portable model-mechanism tests; no raw datasets or historical run caches."""
import importlib.util
import inspect
from pathlib import Path
import unittest

import torch

torch.set_num_threads(2)
root = Path(__file__).resolve().parents[1]
suite = unittest.TestSuite()
modules = ["test_action_units", "test_command_logs", "test_three_signal_ood", "test_duration_subtask",
           "test_change_subtask", "test_streaming_fusion", "test_online_subtask", "test_line_probe"]
for name in modules:
    spec = importlib.util.spec_from_file_location(name, root / "tests" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    for key, test in inspect.getmembers(module, inspect.isfunction):
        if key.startswith("test_"):
            suite.addTest(unittest.FunctionTestCase(test))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
