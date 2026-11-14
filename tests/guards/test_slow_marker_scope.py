import importlib.util
import pathlib

TESTS = pathlib.Path(__file__).resolve().parent.parent


def _qutecipy_conftest():
    """The conftest of the vendored qutecipy tests, loaded as a module"""
    path = TESTS/"qutecipy"/"conftest.py"
    spec = importlib.util.spec_from_file_location("qutecipy_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Item:
    """The two attributes of a pytest item that the hook uses"""
    def __init__(self, path):
        self.path = path
        self.markers = []
    def add_marker(self, marker):
        self.markers.append(marker)
    def is_slow(self):
        return any(getattr(m, "name", None) == "slow" for m in self.markers)


def test_the_qutecipy_conftest_marks_only_its_own_tests():
    """The hook pytest_collection_modifyitems of a conftest receives every
    item of the session. The qutecipy conftest used to mark all of them
    slow, so that `pytest tests -m "not slow"` deselected the whole suite,
    and a plugin that skips slow tests skipped every test collected next to
    tests/qutecipy"""
    inside = _Item(TESTS/"qutecipy"/"test_tensortrain.py")
    outside = _Item(TESTS/"algebra"/"test_hamiltonian_algebra.py")
    _qutecipy_conftest().pytest_collection_modifyitems([inside, outside])
    assert inside.is_slow()
    assert not outside.is_slow()
