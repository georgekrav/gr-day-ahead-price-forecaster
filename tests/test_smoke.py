"""Package import smoke test."""

import gr_epf


def test_package_imports() -> None:
    assert gr_epf.__version__
