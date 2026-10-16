import zlib
import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--seed", type=int, default=0, help="Base seed for all rng fictures"
    )


@pytest.fixture
def rng(request):
    base = request.config.getoption("--seed")
    # Derive a per-test seed from the test's unique ID
    per_test = zlib.adler32(request.node.nodeid.encode())
    return np.random.default_rng([base, per_test])
