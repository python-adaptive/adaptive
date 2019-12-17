import os
import sys

import pytest

try:
    import ipykernel.iostream
    import zmq

    with_notebook_dependencies = True
except ImportError:
    with_notebook_dependencies = False

# XXX: remove when is fixed https://github.com/ipython/ipykernel/issues/468
skip_because_of_bug = os.name == "nt" and sys.version_info[:2] == (3, 8)


@pytest.mark.skipif(
    not with_notebook_dependencies or skip_because_of_bug,
    reason="notebook dependencies are not installed",
)
def test_private_api_used_in_live_info():
    """We are catching all errors in
    adaptive.notebook_integration.should_update
    so if ipykernel changed its API it would happen unnoticed."""
    # XXX: find a potential better solution in
    # https://github.com/ipython/ipykernel/issues/365
    ctx = zmq.Context()
    iopub_socket = ctx.socket(zmq.PUB)
    iopub_thread = ipykernel.iostream.IOPubThread(iopub_socket)
    assert hasattr(iopub_thread, "_events")
