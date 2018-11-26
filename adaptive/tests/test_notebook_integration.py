# -*- coding: utf-8 -*-

import ipykernel.iostream
import zmq


def test_private_api_used_in_live_info():
    """We are catching all errors in
    adaptive.notebook_integration.should_update
    so if ipykernel changed its API it would happen unnoticed."""
    # XXX: find a potential better solution in 
    # https://github.com/ipython/ipykernel/issues/365
    ctx = zmq.Context()
    iopub_socket = ctx.socket(zmq.PUB)
    iopub_thread = ipykernel.iostream.IOPubThread(iopub_socket)
    assert hasattr(iopub_thread, '_events')
