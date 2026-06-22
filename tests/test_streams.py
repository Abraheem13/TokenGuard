import pytest


@pytest.mark.xfail(reason="streams implemented later", strict=False)
def test_recurrence_stream_importable():
    from tokenguard.streams.recurrence import recurrence_stream
    recurrence_stream(bench=None)
