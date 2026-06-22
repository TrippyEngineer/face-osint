from crowd.persistence import should_persist_reading


def test_persists_on_interval_boundary():
    assert should_persist_reading(now=100.0, last_ts=89.0, risk="safe", prev_risk="safe", interval_s=10) is True


def test_skips_within_interval():
    assert should_persist_reading(now=95.0, last_ts=89.0, risk="safe", prev_risk="safe", interval_s=10) is False


def test_persists_on_risk_transition_even_within_interval():
    assert should_persist_reading(now=95.0, last_ts=89.0, risk="high", prev_risk="safe", interval_s=10) is True


def test_first_reading_persists():
    assert should_persist_reading(now=1.0, last_ts=0.0, risk="safe", prev_risk="safe", interval_s=10) is False
    assert should_persist_reading(now=11.0, last_ts=0.0, risk="safe", prev_risk="safe", interval_s=10) is True
