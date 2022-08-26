import autora
import autora.skl.darts as skl_darts
import autora.theorist.darts as theorist_darts


def test_import():
    assert autora is not None

    assert skl_darts.DARTSRegressor is not None
    assert skl_darts.DARTSExecutionMonitor is not None

    assert theorist_darts.Network is not None
    assert theorist_darts.Architect is not None
