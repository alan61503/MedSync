import pytest
from unittest import mock
from backend.services.dxa_service import run_dxa_bmd

MOCK_CLI_OUTPUT = """{
  "bmd": 0.79,
  "t_score": -2.8,
  "z_score": -1.5,
  "fracture_assessment": "none"
}"""

@mock.patch('backend.services.dxa_service.subprocess.run')
def test_run_dxa_bmd(mock_run):
    mock_proc = mock.Mock()
    mock_proc.stdout = MOCK_CLI_OUTPUT
    mock_proc.returncode = 0
    mock_proc.check_returncode = mock.Mock()
    mock_run.return_value = mock_proc

    result = run_dxa_bmd('dummy.dcm')
    assert result['bmd'] == 0.79
    assert result['t_score'] == -2.8
    assert result['risk_level'] == 'High Risk (Osteoporosis)'

def test_run_dxa_bmd_fallback():
    # Calling run_dxa_bmd without bonexpert installed should return fallback result without raising errors
    result = run_dxa_bmd('nonexistent_dummy.dcm')
    assert 'bmd' in result
    assert 't_score' in result
    assert result['disease'] == 'Osteoporosis'
    assert 'risk_level' in result

