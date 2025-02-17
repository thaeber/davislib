from pathlib import Path
import pytest


@pytest.fixture(scope='function')
def data_path():
    path = Path(__file__).parent / 'data'
    return path.absolute()
