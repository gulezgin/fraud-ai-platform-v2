import pytest

from fraud_platform.config import get_settings


@pytest.fixture(scope="session")
def settings():
    return get_settings()
