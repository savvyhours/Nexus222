"""
pytest configuration for NEXUS-II tests.
Sets asyncio mode, registers marks, and provides shared fixtures.
"""
import asyncio
import pytest


# ── asyncio mode ──────────────────────────────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration (requires live API keys)"
    )


@pytest.fixture(scope="session")
def event_loop():
    """
    Use a single event loop for the entire test session.
    Avoids 'event loop is closed' errors with asyncio.get_event_loop() in tests.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
