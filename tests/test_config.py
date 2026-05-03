from src.core.config import get_settings

def test_app_name():
    s = get_settings()
    assert s.APP_NAME == "repo-ingestion"

def test_github_required():
    s = get_settings()
    assert s.GITHUB_APP_ID is not None

import os

def test_github_required():
    # This will print exactly what value is causing the crash
    print(f"\nDEBUG VALUE: {os.environ.get('GITHUB_APP_ID')}")
    s = get_settings()
    assert s.GITHUB_APP_ID is not None
