# from src.services.pre_clone.url_parser import parse_github_url

# print(parse_github_url("https://github.com/torvalds/linux"))
# parse_github_url("git@github.com:torvalds/linux.git")
# parse_github_url("github.com/torvalds/linux")
# parse_github_url("torvalds/linux")
# parse_github_url("https://github.com/torvalds/linux/tree/master")


# #invalid examples 
# print(parse_github_url("https://evil.com/torvalds/linux"))
# parse_github_url("https://github.com/-bad/repo")
# parse_github_url("https://github.com/about/repo")
# parse_github_url("https://github.com/a/")


# #run from the root folder. python -m tests.pre_clone.url_test

import pytest

from src.services.pre_clone.url_parser import parse_github_url


# =========================================================
# VALID URL TESTS
# =========================================================

@pytest.mark.parametrize(
    ("url", "owner", "repo"),
    [
        (
            "https://github.com/torvalds/linux",
            "torvalds",
            "linux",
        ),
        (
            "https://github.com/torvalds/linux.git",
            "torvalds",
            "linux",
        ),
        (
            "https://github.com/torvalds/linux/",
            "torvalds",
            "linux",
        ),
        (
            "https://github.com/torvalds/linux/tree/master",
            "torvalds",
            "linux",
        ),
        (
            "https://github.com/torvalds/linux/blob/master/README.md",
            "torvalds",
            "linux",
        ),
        (
            "git@github.com:torvalds/linux.git",
            "torvalds",
            "linux",
        ),
        (
            "github.com/torvalds/linux",
            "torvalds",
            "linux",
        ),
        (
            "torvalds/linux",
            "torvalds",
            "linux",
        ),
        (
            "TORVALDS/linux",
            "torvalds",  # normalized to lowercase
            "linux",
        ),
    ],
)
def test_valid_github_urls(url, owner, repo):
    result = parse_github_url(url)

    assert result.is_valid is True
    assert result.owner == owner
    assert result.repo == repo
    assert result.error is None


# =========================================================
# INVALID URL TESTS
# =========================================================

@pytest.mark.parametrize(
    ("url", "expected_error"),
    [
        (
            "",
            "URL is empty",
        ),
        (
            "   ",
            "URL is empty",
        ),
        (
            "https://evil.com/torvalds/linux",
            "Only github.com URLs are supported",
        ),
        (
            "https://github.com/torvalds",
            "Expected format",
        ),
        (
            "https://github.com/-bad/repo",
            "Invalid GitHub owner name",
        ),
        (
            "https://github.com/about/repo",
            "reserved GitHub path",
        ),
        (
            "https://github.com/torvalds/..",
            "Invalid repository name",
        ),
        (
            "not a url",
            "Expected format",
        ),
    ],
)
def test_invalid_github_urls(url, expected_error):
    result = parse_github_url(url)

    assert result.is_valid is False
    assert expected_error in result.error


# =========================================================
# LENGTH VALIDATION
# =========================================================

def test_owner_too_long():
    owner = "a" * 40
    url = f"https://github.com/{owner}/repo"

    result = parse_github_url(url)

    assert result.is_valid is False
    assert "Owner name too long" in result.error


def test_repo_too_long():
    repo = "a" * 101
    url = f"https://github.com/test/{repo}"

    result = parse_github_url(url)

    assert result.is_valid is False
    assert "Repo name too long" in result.error


# =========================================================
# NORMALIZATION TESTS
# =========================================================

def test_git_suffix_removed():
    result = parse_github_url(
        "https://github.com/test/repo.git"
    )

    assert result.is_valid is True
    assert result.repo == "repo"


def test_owner_normalized_to_lowercase():
    result = parse_github_url(
        "https://github.com/Torvalds/linux"
    )

    assert result.is_valid is True
    assert result.owner == "torvalds"


# =========================================================
# SECURITY / EDGE CASE TESTS
# =========================================================

@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/////",
        "https://github.com/",
        "https://github.com",
        "https://github.com/../../etc/passwd",
        "https://github.com/%0a/test",
        "https://github.com/test/%0a",
    ],
)
def test_malformed_urls(url):
    result = parse_github_url(url)

    assert result.is_valid is False


def test_parser_never_raises():
    """
    Ensure parser never crashes on weird input.
    """

    weird_inputs = [
        None,
        123,
        [],
        {},
        object(),
    ]

    for value in weird_inputs:
        result = parse_github_url(str(value))

        assert result.is_valid is False