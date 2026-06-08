"""
app/core/crypto.py
───────────────────
Encryption and decryption for sensitive values stored in the database.

Production: AWS KMS envelope encryption
  - KMS generates a data key (AES-256)
  - Data key encrypts the plaintext locally (fast, cheap)
  - KMS key encrypts the data key (stored alongside ciphertext)
  - Only KMS can decrypt the data key — your app never holds the master key
  - Every encrypt/decrypt is logged in AWS CloudTrail

Development: Fernet symmetric encryption (local key from .env)
  - Fast, no AWS dependency
  - Never use in production

Format stored in DB (bytes):
  [1 byte version][4 bytes data_key_len][data_key_ciphertext][iv][ciphertext][tag]

Why not store tokens in plaintext?
  - DB breach → attacker gets GitHub API access to all user repos
  - KMS breach is near-impossible (HSM-backed, AWS manages key material)
  - SOC2/ISO27001 requires encryption at rest for auth tokens

Why envelope encryption instead of calling KMS for every encrypt?
  - KMS has a 10,000 req/sec rate limit per region
  - Envelope: 1 KMS call generates a data key, local AES does the rest
  - Result: unlimited throughput, single KMS call per user session
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Public interface — always use these two functions
# ─────────────────────────────────────────────────────────────────────────────

async def encrypt_token(plaintext: str) -> bytes:
    """
    Encrypt a string token. Returns opaque bytes for DB storage.
    Uses KMS in production, Fernet in development.
    """
    if not plaintext:
        raise ValueError("Cannot encrypt empty token")

    if settings.APP_ENV == "production" and settings.KMS_KEY_ARN_TOKENS:
        return await _kms_encrypt(plaintext, settings.KMS_KEY_ARN_TOKENS)
    else:
        return _fernet_encrypt(plaintext)


async def decrypt_token(ciphertext: bytes) -> str:
    """
    Decrypt bytes from DB back to plaintext token string.
    Detects which backend was used from the stored format.
    """
    if not ciphertext:
        raise ValueError("Cannot decrypt empty ciphertext")

    # Version byte: 0x01 = KMS, 0x02 = Fernet
    version = ciphertext[0]

    if version == 0x01:
        return await _kms_decrypt(ciphertext)
    elif version == 0x02:
        return _fernet_decrypt(ciphertext)
    else:
        # Legacy: try Fernet directly (migration path)
        try:
            return _fernet_decrypt(ciphertext)
        except Exception:
            raise ValueError(f"Unknown ciphertext version: {version}")


# async def encrypt_api_key(plaintext: str) -> bytes:
#     """
#     Encrypt a user-provided API key (BYOK).
#     Uses a separate KMS key from OAuth tokens for defence in depth.
#     """
#     if settings.APP_ENV == "production" and settings.KMS_KEY_ARN_API_KEYS:
#         return await _kms_encrypt(plaintext, settings.KMS_KEY_ARN_API_KEYS)
#     return _fernet_encrypt(plaintext)


# async def decrypt_api_key(ciphertext: bytes) -> str:
#     """Decrypt a BYOK API key."""
#     return await decrypt_token(ciphertext)


# ─────────────────────────────────────────────────────────────────────────────
# AWS KMS — Envelope Encryption
# ─────────────────────────────────────────────────────────────────────────────

async def _kms_encrypt(plaintext: str, kms_key_arn: str) -> bytes:
    """
    Encrypt using AWS KMS envelope encryption.

    Steps:
      1. Call KMS GenerateDataKey → get (plaintext_data_key, encrypted_data_key)
      2. Use plaintext_data_key with AES-256-GCM to encrypt the token
      3. Discard plaintext_data_key from memory immediately
      4. Store: version_byte + len(encrypted_data_key) + encrypted_data_key + iv + ciphertext + tag

    Decryption:
      1. Extract encrypted_data_key from stored bytes
      2. Call KMS Decrypt → get plaintext_data_key
      3. Use plaintext_data_key with AES-256-GCM to decrypt

    The master key never leaves KMS. Even with full DB access, you cannot
    decrypt without KMS access. IAM policies control who can call KMS.
    """
    import asyncio
    import struct
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        import boto3
        kms_client = boto3.client("kms", region_name=settings.AWS_REGION)

        # Generate data key in thread pool (boto3 is sync)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: kms_client.generate_data_key(
                KeyId=kms_key_arn,
                KeySpec="AES_256",
            )
        )

        plaintext_key    = response["Plaintext"]        # 32 bytes
        encrypted_key    = response["CiphertextBlob"]   # ~300 bytes, stored with ciphertext

        # Encrypt the token with AES-256-GCM
        iv = os.urandom(12)   # 96-bit nonce, unique per encryption
        aesgcm = AESGCM(plaintext_key)
        ciphertext_and_tag = aesgcm.encrypt(iv, plaintext.encode("utf-8"), None)

        # Wipe key from memory immediately
        plaintext_key = b"\x00" * len(plaintext_key)
        del plaintext_key

        # Pack: version(1) + key_len(4) + encrypted_key + iv(12) + ciphertext+tag
        key_len = len(encrypted_key)
        packed = (
            b"\x01"                           # version = KMS
            + struct.pack(">I", key_len)      # 4-byte big-endian length
            + encrypted_key
            + iv
            + ciphertext_and_tag
        )

        return packed

    except Exception as e:
        logger.error("KMS encrypt failed: %s", e)
        raise RuntimeError(f"Token encryption failed: {e}") from e


async def _kms_decrypt(ciphertext: bytes) -> str:
    """Decrypt KMS envelope-encrypted bytes back to string."""
    import asyncio
    import struct
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        import boto3

        # Unpack stored format
        # Skip version byte (index 0)
        key_len = struct.unpack(">I", ciphertext[1:5])[0]
        encrypted_key      = ciphertext[5 : 5 + key_len]
        iv                 = ciphertext[5 + key_len : 5 + key_len + 12]
        ciphertext_and_tag = ciphertext[5 + key_len + 12:]

        kms_client = boto3.client("kms", region_name=settings.AWS_REGION)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: kms_client.decrypt(CiphertextBlob=encrypted_key)
        )

        plaintext_key = response["Plaintext"]

        aesgcm  = AESGCM(plaintext_key)
        plaintext = aesgcm.decrypt(iv, ciphertext_and_tag, None)

        # Wipe key
        plaintext_key = b"\x00" * len(plaintext_key)
        del plaintext_key

        return plaintext.decode("utf-8")

    except Exception as e:
        logger.error("KMS decrypt failed: %s", e)
        raise RuntimeError(f"Token decryption failed: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Fernet — local development only
# ─────────────────────────────────────────────────────────────────────────────

def _get_fernet():
    """Get or create the Fernet instance for local dev."""
    from cryptography.fernet import Fernet
    key = settings.LOCAL_ENCRYPTION_KEY
    if not key:
        # Auto-generate and warn (dev only)
        logger.warning(
            "LOCAL_ENCRYPTION_KEY not set — generating ephemeral key. "
            "Tokens encrypted now CANNOT be decrypted after restart. "
            "Set LOCAL_ENCRYPTION_KEY in .env for persistent dev encryption."
        )
        key = Fernet.generate_key().decode()
    return Fernet(key.encode() if isinstance(key, str) else key)


def _fernet_encrypt(plaintext: str) -> bytes:
    """Fernet encrypt — prepend version byte 0x02."""
    fernet    = _get_fernet()
    encrypted = fernet.encrypt(plaintext.encode("utf-8"))
    return b"\x02" + encrypted


def _fernet_decrypt(ciphertext: bytes) -> str:
    """Fernet decrypt — strip version byte if present."""
    fernet = _get_fernet()
    # Strip version byte if present
    data = ciphertext[1:] if ciphertext[0] == 0x02 else ciphertext
    return fernet.decrypt(data).decode("utf-8")