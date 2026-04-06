"""
Firebase Authentication middleware for FastAPI.

Verifies Firebase ID tokens from the Authorization header.
Initialize using either FIREBASE_SERVICE_ACCOUNT_JSON env var (raw JSON string)
or defaults to Firebase Application Default Credentials.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import auth as firebase_auth, credentials
from fastapi import Header, HTTPException

from config import settings

logger = logging.getLogger(__name__)

# ── Initialize Firebase Admin SDK ──

_firebase_app: Optional[firebase_admin.App] = None


def _parse_service_account_blob(blob: str) -> Optional[dict]:
    """
    Parse a service account payload that can be either:
    1) pure JSON, or
    2) env-style assignment text like: FIREBASE_SERVICE_ACCOUNT_JSON = { ... }
    """
    if not blob:
        return None

    raw = blob.strip()
    candidates = [raw]

    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(raw[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def _resolve_project_id(sa_dict: Optional[dict] = None) -> Optional[str]:
    explicit = settings.FIREBASE_PROJECT_ID or os.getenv("GOOGLE_CLOUD_PROJECT")
    if explicit:
        return explicit
    if sa_dict and isinstance(sa_dict, dict):
        return sa_dict.get("project_id")
    return None


def _initialize_app_with_cert(cert_source, sa_dict: Optional[dict] = None) -> firebase_admin.App:
    project_id = _resolve_project_id(sa_dict)
    cred = credentials.Certificate(cert_source)
    if project_id:
        logger.info("Initializing Firebase Admin with explicit projectId=%s", project_id)
        return firebase_admin.initialize_app(cred, {"projectId": project_id})
    return firebase_admin.initialize_app(cred)


def _init_firebase():
    global _firebase_app
    if _firebase_app is not None:
        return

    sa_path = settings.FIREBASE_SERVICE_ACCOUNT_PATH
    sa_json = settings.FIREBASE_SERVICE_ACCOUNT_JSON

    logger.info("Initializing Firebase Admin SDK (path=%s, has_json=%s)", sa_path, bool(sa_json))

    # Try path first (if provided), then JSON (if provided), then ADC.
    if sa_path:
        try:
            logger.info("Attempting Firebase init from service account path: %s", sa_path)
            _firebase_app = _initialize_app_with_cert(sa_path)
            logger.info("Firebase Admin initialized from service account path")
            return
        except Exception as e:
            logger.warning("Failed to initialize Firebase Admin from path: %s", e)

            # Fallback: parse non-standard text content from file (env-style assignment).
            try:
                file_blob = Path(sa_path).read_text(encoding="utf-8")
                sa_dict = _parse_service_account_blob(file_blob)
                if sa_dict:
                    logger.info(
                        "Parsed Firebase service account file as text blob (project_id=%s)",
                        sa_dict.get("project_id"),
                    )
                    _firebase_app = _initialize_app_with_cert(sa_dict, sa_dict)
                    logger.info("Firebase Admin initialized from parsed service account file content")
                    return
                logger.warning("Service account file did not contain parseable JSON content")
            except Exception as parse_file_err:
                logger.warning("Failed to parse service account file content: %s", parse_file_err)

    if sa_json and _firebase_app is None:
        try:
            logger.info("Attempting Firebase init from service account JSON (length=%d)", len(sa_json))
            sa_dict = _parse_service_account_blob(sa_json)
            if not sa_dict:
                raise json.JSONDecodeError("Invalid JSON payload", sa_json, 0)
            logger.info("Successfully parsed service account JSON (project_id=%s)", sa_dict.get("project_id"))
            _firebase_app = _initialize_app_with_cert(sa_dict, sa_dict)
            logger.info("Firebase Admin initialized from service account JSON")
            return
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: %s", e)
        except Exception as e:
            logger.warning("Failed to initialize Firebase Admin from JSON: %s", e)

    # Fall back to Application Default Credentials.
    try:
        logger.info("Attempting Firebase init with Application Default Credentials")
        adc_project_id = _resolve_project_id()
        if adc_project_id:
            logger.info("Using ADC with explicit projectId=%s", adc_project_id)
            _firebase_app = firebase_admin.initialize_app(options={"projectId": adc_project_id})
        else:
            _firebase_app = firebase_admin.initialize_app()
        logger.info("Firebase Admin initialized with default credentials")
    except Exception as e:
        logger.warning("Firebase Admin not initialized with ADC: %s", e)


# Initialize on module import
_init_firebase()


# ── FastAPI Dependency ──

async def verify_firebase_token(authorization: str = Header(default="")) -> dict:
    """
    FastAPI dependency that extracts and verifies a Firebase ID token
    from the Authorization header.

    Returns the decoded token dict (contains uid, email, etc.)
    Raises HTTPException(401) on failure.
    """
    if not authorization.startswith("Bearer "):
        logger.warning("Missing or invalid Authorization header")
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Expected: Bearer <token>",
        )

    token = authorization[len("Bearer "):]
    logger.info("Attempting to verify Firebase token (length=%d)", len(token))
    
    if _firebase_app is None:
        logger.error("Firebase Admin SDK not initialized. Check FIREBASE_SERVICE_ACCOUNT_JSON or FIREBASE_SERVICE_ACCOUNT_PATH env vars.")
        raise HTTPException(
            status_code=401,
            detail="Server misconfiguration: Firebase Admin not initialized.",
        )

    try:
        decoded = firebase_auth.verify_id_token(token, app=_firebase_app)
        logger.info("Token verified successfully for user: %s", decoded.get("uid"))
        return decoded
    except firebase_auth.ExpiredIdTokenError:
        logger.warning("Token has expired")
        raise HTTPException(status_code=401, detail="Token has expired. Please sign in again.")
    except firebase_auth.InvalidIdTokenError as e:
        logger.warning("Invalid authentication token: %s", e)
        raise HTTPException(status_code=401, detail="Invalid authentication token.")
    except firebase_auth.RevokedIdTokenError:
        logger.warning("Token has been revoked")
        raise HTTPException(status_code=401, detail="Token has been revoked.")
    except ValueError as e:
        if "project ID is required" in str(e):
            logger.error("Firebase Admin misconfigured: missing project ID", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=(
                    "Server misconfiguration: Firebase project ID is missing. "
                    "Set FIREBASE_PROJECT_ID or GOOGLE_CLOUD_PROJECT."
                ),
            )
        logger.error("Firebase token verification failed with ValueError: %s", e, exc_info=True)
        raise HTTPException(status_code=401, detail="Authentication failed.")
    except Exception as e:
        logger.error("Firebase token verification failed: %s (type: %s)", e, type(e).__name__, exc_info=True)
        raise HTTPException(status_code=401, detail="Authentication failed.")
