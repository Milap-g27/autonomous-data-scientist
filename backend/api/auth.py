"""
Firebase Authentication middleware for FastAPI.

Verifies Firebase ID tokens from the Authorization header.
Initialize using either FIREBASE_SERVICE_ACCOUNT_JSON env var (raw JSON string)
or defaults to Firebase Application Default Credentials.
"""

import json
import logging
from typing import Optional

import firebase_admin
from firebase_admin import auth as firebase_auth, credentials
from fastapi import Header, HTTPException

from config import settings

logger = logging.getLogger(__name__)

# ── Initialize Firebase Admin SDK ──

_firebase_app: Optional[firebase_admin.App] = None


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
            cred = credentials.Certificate(sa_path)
            _firebase_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin initialized from service account path")
            return
        except Exception as e:
            logger.warning("Failed to initialize Firebase Admin from path: %s", e)

    if sa_json and _firebase_app is None:
        try:
            logger.info("Attempting Firebase init from service account JSON (length=%d)", len(sa_json))
            sa_dict = json.loads(sa_json)
            logger.info("Successfully parsed service account JSON (project_id=%s)", sa_dict.get("project_id"))
            cred = credentials.Certificate(sa_dict)
            _firebase_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin initialized from service account JSON")
            return
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: %s", e)
        except Exception as e:
            logger.warning("Failed to initialize Firebase Admin from JSON: %s", e)

    # Fall back to Application Default Credentials.
    try:
        logger.info("Attempting Firebase init with Application Default Credentials")
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
        decoded = firebase_auth.verify_id_token(token)
        logger.info("Token verified successfully for user: %s", decoded.get("uid"))
        return decoded
    except firebase_auth.ExpiredIdTokenError:
        logger.warning("Token has expired")
        raise HTTPException(status_code=401, detail="Token has expired. Please sign in again.")
    except firebase_auth.InvalidIdTokenError:
        logger.warning("Invalid authentication token: %s", str(e) if 'e' in locals() else "unknown")
        raise HTTPException(status_code=401, detail="Invalid authentication token.")
    except firebase_auth.RevokedIdTokenError:
        logger.warning("Token has been revoked")
        raise HTTPException(status_code=401, detail="Token has been revoked.")
    except Exception as e:
        logger.error("Firebase token verification failed: %s (type: %s)", e, type(e).__name__, exc_info=True)
        raise HTTPException(status_code=401, detail="Authentication failed.")
