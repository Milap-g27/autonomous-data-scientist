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

    # Try path first (if provided), then JSON (if provided), then ADC.
    if sa_path:
        try:
            cred = credentials.Certificate(sa_path)
            _firebase_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin initialized from service account path")
            return
        except Exception as e:
            logger.warning("Failed to initialize Firebase Admin from path: %s", e)

    if sa_json and _firebase_app is None:
        try:
            sa_dict = json.loads(sa_json)
            cred = credentials.Certificate(sa_dict)
            _firebase_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin initialized from service account JSON")
            return
        except Exception as e:
            logger.warning("Failed to initialize Firebase Admin from JSON: %s", e)

    # Fall back to Application Default Credentials.
    try:
        _firebase_app = firebase_admin.initialize_app()
        logger.info("Firebase Admin initialized with default credentials")
    except Exception as e:
        logger.warning("Firebase Admin not initialized: %s", e)


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
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Expected: Bearer <token>",
        )

    token = authorization[len("Bearer "):]

    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded
    except firebase_auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Token has expired. Please sign in again.")
    except firebase_auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")
    except firebase_auth.RevokedIdTokenError:
        raise HTTPException(status_code=401, detail="Token has been revoked.")
    except Exception as e:
        logger.error("Firebase token verification failed: %s", e, exc_info=True)
        raise HTTPException(status_code=401, detail="Authentication failed.")
