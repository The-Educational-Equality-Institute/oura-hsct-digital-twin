#!/usr/bin/env python3
"""
Oura Ring OAuth2 Setup Utility

Performs the OAuth2 authorization code flow to obtain access and refresh tokens
for the Oura API. Stores tokens in the repo root .env file for use by
import_oura.py.

PATs (Personal Access Tokens) were deprecated end of 2025. OAuth2 is now required.

Prerequisites:
    1. Register an app at https://cloud.ouraring.com (developer portal)
    2. Set redirect URI to http://localhost:4421/callback
    3. Add OURA_CLIENT_ID and OURA_CLIENT_SECRET to the parent .env

Usage:
    python api/oura_oauth2_setup.py              # First-time setup (opens browser)
    python api/oura_oauth2_setup.py --refresh    # Refresh expired access token
    python api/oura_oauth2_setup.py --status      # Check token status
"""

from __future__ import annotations

import argparse
import http.server
import os
import re
import secrets
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load the parent project's .env first; repo-local .env is a fallback only.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
LOCAL_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)  # parent project .env is authoritative
load_dotenv(LOCAL_ENV_PATH)  # optional repo-local fallback for missing vars

# Oura OAuth2 endpoints
AUTH_URL = "https://cloud.ouraring.com/oauth/authorize"
TOKEN_URL = "https://api.ouraring.com/oauth/token"
# Issuer identifier returned in callback (RFC 9207 — Authorization Server Issuer Identification)
EXPECTED_ISSUER = "https://moi.ouraring.com/oauth/v2/ext/oauth-anonymous"

# Local callback server (using approved dev server port)
REDIRECT_HOST = "localhost"
REDIRECT_PORT = 4421
REDIRECT_URI = f"http://{REDIRECT_HOST}:{REDIRECT_PORT}/callback"

# All available scopes
SCOPES = "email personal daily heartrate workout tag session spo2"


class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth2 authorization code from the callback."""

    auth_code: str | None = None
    error: str | None = None
    expected_state: str | None = None
    received_state: str | None = None
    received_issuer: str | None = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            OAuthCallbackHandler.received_state = params.get("state", [None])[0]
            OAuthCallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Authorization successful!</h2>"
                b"<p>You can close this window and return to the terminal.</p>"
                b"</body></html>"
            )
        elif "error" in params:
            OAuthCallbackHandler.error = params.get("error_description", params["error"])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<html><body><h2>Authorization failed</h2><p>{OAuthCallbackHandler.error}</p></body></html>".encode()
            )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress server logs


def update_env_var(key: str, value: str) -> None:
    """Update or add a variable in the parent project's .env file."""
    env_text = ENV_PATH.read_text() if ENV_PATH.exists() else ""

    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    new_line = f"{key}={value}"

    if pattern.search(env_text):
        env_text = pattern.sub(new_line, env_text)
    else:
        if env_text and not env_text.endswith("\n"):
            env_text += "\n"
        env_text += new_line + "\n"

    ENV_PATH.write_text(env_text)
    # Restrict .env permissions to owner-only (F43)
    os.chmod(ENV_PATH, 0o600)


def authorize() -> None:
    """Run the full OAuth2 authorization code flow."""
    client_id = os.getenv("OURA_CLIENT_ID")
    client_secret = os.getenv("OURA_CLIENT_SECRET")

    if not client_id or not client_secret:
        print(f"Error: OURA_CLIENT_ID and OURA_CLIENT_SECRET must be set in {ENV_PATH}")
        print()
        print("Steps to set up:")
        print("  1. Go to https://cloud.ouraring.com and sign in")
        print("  2. Navigate to 'My Applications' in the developer portal")
        print("  3. Create a new application:")
        print(f"     - Redirect URI: {REDIRECT_URI}")
        print("     - App name: Oura Digital Twin")
        print("  4. Copy the Client ID and Client Secret")
        print(f"  5. Add to {ENV_PATH}:")
        print("     OURA_CLIENT_ID=your_client_id")
        print("     OURA_CLIENT_SECRET=your_client_secret")
        sys.exit(1)

    # Build authorization URL with random state for CSRF protection (F42)
    oauth_state = secrets.token_urlsafe(32)
    OAuthCallbackHandler.expected_state = oauth_state
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": oauth_state,
    }
    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    # Start local server to receive callback
    server = http.server.HTTPServer((REDIRECT_HOST, REDIRECT_PORT), OAuthCallbackHandler)
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    print("Opening browser for Oura authorization...")
    print(f"If the browser doesn't open, visit:\n{auth_url}\n")
    webbrowser.open(auth_url)

    # Wait for callback
    server_thread.join(timeout=120)
    server.server_close()

    if OAuthCallbackHandler.error:
        print(f"Authorization failed: {OAuthCallbackHandler.error}")
        sys.exit(1)

    if not OAuthCallbackHandler.auth_code:
        print("Timed out waiting for authorization callback (120s)")
        sys.exit(1)

    # Verify state to prevent CSRF (F42)
    if OAuthCallbackHandler.received_state != oauth_state:
        print("Error: OAuth state mismatch — possible CSRF attack")
        print(f"  Expected: {oauth_state[:8]}...")
        print(f"  Received: {(OAuthCallbackHandler.received_state or 'None')[:8]}...")
        sys.exit(1)

    print("Received authorization code, exchanging for tokens...")

    # Exchange code for tokens
    resp = requests.post(TOKEN_URL, data={
        "grant_type": "authorization_code",
        "code": OAuthCallbackHandler.auth_code,
        "redirect_uri": REDIRECT_URI,
        "client_id": client_id,
        "client_secret": client_secret,
    }, timeout=30)

    if resp.status_code != 200:
        print(f"Token exchange failed ({resp.status_code}): {resp.text}")
        sys.exit(1)

    tokens = resp.json()
    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token", "")

    # Save tokens to .env
    update_env_var("OURA_ACCESS_TOKEN", access_token)
    if refresh_token:
        update_env_var("OURA_REFRESH_TOKEN", refresh_token)

    print("OAuth2 setup complete!")
    print(f"  Access token saved to {ENV_PATH} ({len(access_token)} chars)")
    if refresh_token:
        print(f"  Refresh token saved to {ENV_PATH} ({len(refresh_token)} chars)")
    print(f"  Scopes: {SCOPES}")
    print("\nimport_oura.py will now use OAuth2 automatically (falls back to PAT if needed)")


def refresh() -> None:
    """Refresh an expired access token using the refresh token."""
    client_id = os.getenv("OURA_CLIENT_ID")
    client_secret = os.getenv("OURA_CLIENT_SECRET")
    refresh_token = os.getenv("OURA_REFRESH_TOKEN")

    if not all([client_id, client_secret, refresh_token]):
        print("Error: OURA_CLIENT_ID, OURA_CLIENT_SECRET, and OURA_REFRESH_TOKEN required")
        print("Run without --refresh first to complete initial setup")
        sys.exit(1)

    resp = requests.post(TOKEN_URL, data={
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }, timeout=30)

    if resp.status_code != 200:
        print(f"Token refresh failed ({resp.status_code}): {resp.text}")
        print("You may need to re-authorize: python api/oura_oauth2_setup.py")
        sys.exit(1)

    tokens = resp.json()
    access_token = tokens["access_token"]
    new_refresh = tokens.get("refresh_token", "")

    update_env_var("OURA_ACCESS_TOKEN", access_token)
    if new_refresh:
        update_env_var("OURA_REFRESH_TOKEN", new_refresh)

    print("Token refreshed successfully!")
    print(f"  New access token saved ({len(access_token)} chars)")


def check_status() -> None:
    """Check current token status."""
    pat = os.getenv("OURA_PAT")
    access_token = os.getenv("OURA_ACCESS_TOKEN")
    refresh_token = os.getenv("OURA_REFRESH_TOKEN")
    client_id = os.getenv("OURA_CLIENT_ID")

    print("Oura API Authentication Status:")
    print(f"  OAuth2 Client ID:     {'configured' if client_id else 'NOT SET'}")
    print(f"  OAuth2 Access Token:  {'configured' if access_token else 'NOT SET'}")
    print(f"  OAuth2 Refresh Token: {'configured' if refresh_token else 'NOT SET'}")
    print(f"  PAT (legacy):         {'configured' if pat else 'NOT SET'}")

    # Test whichever token is available
    token = access_token or pat
    if token:
        resp = requests.get(
            "https://api.ouraring.com/v2/usercollection/personal_info",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.status_code == 200:
            print(f"\n  API access: WORKING (using {'OAuth2' if access_token else 'PAT'})")
        elif resp.status_code == 401:
            print("\n  API access: EXPIRED/INVALID (HTTP 401)")
            if access_token and refresh_token:
                print("  Try: python api/oura_oauth2_setup.py --refresh")
        else:
            print(f"\n  API access: ERROR (HTTP {resp.status_code})")
    else:
        print("\n  No tokens configured. Run: python api/oura_oauth2_setup.py")


def main():
    parser = argparse.ArgumentParser(description="Oura Ring OAuth2 Setup")
    parser.add_argument("--refresh", action="store_true", help="Refresh expired access token")
    parser.add_argument("--status", action="store_true", help="Check token status")
    args = parser.parse_args()

    if args.status:
        check_status()
    elif args.refresh:
        refresh()
    else:
        authorize()


if __name__ == "__main__":
    main()
