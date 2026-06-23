"""Preflight credential validation for external climate/remote-sensing data sources.

Validates local auth setup for:
- NASA Earthdata (MODIS NDVI via earthaccess)
- Copernicus CDS API (ERA5 downloads)

Usage:
    python3 scripts/check_data_auth.py
"""

from __future__ import annotations

import netrc
import os
import stat
from pathlib import Path


def _check_earthdata() -> tuple[bool, str]:
    """Validate Earthdata credentials from env or ~/.netrc.

    Logic Flow:
        Checks EARTHDATA_USERNAME and EARTHDATA_PASSWORD first.
        Falls back to ~/.netrc machine entry for urs.earthdata.nasa.gov.
        Emits warning-style failure if netrc permissions are too open.

    Returns:
        Tuple (is_valid, message).

    Expected Exceptions:
        None. All parsing errors are converted to failure messages.
    """
    env_user = os.environ.get("EARTHDATA_USERNAME")
    env_pass = os.environ.get("EARTHDATA_PASSWORD")
    if env_user and env_pass:
        return True, "EARTHDATA via env: OK"

    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return False, "EARTHDATA missing: set env vars or add ~/.netrc entry for urs.earthdata.nasa.gov"

    try:
        mode = stat.S_IMODE(netrc_path.stat().st_mode)
        if mode & 0o077:
            return False, "EARTHDATA ~/.netrc permissions too open (expected 600)"

        auth = netrc.netrc(str(netrc_path)).authenticators("urs.earthdata.nasa.gov")
    except (netrc.NetrcParseError, OSError):
        return False, "EARTHDATA ~/.netrc unreadable or malformed"

    if not auth:
        return False, "EARTHDATA ~/.netrc missing machine urs.earthdata.nasa.gov"

    login, _, password = auth
    if not login or not password:
        return False, "EARTHDATA ~/.netrc entry missing login/password"

    return True, "EARTHDATA via ~/.netrc: OK"


def _parse_cdsapirc() -> dict[str, str]:
    """Parse ~/.cdsapirc key-value pairs.

    Returns:
        Dict with lowercase keys.

    Expected Exceptions:
        None. Returns empty dict when unavailable.
    """
    path = Path.home() / ".cdsapirc"
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        values[k.strip().lower()] = v.strip()
    return values


def _check_cds() -> tuple[bool, str]:
    """Validate CDS API credentials from env or ~/.cdsapirc.

    Logic Flow:
        Reads CDSAPI_KEY env first, then ~/.cdsapirc key fallback.
        Validates required uid:token shape expected by cdsapi.

    Returns:
        Tuple (is_valid, message).

    Expected Exceptions:
        None. Validation errors return False with message.
    """
    key = os.environ.get("CDSAPI_KEY")
    source = "env"
    if not key:
        cfg = _parse_cdsapirc()
        key = cfg.get("key")
        source = "~/.cdsapirc"

    if not key:
        return False, "CDS missing: set CDSAPI_KEY or configure ~/.cdsapirc"

    if ":" not in key:
        return False, "CDS key format invalid (expected uid:api-token)"

    return True, f"CDS via {source}: OK"


def main() -> int:
    """Run auth checks and print concise status.

    Logic Flow:
        Runs Earthdata and CDS checks.
        Prints PASS/FAIL per provider and returns process exit status.

    Returns:
        Exit code 0 when all checks pass, otherwise 1.

    Expected Exceptions:
        None. Returns non-zero instead of throwing.
    """
    checks = {
        "EARTHDATA": _check_earthdata(),
        "CDS": _check_cds(),
    }

    all_ok = True
    for provider, (ok, msg) in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {provider}: {msg}")
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
