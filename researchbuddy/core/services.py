"""
services.py — Docker-based service lifecycle helpers.

ResearchBuddy can talk to two optional services that the user normally has
to start by hand:

  * Neo4j   — graph database backend (port 7687 + Browser on 7474)
  * GROBID  — academic-PDF parser    (port 8070)

This module knows how to:

  * Detect whether each service is already responding
  * Detect whether Docker is installed and reachable
  * Start each service (using existing container if present, else `docker run`)
  * Wait for the service to become healthy
  * Cache the user's "yes" / "no, don't ask again" decision per session

Everything here is best-effort and silent on failure: if Docker isn't
installed or a container fails to start, the user can still run
ResearchBuddy with the in-memory NetworkX backend and pdfplumber.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)


# ── Service definitions ───────────────────────────────────────────────────────

@dataclass
class ServiceSpec:
    name:           str               # human-readable name ("Neo4j")
    container_name: str               # docker container name to use
    image:          str               # docker image with tag
    ports:          dict[int, int]    # host_port -> container_port
    env:            dict[str, str]    # NEO4J_AUTH=... etc.
    health_url:     str               # URL we GET to check liveness
    health_match:   Callable[[requests.Response], bool]
    extra_args:     list[str]         # additional `docker run` flags
    ready_timeout:  int = 60          # seconds to wait for health


def _is_neo4j_alive(r: requests.Response) -> bool:
    # Neo4j HTTP endpoint returns 200 with a "neo4j_version" JSON field
    try:
        return r.status_code == 200 and "neo4j_version" in r.json()
    except Exception:
        return r.status_code == 200


def _is_grobid_alive(r: requests.Response) -> bool:
    return r.status_code == 200 and r.text.strip().lower() == "true"


NEO4J_SPEC = ServiceSpec(
    name           = "Neo4j",
    container_name = "researchbuddy-neo4j",
    image          = "neo4j:5-community",
    ports          = {7474: 7474, 7687: 7687},
    env            = {"NEO4J_AUTH": "neo4j/researchbuddy"},
    health_url     = "http://localhost:7474",
    health_match   = _is_neo4j_alive,
    extra_args     = [],
    ready_timeout  = 90,   # Neo4j cold-starts in 30-60s
)

GROBID_SPEC = ServiceSpec(
    name           = "GROBID",
    container_name = "researchbuddy-grobid",
    image          = "lfoppiano/grobid:0.8.1",
    ports          = {8070: 8070},
    env            = {},
    health_url     = "http://localhost:8070/api/isalive",
    health_match   = _is_grobid_alive,
    extra_args     = ["--init", "--ulimit", "core=0"],
    ready_timeout  = 120,
)


# ── Persistent user preferences ───────────────────────────────────────────────

def _prefs_path() -> Path:
    """Where we cache the user's auto-launch preferences."""
    from researchbuddy.config import DATA_DIR
    return Path(DATA_DIR) / "service_prefs.json"


def load_prefs() -> dict:
    p = _prefs_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_prefs(prefs: dict) -> None:
    p = _prefs_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except Exception as e:
        logger.debug("Could not save service prefs: %s", e)


# ── Docker plumbing ───────────────────────────────────────────────────────────

def docker_available() -> bool:
    """True iff `docker` is on PATH and the daemon is responsive."""
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, timeout=5, text=True,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False


def _container_state(container_name: str) -> str:
    """
    Returns one of: 'running', 'stopped', 'missing'.
    """
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
            capture_output=True, timeout=5, text=True,
        )
        if r.returncode != 0:
            return "missing"
        status = r.stdout.strip().lower()
        if status == "running":
            return "running"
        return "stopped"
    except (subprocess.SubprocessError, OSError):
        return "missing"


def _docker_run(spec: ServiceSpec) -> bool:
    """`docker run` a fresh container from the spec."""
    cmd = ["docker", "run", "-d", "--name", spec.container_name]
    cmd.extend(spec.extra_args)
    for host_port, container_port in spec.ports.items():
        cmd.extend(["-p", f"{host_port}:{container_port}"])
    for k, v in spec.env.items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.append(spec.image)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
        if r.returncode != 0:
            logger.warning(
                "Failed to start %s container: %s",
                spec.name, (r.stderr or r.stdout).strip()[:300],
            )
            return False
        return True
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("Failed to start %s: %s", spec.name, e)
        return False


def _docker_start(container_name: str) -> bool:
    """Start an existing stopped container."""
    try:
        r = subprocess.run(
            ["docker", "start", container_name],
            capture_output=True, timeout=30, text=True,
        )
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


# ── Health checks ─────────────────────────────────────────────────────────────

def _service_alive(spec: ServiceSpec, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(spec.health_url, timeout=timeout)
        return spec.health_match(r)
    except requests.RequestException:
        return False


# ── Neo4j bolt probe ──────────────────────────────────────────────────────────

@dataclass
class Neo4jProbeResult:
    """Outcome of trying to actually connect to Neo4j via the bolt driver."""
    ok: bool
    reason: str = ""   # "" on success, otherwise a one-line diagnostic


def probe_neo4j_bolt(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "researchbuddy",
    timeout: float = 3.0,
) -> Neo4jProbeResult:
    """
    Try to authenticate against Neo4j over bolt. Returns a structured result
    so callers (status banner, manage-services menu) can show actionable info
    instead of a generic 'down' / 'connection failed'.
    """
    try:
        import neo4j as _neo4j
    except ImportError:
        return Neo4jProbeResult(
            ok=False,
            reason="neo4j Python driver not installed (pip install neo4j)",
        )

    try:
        driver = _neo4j.GraphDatabase.driver(
            uri, auth=(user, password),
            connection_timeout=timeout,
        )
        try:
            driver.verify_connectivity()
            return Neo4jProbeResult(ok=True)
        finally:
            driver.close()
    except Exception as e:
        msg = str(e).lower()
        if "auth" in msg or "unauthorized" in msg or "credentials" in msg:
            return Neo4jProbeResult(
                ok=False,
                reason=f"auth failed (set RESEARCHBUDDY_NEO4J_PASSWORD; tried '{password}')",
            )
        return Neo4jProbeResult(ok=False, reason=str(e)[:140])


def _wait_until_alive(spec: ServiceSpec, total_seconds: int) -> bool:
    """Poll the health endpoint until it returns true or we time out."""
    deadline = time.monotonic() + total_seconds
    delay = 1.0
    while time.monotonic() < deadline:
        if _service_alive(spec, timeout=2.0):
            return True
        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)
    return False


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class StartupResult:
    name: str
    started: bool
    already_running: bool
    error: Optional[str] = None


def ensure_running(spec: ServiceSpec, *, ready_message: bool = True) -> StartupResult:
    """
    Make sure the service described by `spec` is up and responding.

    Order of operations:
      1. If health endpoint already responds, do nothing.
      2. Otherwise, look at Docker container state:
         - 'running'  → wait for health
         - 'stopped'  → `docker start`, then wait for health
         - 'missing'  → `docker run` a new container, then wait for health
      3. If Docker isn't installed, give up gracefully.
    """
    if _service_alive(spec):
        return StartupResult(spec.name, started=False, already_running=True)

    if not docker_available():
        return StartupResult(
            spec.name, started=False, already_running=False,
            error="Docker is not installed or not running.",
        )

    state = _container_state(spec.container_name)
    if state == "running":
        # Container says it's running but health check failed — give it time
        pass
    elif state == "stopped":
        logger.info("[services] Starting existing %s container ...", spec.name)
        if not _docker_start(spec.container_name):
            return StartupResult(
                spec.name, started=False, already_running=False,
                error="`docker start` failed.",
            )
    else:  # missing
        logger.info(
            "[services] Pulling and starting %s (%s) — this may take a minute on first run ...",
            spec.name, spec.image,
        )
        if not _docker_run(spec):
            return StartupResult(
                spec.name, started=False, already_running=False,
                error="`docker run` failed (image pull or port conflict?).",
            )

    if ready_message:
        logger.info("[services] Waiting for %s to become healthy ...", spec.name)

    if _wait_until_alive(spec, spec.ready_timeout):
        return StartupResult(spec.name, started=True, already_running=False)

    return StartupResult(
        spec.name, started=False, already_running=False,
        error=f"Started but did not become healthy within {spec.ready_timeout}s.",
    )


def stop_service(spec: ServiceSpec) -> bool:
    """`docker stop` the service. Returns True on success."""
    if not docker_available():
        return False
    try:
        r = subprocess.run(
            ["docker", "stop", spec.container_name],
            capture_output=True, timeout=30, text=True,
        )
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False
