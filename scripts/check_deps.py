#!/usr/bin/env python3
"""Check and report status of Autonomous-Loop dependencies."""

import sys
import subprocess
import importlib.util

def check(cmd, args=""):
    try:
        r = subprocess.run(f"{cmd} {args}".split(), capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False

def check_docker():
    """Docker must be running."""
    r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
    if r.returncode != 0:
        return False, "Docker not running"
    r = subprocess.run(["docker", "ps", "--filter", "name=hermes-neo4j", "--format", "{{.Names}}"], capture_output=True, text=True)
    if "hermes-neo4j" in r.stdout:
        return True, "Neo4j container running"
    return True, "Docker OK, but Neo4j container NOT running"

def check_neo4j_image():
    r = subprocess.run(["docker", "images", "-q", "neo4j:5.26.0-community"], capture_output=True, text=True)
    if r.stdout.strip():
        return True, "Neo4j image present"
    return False, "Neo4j image not downloaded yet"

def check_neo4j_ready():
    """Try to connect to Neo4j bolt port."""
    import socket
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("localhost", 7687))
        s.close()
        return True, "Bolt port open"
    except:
        return False, "Bolt port not reachable (Neo4j not ready)"

def check_faiss():
    spec = importlib.util.find_spec("faiss")
    if spec is None:
        return False, "faiss not installed"
    try:
        import faiss
        return True, f"faiss {faiss.__version__ if hasattr(faiss, '__version__') else 'ok'}"
    except Exception as e:
        return False, f"faiss error: {e}"

def check_neo4j_driver():
    spec = importlib.util.find_spec("neo4j")
    if spec is None:
        return False, "neo4j driver not installed"
    return True, "neo4j driver ok"

def main():
    checks = [
        ("Docker", lambda: check("docker", "info")),
        ("Neo4j container", check_docker),
        ("Neo4j image", check_neo4j_image),
        ("Neo4j ready", check_neo4j_ready),
        ("faiss", check_faiss),
        ("neo4j driver", check_neo4j_driver),
    ]

    print("=== Autonomous-Loop Dependency Check ===\n")
    all_ok = True
    for name, fn in checks:
        try:
            ok, msg = fn()
        except Exception as e:
            ok = False
            msg = str(e)
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: {msg}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All checks passed.")
        return 0
    else:
        print("Some checks failed. Run setup commands below.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
