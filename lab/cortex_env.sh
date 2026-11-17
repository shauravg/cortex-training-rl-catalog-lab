#!/usr/bin/env bash
# Export ARCTIC_CORTEX_* for the TRL driver from a Snowflake connection profile.
# Usage: source lab/cortex_env.sh [profile]   (default: cortex-training)
# The PAT is read from ~/.snowflake/connections.toml and never printed.
_profile="${1:-cortex-training}"
eval "$(python3 - "$_profile" <<'EOF'
import pathlib, shlex, sys, tomllib
c = tomllib.loads(pathlib.Path("~/.snowflake/connections.toml").expanduser().read_text())[sys.argv[1]]
host = c.get("host") or f"{c['account']}.snowflakecomputing.com"
# TLS certs cover the dashed hostname; account names with "_" fail verification.
name, _, domain = host.partition(".")
host = f"{name.replace('_', '-')}.{domain}"
for k, v in {"HOST": host, "DATABASE": c.get("database", "CORTEX_TRAINING_DB"),
             "SCHEMA": c.get("schema", "PUBLIC"), "PAT": c["token"]}.items():
    print(f"export ARCTIC_CORTEX_{k}={shlex.quote(v)}")
EOF
)"
echo "ARCTIC_CORTEX_HOST=$ARCTIC_CORTEX_HOST DATABASE=$ARCTIC_CORTEX_DATABASE SCHEMA=$ARCTIC_CORTEX_SCHEMA PAT=<set>"
unset _profile
