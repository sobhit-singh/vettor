from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from vettor.config import VettorConfig
from vettor.orchestrator import Orchestrator


async def handle_mcp_request(req: dict[str, Any], cfg: VettorConfig) -> dict[str, Any]:
    """Handles a single JSON-RPC 2.0 MCP request."""
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "vettor-qa", "version": "0.4.0"},
                "capabilities": {"tools": {}},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "vettor_run_suite",
                        "description": "Run autonomous multi-sector Web QA against a target URL.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "goal": {"type": "string"},
                                "mode": {"type": "string", "enum": ["data_driven", "autonomous_goal", "crawl_smoke"]},
                            },
                            "required": ["url"],
                        },
                    }
                ]
            },
        }

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "vettor_run_suite":
            run_cfg = cfg.model_copy(deep=True)
            run_cfg.target.url = args.get("url", run_cfg.target.url)
            if args.get("goal"):
                run_cfg.suite.goal = args["goal"]
                run_cfg.suite.mode = "autonomous_goal"
            elif args.get("mode"):
                run_cfg.suite.mode = args["mode"]
            summary = await Orchestrator(run_cfg).run()
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": summary.model_dump_json(indent=2)}]
                },
            }

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def serve_mcp_stdio(cfg: VettorConfig) -> int:
    """Runs the stdio JSON-RPC MCP server loop."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = asyncio.run(handle_mcp_request(req, cfg))
        except Exception as exc:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(exc)}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    return 0
