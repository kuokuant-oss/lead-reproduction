from __future__ import annotations

import argparse
import json
from typing import Any

from colab_cli.auth import AuthProvider
from colab_cli.client import ColabRequestError
from colab_cli.commands import session as session_commands
from colab_cli.common import state
from colab_cli.utils import get_status_code


def _safe_error_payload(error: ColabRequestError) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "http_status": get_status_code(error),
        "reason": getattr(error.response, "reason", None),
    }
    try:
        body = json.loads(error.response_body or "{}")
    except (TypeError, json.JSONDecodeError):
        body = {}
    detail = body.get("error", {}) if isinstance(body, dict) else {}
    if isinstance(detail, dict):
        payload["error_status"] = detail.get("status")
        message = detail.get("message")
        if isinstance(message, str):
            payload["message"] = message[:300]
    retry_after = getattr(error.response, "headers", {}).get("Retry-After")
    if retry_after:
        payload["retry_after"] = retry_after
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--gpu", default="T4")
    args = parser.parse_args()

    state.auth_provider = AuthProvider.ADC
    try:
        session_commands.new(session=args.session, tpu=None, gpu=args.gpu)
    except ColabRequestError as error:
        print(json.dumps(_safe_error_payload(error), sort_keys=True), flush=True)
        return 1
    print(
        json.dumps(
            {"ok": True, "session": args.session, "accelerator": args.gpu},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
