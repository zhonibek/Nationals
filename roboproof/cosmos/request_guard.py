"""Reject remote file references before Gradio can copy/fetch input resources."""

import json

from .errors import CosmosError
from .media import guard_media_path


class LocalMediaRequestGuard:
    def __init__(self, app, config):
        self.app = app
        self.config = config

    async def _reject(self, send, code="path", status=400):
        body = json.dumps({"status": "error", "code": code, "message": str(CosmosError(code))}).encode("utf-8")
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode("ascii"))]})
        await send({"type": "http.response.body", "body": body})

    def _validate_references(self, payload):
        pending = [(payload, 0)]
        references = 0
        while pending:
            value, depth = pending.pop()
            if depth > 16:
                raise CosmosError("path")
            if isinstance(value, dict):
                metadata = value.get("meta")
                is_file = isinstance(metadata, dict) and metadata.get("_type") == "gradio.FileData"
                if is_file or "path" in value:
                    references += 1
                    if references > 1:
                        raise CosmosError("upload")
                    path = guard_media_path(value.get("path"), self.config)
                    if not 0 < path.stat().st_size <= self.config.limits.max_upload_bytes:
                        raise CosmosError("upload")
                pending.extend((child, depth + 1) for child in value.values())
            elif isinstance(value, list):
                pending.extend((child, depth + 1) for child in value)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        route = scope.get("path", "")
        if "/proxy=" in route:
            return await self._reject(send)
        if "/file=" in route:
            try:
                guard_media_path(route.split("/file=", 1)[1], self.config)
            except Exception:
                return await self._reject(send)
        headers = {name.lower(): value for name, value in scope.get("headers", [])}
        is_json = headers.get(b"content-type", b"").lower().split(b";", 1)[0].strip() == b"application/json"
        if scope.get("method") not in ("POST", "PUT") or not is_json:
            return await self.app(scope, receive, send)
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > 64 * 1024:
                return await self._reject(send, "upload", 413)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        try:
            self._validate_references(json.loads(body))
        except CosmosError as error:
            return await self._reject(send, error.code)
        except Exception:
            return await self._reject(send)

        delivered = False

        async def replay_receive():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        return await self.app(scope, replay_receive, send)


def guarded_app_kwargs(config) -> dict:
    try:
        from starlette.middleware import Middleware
    except Exception:
        raise CosmosError("dependency") from None
    return {"middleware": [Middleware(LocalMediaRequestGuard, config=config)]}
