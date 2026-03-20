"""SlideCraft API application entrypoint."""

import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routers import document, presentation
from src.bootstrap import startup

_API_PREFIX = "/api/v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress verbose third-party HTTP / SDK loggers
for _noisy in (
    "httpx",
    "httpcore",
    "google.generativeai",
    "google_genai",
    "google.api_core",
    "langchain_google_genai",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Re-enable SDK retry messages from the Google GenAI client, but condense the
# multi-KB JSON error body into a single readable line so rate-limit waits are
# visible without flooding the log.
_RETRY_RE = re.compile(r"Retrying \S+ in ([\d.]+) seconds.*?(\d+)s'\}\]", re.DOTALL)


class _SdkRetryFilter(logging.Filter):
    """Condense verbose SDK retry log records into a clean one-liner."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Retrying" not in msg:
            return False
        m = _RETRY_RE.search(msg)
        if m:
            record.msg = (
                "429 RESOURCE_EXHAUSTED — retrying in %.2fs (quota resets in ~%ss)"
                % (float(m.group(1)), m.group(2))
            )
        else:
            record.msg = "429 RESOURCE_EXHAUSTED — retrying (could not parse delay)"
        record.args = ()
        return True


_sdk_retry_log = logging.getLogger("google_genai._api_client")
_sdk_retry_log.setLevel(logging.INFO)
_sdk_retry_log.addFilter(_SdkRetryFilter())

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager for application lifespan."""
    startup()
    yield


app = FastAPI(title="SlideCraft API", version="0.1", lifespan=lifespan)

app.include_router(document.router, prefix=_API_PREFIX)
app.include_router(presentation.router, prefix=_API_PREFIX)
