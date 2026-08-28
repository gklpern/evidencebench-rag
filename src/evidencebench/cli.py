import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from evidencebench.evaluation_runner import evaluate_api, load_api_cases

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.callback()
def main() -> None:
    """EvidenceBench operational commands."""


@app.command()
def evaluate(
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    base_url: Annotated[str, typer.Option(envvar="EB_EVAL_BASE_URL")] = "http://127.0.0.1:8080",
    api_key: Annotated[str, typer.Option(envvar="EB_EVAL_API_KEY", hide_input=True)] = "",
) -> None:
    """Evaluate retrieval, citation and abstention against a running API."""

    async def run() -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"authorization": f"Bearer {api_key}"},
            timeout=120,
        ) as client:
            return await evaluate_api(client=client, cases=load_api_cases(dataset))

    result = asyncio.run(run())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"Wrote {output}")
