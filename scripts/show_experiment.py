from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from vfm_yolo_distillation.config import load_experiment_config, ultralytics_train_command


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(config_path: Path) -> None:
    config = load_experiment_config(config_path)
    command = " \\\n+  ".join(ultralytics_train_command(config))
    console.print(Panel.fit(config.description, title=config.name))
    console.print(command)


if __name__ == "__main__":
    app()
