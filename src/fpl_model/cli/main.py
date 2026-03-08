"""CLI entry points for fpl-model."""

import asyncio

import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """FPL Model -- ML pipeline for Fantasy Premier League recommendations."""
    pass


@cli.command()
@click.option("--seasons", type=str, help="Comma-separated seasons, e.g. '2022-23,2023-24'")
@click.option("--current", is_flag=True, help="Fetch current season from FPL API")
@click.option("--db-path", default="data/fpl.db", help="Path to SQLite database")
@click.option("--cache-dir", default="data/raw", help="Path to raw file cache")
def ingest(seasons, current, db_path, cache_dir):
    """Download and ingest FPL data into the database."""
    from fpl_model.data.ingest import Ingester

    ingester = Ingester(db_path=db_path, cache_dir=cache_dir)

    async def _run():
        if current:
            click.echo("Ingesting current season from FPL API...")
            await ingester.ingest_current(season="2025-26")
            click.echo("Done.")
        if seasons:
            season_list = [s.strip() for s in seasons.split(",")]
            click.echo(f"Ingesting seasons: {season_list}")
            await ingester.ingest_seasons(season_list)
            click.echo("Done.")
        if not current and not seasons:
            click.echo("Specify --seasons or --current. Use --help for details.")

    asyncio.run(_run())


@cli.command()
@click.argument("model_name")
@click.option("--season", required=True, help="Season to simulate, e.g. '2024-25'")
@click.option("--db-path", default="data/fpl.db", help="Path to SQLite database")
def simulate(model_name, season, db_path):
    """Simulate a model over a historical season."""
    from fpl_model.data.db import Database
    from fpl_model.evaluation.metrics import compute_metrics
    from fpl_model.evaluation.reports import format_report
    from fpl_model.models.registry import ModelRegistry
    from fpl_model.simulation.engine import SeasonSimulator

    db = Database(db_path)
    registry = ModelRegistry()
    model = registry.get(model_name)
    sim = SeasonSimulator(model=model, season=season, db=db)
    result = sim.run()
    metrics = compute_metrics(result)
    click.echo(format_report(model_name, metrics))


@cli.command()
@click.argument("model_name")
@click.option("--seasons", required=True, help="Comma-separated seasons to evaluate")
@click.option("--db-path", default="data/fpl.db", help="Path to SQLite database")
def evaluate(model_name, seasons, db_path):
    """Evaluate a model across multiple seasons."""
    from fpl_model.data.db import Database
    from fpl_model.evaluation.metrics import compute_metrics
    from fpl_model.evaluation.reports import format_report
    from fpl_model.models.registry import ModelRegistry
    from fpl_model.simulation.engine import SeasonSimulator

    db = Database(db_path)
    registry = ModelRegistry()
    model = registry.get(model_name)
    season_list = [s.strip() for s in seasons.split(",")]
    for season in season_list:
        sim = SeasonSimulator(model=model, season=season, db=db)
        result = sim.run()
        metrics = compute_metrics(result)
        click.echo(format_report(f"{model_name} ({season})", metrics))
        click.echo()


@cli.command()
@click.argument("model_names", nargs=-1, required=True)
@click.option("--seasons", required=True, help="Comma-separated seasons to compare on")
@click.option("--db-path", default="data/fpl.db", help="Path to SQLite database")
def compare(model_names, seasons, db_path):
    """Compare multiple models on the same season(s)."""
    from fpl_model.data.db import Database
    from fpl_model.evaluation.comparison import compare_results
    from fpl_model.evaluation.reports import format_comparison
    from fpl_model.models.registry import ModelRegistry
    from fpl_model.simulation.engine import SeasonSimulator

    db = Database(db_path)
    registry = ModelRegistry()
    season_list = [s.strip() for s in seasons.split(",")]
    all_results = {}
    for name in model_names:
        model = registry.get(name)
        for season in season_list:
            sim = SeasonSimulator(model=model, season=season, db=db)
            result = sim.run()
            all_results[f"{name} ({season})"] = result
    comparison = compare_results(all_results)
    click.echo(format_comparison(comparison))
