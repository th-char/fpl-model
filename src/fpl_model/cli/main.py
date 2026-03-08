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
    from fpl_model.models.defaults import get_default_registry
    from fpl_model.simulation.engine import SeasonSimulator

    db = Database(db_path)
    registry = get_default_registry()
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
    from fpl_model.models.defaults import get_default_registry
    from fpl_model.simulation.engine import SeasonSimulator

    db = Database(db_path)
    registry = get_default_registry()
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
    from fpl_model.models.defaults import get_default_registry
    from fpl_model.simulation.engine import SeasonSimulator

    db = Database(db_path)
    registry = get_default_registry()
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


@cli.command()
@click.argument("model_name")
@click.option("--seasons", required=True, help="Comma-separated training seasons, e.g. '2022-23,2023-24'")
@click.option("--db-path", default="data/fpl.db", help="Path to SQLite database")
def train(model_name, seasons, db_path):
    """Train a model on historical data."""
    from fpl_model.data.db import Database
    from fpl_model.models.base import HistoricalData, SeasonData
    from fpl_model.models.defaults import get_default_registry

    db = Database(db_path)
    registry = get_default_registry()
    model = registry.get(model_name)

    season_list = [s.strip() for s in seasons.split(",")]
    historical = HistoricalData()

    for season in season_list:
        click.echo(f"Loading data for season {season}...")
        players = db.read("players", where={"season": season})
        gw_perf = db.read("gameweek_performances", where={"season": season})
        fixtures = db.read("fixtures", where={"season": season})
        teams = db.read("teams", where={"season": season})

        max_gw = int(gw_perf["gameweek"].max()) if len(gw_perf) > 0 else 38
        historical.seasons[season] = SeasonData(
            gameweek_performances=gw_perf,
            fixtures=fixtures,
            players=players,
            teams=teams,
            current_gameweek=max_gw + 1,
            season=season,
        )

    click.echo(f"Training model '{model_name}' on {len(season_list)} season(s)...")
    model.train(historical)
    click.echo("Training complete.")


@cli.command()
@click.argument("model_name")
@click.option("--season", required=True, help="Season to get recommendations for, e.g. '2024-25'")
@click.option("--gameweek", required=True, type=int, help="Gameweek number")
@click.option("--db-path", default="data/fpl.db", help="Path to SQLite database")
def recommend(model_name, season, gameweek, db_path):
    """Get model recommendations for a specific gameweek."""
    from fpl_model.data.db import Database
    from fpl_model.models.base import SeasonData
    from fpl_model.models.defaults import get_default_registry
    from fpl_model.simulation.actions import ChipType
    from fpl_model.simulation.state import PlayerInSquad, SquadState

    db = Database(db_path)
    registry = get_default_registry()
    model = registry.get(model_name)

    # Load season data
    players = db.read("players", where={"season": season})
    gw_perf = db.read("gameweek_performances", where={"season": season})
    fixtures = db.read("fixtures", where={"season": season})
    teams = db.read("teams", where={"season": season})

    data = SeasonData(
        gameweek_performances=gw_perf,
        fixtures=fixtures,
        players=players,
        teams=teams,
        current_gameweek=gameweek,
        season=season,
    )

    # Build a basic squad state from top-15 players by total_points
    # (a simple heuristic default when no specific squad is provided)
    if len(players) == 0:
        click.echo(f"No player data found for season {season}.")
        return

    squad_players = _build_default_squad(players)
    state = SquadState(
        players=squad_players,
        budget=0,
        free_transfers=1,
        chips_available={ct: 2 for ct in ChipType},
        current_gameweek=gameweek,
    )

    actions = model.recommend(state, data)

    click.echo(f"Recommendations for {model_name} -- {season} GW{gameweek}:")
    for action in actions:
        click.echo(f"  {action}")


def _build_default_squad(players):
    """Build a simple 15-player squad (2 GK, 5 DEF, 5 MID, 3 FWD) from players DataFrame."""
    from fpl_model.simulation.state import PlayerInSquad

    targets = {1: 2, 2: 5, 3: 5, 4: 3}  # GK, DEF, MID, FWD
    squad = []
    for pos, count in targets.items():
        pos_players = players[players["element_type"] == pos].nlargest(count, "total_points")
        for _, row in pos_players.iterrows():
            squad.append(
                PlayerInSquad(
                    code=int(row["code"]),
                    element_type=int(row["element_type"]),
                    buy_price=int(row.get("now_cost", 50)),
                    sell_price=int(row.get("now_cost", 50)),
                )
            )
    return squad
