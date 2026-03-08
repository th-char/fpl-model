# tests/test_cli.py
from click.testing import CliRunner
from fpl_model.cli.main import cli


class TestCLI:
    def test_cli_group_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "FPL Model" in result.output

    def test_ingest_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "--seasons" in result.output

    def test_simulate_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["simulate", "--help"])
        assert result.exit_code == 0

    def test_evaluate_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
