# FPL Model — Development Guide

## Project Structure
```
src/fpl_model/
  data/           # ETL: Sources -> Transformers -> Unifier -> SQLite DB
    etl/          # VaastavSource, VaastavTransformer, Unifier
    ingest.py     # Ingester (async, orchestrates ETL)
    db.py         # Database (SQLite wrapper, chunked inserts)
  models/
    base.py       # ActionModel, PredictOptimizeModel, Predictor, Optimizer
    defaults.py   # get_default_registry(), create_ppo_agent()
    features.py   # Shared feature engineering (XGBoost, LSTM, RL)
    predictors/   # FormPredictor, XGBoostPredictor, SequencePredictor
    optimizers/   # GreedyOptimizer, LPOptimizer
    rl/           # PPOAgent, FPLEnvironment
  simulation/
    engine.py     # SeasonSimulator (replays seasons, mid-season retraining)
    state.py      # SquadState, PlayerInSquad
    actions.py    # Transfer, SetLineup, SetCaptain, PlayChip, etc.
    scoring.py    # GW scoring logic
  evaluation/     # compare_results(), format_comparison()
  cli/            # Click CLI: ingest, simulate, evaluate, compare, train, recommend
tests/            # 150 tests
notebooks/        # model_comparison.ipynb
docs/plans/       # Design and implementation docs
```

## Tooling
- **Package manager:** UV (`/home/node/.local/bin/uv`), Python 3.12
- **Run tests:** `uv run pytest -x -q`
- **Lint:** `uv run ruff check src/ tests/ --fix`
- **CLI:** `uv run fpl-model <command>`

## Key Conventions
- `element_type`: 1=GK, 2=DEF, 3=MID, 4=FWD, 5=MGR (managers in 2024-25+)
- `now_cost` / prices in 0.1m units (e.g., 145 = £14.5m)
- Cross-season player linking via `code` field (stable), NOT `id` (changes per season)
- SQLite has a 999 variable limit — use chunked inserts (`chunksize` in `db.py`)
- Ingester is async — use `await ingester.ingest_seasons()` in notebooks

## Models
| Name | Predictor | Optimizer | Notes |
|------|-----------|-----------|-------|
| `form-greedy` | FormPredictor | GreedyOptimizer | Baseline, no training |
| `xgb-greedy` | XGBoostPredictor | GreedyOptimizer(enable_transfers=True) | 25+ features |
| `xgb-lp` | XGBoostPredictor | LPOptimizer | ILP via PuLP |
| `sequence-lp` | SequencePredictor (LSTM) | LPOptimizer | Sequence features |
| `ppo-agent` | PPOAgent | Built-in policy | Requires db+seasons, use create_ppo_agent() |

### Mid-Season Retraining
- `SeasonSimulator(retrain_every_n_gws=N, train_seasons=[...])` retrains model periodically
- Predictors support `recency_decay` param: exponential weighting `decay^(max_gw - gw)`
- XGBoost uses `sample_weight`, LSTM uses weighted MSE loss

## Key Paths
- Design docs: `docs/plans/` (pipeline design, advanced models, mid-season retraining)
- Source code: `src/fpl_model/` (data, models, simulation, evaluation, cli)
- Tests: `tests/` (150 tests)
- Notebook: `notebooks/model_comparison.ipynb`
- DB: `data/fpl.db` (SQLite, gitignored)
- Cache: `data/cache/` (gitignored)

## Architecture Notes
- `ActionModel` is universal interface; `PredictOptimizeModel` composes `Predictor` + `Optimizer`
- `SeasonSimulator` replays historical seasons calling model per GW
- Shared features in `models/features.py` (used by XGBoost, LSTM, RL)
- Data flows: Source -> ETL Transformer -> Unifier -> SQLite DB
- Cross-season player linking via `code` field (stable, unlike `id`)

## Known Limitations
- LP optimizer: no multi-GW lookahead, no chip heuristics, no paid transfers
- Greedy optimizer: only free transfers (no paid hits)
- Recency decay is intra-season only (GW numbers overlap across seasons)
- `_build_training_data()` doesn't filter future fixture scores during retraining
- Pending work: Free Hit revert, budget validation & sell price tracking, future data leakage prevention

---

# How to play FPL

Fantasy Premier League (FPL) is a season-long manager game where you build, manage and score points from a virtual 15-player squad based on real Premier League performances. Success requires squad construction, weekly selection, chip/tactical use, and transfer planning. Below is a compact, practical guide to how it works and how to play.

Essentials — squad, budget, scoring

    Squad: 15 players — 2 goalkeepers, 5 defenders, 5 midfielders, 3 forwards.
    Budget: £100.0m (subject to small yearly changes). Each player has a price; you must stay within the budget.
    Formation constraints: Choose a starting XI each gameweek respecting positions (e.g., 3-4-3, 4-4-2). Bench order matters for automatic substitutions.
    Scoring highlights (season rules may tweak points): Goals, assists, clean sheets, minutes played, goal conceded penalties for defenders/goalkeepers, bonus points, and negative points for yellow/red cards and own goals.
    Captaincy: Captain scores double; vice-captain inherits if captain doesn’t play.
    Automatic substitutions: If a starter doesn’t play, bench players come on in order, keeping formation constraints.

Weekly cycle and transfers

    Gameweek: Each Premier League matchweek corresponds to an FPL gameweek. Deadline is typically 90 minutes before the first match of that gameweek.
    Free transfer: You get one free transfer per gameweek. Free transfers can be saved (max usually 2). Additional transfers cost -4 points each (per extra transfer beyond the free transfers).
    Wildcard: Two wildcards per season (one early, one later) that allow unlimited free transfers for that gameweek only (no point hits).
    Chips (special one-use options): Typical chips include Bench Boost (score bench players), Triple Captain (captain scores triple), and Free Hit (unlimited transfers for one gameweek only, squad reverts afterwards). Chips are strategic for blanks and double gameweeks.

Managing blanks and double gameweeks

    Blank gameweek: Some teams don’t play (cup scheduling), causing fewer players available. Use Free Hit or Wildcard to navigate, or transfer players from non-playing teams.
    Double gameweek: Some teams play twice in one gameweek; stacking players from those teams can boost points. Use Bench Boost or Triple Captain during well-chosen double gameweeks for maximum gain.

Team value and transfers

    Player prices fluctuate based on transfers in/out. Increasing team value through popular picks enables bigger buying power later.
    Early-season value rises are valuable but don’t force uncapped transfers for small gains; prioritize player form and fixtures.

Captaincy and fixtures

    Captain choice is the single most important weekly decision. Bias toward reliable, attacking, fit players with favorable fixtures.
    Fixture difficulty: Use fixture schedules to plan transfers and chips. Fixtures rotate; a run of easy games is a good time to bring in players.

Bench strategy and rotation

    Bench should cover position and rotation risk (e.g., keep a cheap playing midfielder and goalkeeper).
    First substitute: set a sensible order so automatic subs preserve formation and replace non-playing starters sensibly.

Bonus Points System (BPS)

    BPS awards extra “bonus” points to the best-performing players in a match based on underlying stats (passes, tackles, saves, shots, etc.). Top 3 performers typically get 3/2/1 bonus.

Leagues and formats

    Head-to-head and classic leagues: Classic ranks total points; head-to-head pits managers weekly with wins/draws/losses.
    Cups and mini-leagues: Run private leagues for friends and online public leagues.

Typical season plan (practical steps)

    Draft an initial 15-man squad: balance premium (e.g., top forwards), mid-range, and cheap playing bench fodder.
    Each week: set lineup, pick captain, use free transfer(s) if needed, check injuries/rotations before deadline.
    Use first wildcard around early chaos or to restructure; hold second for late-season fixture congestion/double gameweeks.
    Save chips for double gameweeks or use Free Hit on blanks.
    Track form, fixtures, and injury news; plan transfers 2–4 gameweeks ahead for fixture swings.

Common pitfalls to avoid

    Overreacting to one bad week; chasing template picks blindly.
    Burning chips too early (use them when blanks/double gameweeks give leverage).
    Ignoring rotation risk and squad depth; cheap bench players who play regularly are valuable.
    Taking frequent -4 hits out of impatience — sustainable use is when long-term gain outweighs immediate hit.

Resources and tools

    Official FPL site/app: squad management, live scoring, news.
    Fixture trackers, expected points (xP) models, form tables, and community sites (Twitter, Reddit, YouTube) for injury/rotation intel.
    Use spreadsheets or third-party sites for transfer planning and chip planning.

Examples of common manager moves

    Week with a favored double fixture: Bench Boost + pick a captain who also has two matches.
    Blank gameweek with many players missing: Free Hit to field a full XI.
    Mid-season poor form: Wildcard to restructure around players with easier calendars.

Closing principle
FPL is a balance of planned strategy and adaptable micro-decisions: build a balanced, playing 15; prioritize captaincy and transfers; conserve chips for high-leverage weeks; and use fixtures and rotation insight to shape the squad across the season.

## How FPL Points are Awarded to a Player

Managers in Fantasy Premier League earn points from their players for a number of actions.

These include goals, assists, clean sheets, saves and NEW for 2025/26: defensive contributions. They can also earn additional bonus points if they are among the top-performing players in the Bonus Points System (BPS) in any given match.
Action 	Points
For playing up to 60 minutes 	1
For playing 60 minutes or more (excluding stoppage time)  	2
For each goal scored by a goalkeeper 	10
For each goal scored by a defender  	6
For each goal scored by a midfielder 	5
For each goal scored by a forward 	4
For each assist for a goal 	3
For a clean sheet by a goalkeeper or defender 	4
For a clean sheet by a midfielder  	1
For every 3 shots saved by a goalkeeper  	1
For each penalty save 	5
For 10 defensive contributions by a defender 	2
For 12 defensive contributions by a midfielder 	2
For 12 defensive contributions by a forward 	2
For each penalty miss  	-2
Bonus points for the best players in a match  	1-3
For every 2 goals conceded by a goalkeeper or defender 	-1
For each yellow card  	-1
For each red card 	-3
For each own goal 	-2
Clean sheets

If a player has been substituted before a goal is conceded, this will not affect any clean sheet points provided they have played 60 minutes or more.
Assists: Passes

Assists are awarded to the player from the scoring team who makes the final pass or touch before a goal is scored.

An assist can be awarded even if an opponent gets a touch before the goal is scored. However, there must be only one defensive touch and the goalscorer must receive the ball inside the penalty area.

If the goalscorer received the ball outside the box, then the defensive touch must not significantly alter the intended destination of the pass or cross.

That "intended destination" can often be a team-mate if he is a clear target. But it may also be an area of the pitch.

If the goalscorer loses and then regains possession, then no assist is awarded.

You can read more about the new assist rules here.
Assists: Rebounds

If a shot on goal is blocked by an opposition player, is saved by a goalkeeper or hits the woodwork, and a goal is scored from the rebound, then an assist is awarded.
Assists: Own goals

If a player shoots or passes the ball and forces an opposing player to put the ball in his own net, then an assist is awarded.
Assists: Penalties and free-kicks

In the event of a penalty or a free-kick, the player earning the foul is awarded an assist if a goal is directly scored.

If the taker also won the penalty or free-kick, no assist is given.
Bonus points

The Bonus Points System (BPS) utilises a range of statistics to create a BPS score for every player.

The three best performing players in each match will be awarded bonus points. Three points will be awarded to the highest-scoring player, two to the second best and one to the third.

Examples of how bonus point ties will be resolved are as follows:

    If there is a tie for first place, Players 1 & 2 will receive 3 points each and Player 3 will receive 1 point.
    If there is a tie for second place, Player 1 will receive 3 points and Players 2 and 3 will receive 2 points each.
    If there is a tie for third place, Player 1 will receive 3 points, Player 2 will receive 2 points and Players 3 & 4 will receive 1 point each.

More than three players can earn bonus points in a match. For example, if two players top the BPS in a match with a score of 35, with three players then tied on 34BPS, the two players with 35BPS will both earn three bonus points, while the three players with 34BPS will all claim one bonus point each

## How to Use Chips

Fantasy Premier League managers have the opportunity to give their points total a big boost by using their chips.

There are a total of eight chips to use over the course of a season. Only one chip can be played in a single Gameweek.

The chips available are listed below. This season you will have TWO of each chip - one to use in the first half of the season, before the end of Gameweek 19, and the other to use in the second half of the season from Gameweek 20 onwards.
Bench Boost

The points scored by your benched players are included in your total.
Free Hit

Make unlimited free transfers for a single Gameweek. At the next deadline, your squad is returned to how it was before those unlimited moves were made.

The Free Hit can't be played in Gameweek 1.
Triple Captain

Your captain points are tripled instead of doubled.
Wildcard

All transfers (including those already made) are free of charge.
An image of Igor Thiago for FPL

How and when to use your chips in 2025/26 Fantasy?
Fantasy Premier League
How many times can you use a chip and how do you play them?

New for Fantasy 2025/26, all four chips can each be used twice a season, once in either half of the campaign.

The Bench Boost and Triple Captain chips are played when saving your team on the "Pick Team" page, as seen below. They can be cancelled at any time before the Gameweek deadline.
IMG_1858

The Free Hit chip is played when confirming your transfers. It can't be cancelled after being confirmed and it can't be used in both Gameweek 19 and Gameweek 20.

Your Wildcard chip is played when confirming transfers that cost points and can't be cancelled once played or played in Gameweek 1.

When playing either a Wildcard or your Free Hit chip, you will KEEP your banked transfers.
When is a good time to play a chip?

Many Fantasy managers like to save their second set of chips for Blank or Double Gameweeks, although there are plenty of alternative strategies.

In Blank Gameweeks, at least one of the Premier League clubs - usually more - will be without a league fixture. Using a Free Hit can help navigate this.

In Double Gameweeks, at least one of the Premier League clubs - again, usually more - has two league fixtures to contest. These are typically popular Gameweeks for the Bench Boost and Triple Captain chips.

A Wildcard is often best saved for when your squad most needs an overhaul. For example, this could be if you have multiple players injured, suspended or not starting for their teams. However, it could also be used to target a particular set of fixtures for certain teams.

Many managers often use their first Wildcard quite early in the season to recruit players who have caught the eye in the first few weeks of the campaign. It should always be played with a longer-term outlook in mind, however.

In the first half of the season, the Bench Boost chip could be played in Gameweek 1 or straight after a Wildcard, as managers will have unlimited transfers to build a 15-man squad.

The Triple Captain chip could be used when Liverpool's Mohamed Salah (£14.5m) has a home fixture against Sunderland in Gameweek 14, or when Erling Haaland (£14.0m) faces Burnley in Gameweek 6, for example.

## Managing the Team

In Fantasy Premier League, the season is split into 38 "Gameweeks".

Usually, a "Gameweek" consists of 10 fixtures, where each of the Premier League's 20 clubs play one match against another.

Fantasy managers must make all of their changes and confirm their final line-up before the "Gameweek deadline", which is 90 minutes before the opening match of the Gameweek.

Managers must choose a starting 11 from their 15-man squad. And there are three decisions to make.
1. Formation

In each Gameweek you can put your players in different formations, such as a 3-4-3 or 4-4-2.

Any of these formations can be chosen as long as your line-up includes one goalkeeper, at least three defenders, at least two midfielders and at least one forward.

You can change formation by replacing a starter with a player who is on the bench and is classified in a different position in the game.

So for example, if your line-up is in a 4-4-2 formation and you want to switch to 3-4-3, go to the "Pick Team" page and click on the yellow circle with the red and green arrows in the top-left corner of the player card of one of your starting defenders, and then click on the same circle of the forward who is on your bench.

In this example, you would click on the icon on Gabriel's card, and then do the same on Jarrod Bowen's card, to swap them.
Screen grab 6

Your third forward (Bowen) will therefore move from your bench into your starting 11, and your chosen defender (Gabriel) will move down to the bench.
FPL screen grab 2

Click "Save My Team" to make sure the changes are saved.
2. Captaincy

You must also choose a captain and a vice-captain from the players in your line-up.

To select your captain, go to the "Pick Team" page, click on the player who you want to captain and tick the "Captain" box at the bottom. Do the same for your vice-captain, ticking the "Vice Captain" box instead.

If your captain plays in his Premier League match in that Gameweek, he will score DOUBLE POINTS for your Fantasy team. If your captain fails to feature at all in his Premier League match for any reason, then the captain's armband passes to your vice-captain, who will score double points instead.

If neither your captain nor vice-captain plays, you won't receive double points from any player in that Gameweek.

Again, click "Save My Team" below the pitch graphic to make sure the changes to your captain and vice-captain are saved.
3. Order of your bench

If one or more of the players in your starting 11 don't play in their Premier League match that Gameweek, they will be automatically substituted at the end of the Gameweek. This makes the order of your substitutes important.

If your goalkeeper doesn't play in the Gameweek, he will be substituted by your replacement 'keeper if they played. There is no decision for you to make here.

But if any outfield player doesn't play, they will be substituted and replaced by the outfield substitute who you have placed FIRST on your bench (from left to right), as long as that substitute has played in the Gameweek and doesn't break the formation rules.

If your first substitute hasn't played in the Gameweek, your second substitute will replace the outfield player in your starting 11 who didn't play. This is also the case if your second substitute hasn't played in the Gameweek - your third substitute will replace the outfield player in your starting 11 who didn't play.

The formation rules are that your line-up must include one goalkeeper, at least three defenders, at least two midfielders and at least one forward.

So, if your starting 11 has three defenders, then a defender can only be automatically substituted by another defender. Otherwise, you would end up with just two defenders in your scoring 11.

To change your substitute order, go to the "Pick Team" page and click on the benched player in question. You can then select "Substitute" at the bottom of the right-hand column and click on the player card of the player you want to swap them with, before clicking "Substitute" again.

Then click "Save My Team" to make sure the changes are saved.

## Transfers

After selecting your Fantasy Premier League squad, you can buy and sell players in the transfer market.
Transfer allowance

Unlimited transfers can be made at no cost until the season starts.

Once the first "Gameweek deadline" of the season has passed, managers are given ONE free transfer for each Gameweek.

This enables you to sign a player for your 15-man squad, but you must also sell one, and the switch must be within your overall budget of £100.0m.

If you don't use your free transfer, you can roll it over into the following Gameweek, and have TWO free transfers. You can continue doing this up a maximum of FIVE free transfers.

If you want to make any additional transfers in a Gameweek beyond your free one(s), it will come at a cost of four Fantasy points.
How to make a transfer

On the transfers page, remove the player you want to transfer out by clicking the "X" on their player card on the pitch graphic.
Screenshot 2025-07-25 at 11.00.29

In the "Player Selection" column on the left-hand side, choose the player you want to transfer in.

This new signing has to be in the SAME position as your departing player, and they must fit within your budget. So you must sell a midfielder to buy a midfielder.

If you want to buy a player but you already own three players from their Premier League club, you won't be able to make the switch.

Once you've clicked on the player you want to transfer in, click "Make Transfers" underneath the pitch graphic, and then "Confirm".
Player prices

Player prices change during the season based on the popularity of the player in the transfer market. Players who are heavily bought by managers will rise in price, while those who are sold by many will become cheaper.

Player prices do not change until the season starts.
Player profits and losses

When you're on your "Transfers" page, and can see the price tag for each of the players in your existing squad, the price you see is the amount you will get for SELLING the player.

This may differ from the player's current PURCHASE price.

That's because you only get £0.1m of profit for every £0.2m that the player rises in cost.

For example, you could buy a player for £5.0m who then rises in price by £0.4m during the time that you own him.

Other managers will have to pay £5.4m to buy that player. But your selling price will be £5.2m.

But if you buy a player for £5.0m who then rises in price by £0.3m during the time that you own him, your selling price will be £5.1m.

Be warned, a player's selling price can also go down. For example, if you buy a player for £5.0m and their value drops to £4.7m, your selling price will be £4.7m. Every £0.1m change is lost.

The "List" view on the "Transfers" page breaks down current price (CP), selling price (SP) and purchase price (PP) more clearly, as seen below.



# FPL API

The Fantasy Premier League (FPL) API provides a variety of endpoints that allow developers and enthusiasts to retrieve information about the game, including player data, team data, league standings, and much more. Below is a comprehensive summary of the main FPL API endpoints and their purposes:

1. General Information

    Endpoint: https://fantasy.premierleague.com/api/bootstrap-static/

    Description: Retrieves a bulk of the static data required for the game, including information on all players, teams, events (gameweeks), and more.

    Key Data: Players, teams, fixtures, phases, game settings, etc.

2. Player Specific Information

    Endpoint: https://fantasy.premierleague.com/api/element-summary/{player_id}/

    Description: Provides detailed statistics and history for a specific player, identified by their player ID.

    Key Data: Player history per gameweek, past seasons' history, fixtures, and stats.

3. Team Specific Information

    Endpoint: https://fantasy.premierleague.com/api/entry/{team_id}/

    Description: Retrieves general information about a specific user's FPL team.

    Key Data: Team name, total points, overall rank, leagues, and more.

4. Team History

    Endpoint: https://fantasy.premierleague.com/api/entry/{team_id}/history/

    Description: Provides a detailed history of a specific user's team across gameweeks and seasons.

    Key Data: Season history, current gameweek points, chips used, transfers made, etc.

5. Team Picks

    Endpoint: https://fantasy.premierleague.com/api/entry/{team_id}/event/{event_id}/picks/

    Description: Shows the picks for a specific team for a given gameweek (event).

    Key Data: Players selected, captain, vice-captain, bench, etc.

6. Transfers

    Endpoint: https://fantasy.premierleague.com/api/entry/{team_id}/transfers/

    Description: Retrieves the transfer history for a specific team.

    Key Data: Transfers made, cost, points hits, transfer date, etc.

7. Classic League Standings

    Endpoint: https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/

    Description: Provides standings for a classic league (i.e., standard league based on total points).

    Key Data: Team standings, points, overall rank, etc.

8. H2H League Standings

    Endpoint: https://fantasy.premierleague.com/api/leagues-h2h/{league_id}/standings/

    Description: Retrieves the standings for a head-to-head (H2H) league.

    Key Data: Team standings, matches, points, etc.

9. League Fixtures (H2H)

    Endpoint: https://fantasy.premierleague.com/api/leagues-h2h-matches/league/{league_id}/?page={page_number}&event={event_id}

    Description: Provides the fixtures and results for a specific H2H league and gameweek.

    Key Data: Fixtures, results, opponent data, etc.

10. Live Gameweek Data

    Endpoint: https://fantasy.premierleague.com/api/event/{event_id}/live/

    Description: Shows live data for a specific gameweek, including real-time player scores.

    Key Data: Player scores, bonus points, assists, goals, etc.

11. Fixtures

    Endpoint: https://fantasy.premierleague.com/api/fixtures/

    Description: Provides all the fixtures for the season.

    Key Data: Fixture dates, teams involved, whether the match is finished, etc.

12. Game Settings

    Endpoint: https://fantasy.premierleague.com/api/game-settings/

    Description: Retrieves the settings for the game, including scoring rules, bonus point system, etc.

    Key Data: Scoring rules, deadlines, chip information, etc.

13. Player Ownership and Statistics

    Endpoint: https://fantasy.premierleague.com/api/stats/top/{statistic_type}/

    Description: Provides the top players in various statistical categories (e.g., most selected, most transferred in).

    Key Data: Player ownership, transfers, points, etc.

14. Transfers Market

    Endpoint: https://fantasy.premierleague.com/api/transfers/

    Description: Shows the latest transfers in the market (e.g., most transferred in/out players).

    Key Data: Player transfer data, market trends, etc.

15. Live Bonus Points System (BPS)

    Endpoint: https://fantasy.premierleague.com/api/event/{event_id}/live/

    Description: Provides live updates on the Bonus Points System (BPS) during matches for a specific gameweek.

    Key Data: BPS points, real-time match updates, etc.

16. Dream Team

    Endpoint: https://fantasy.premierleague.com/api/dream-team/{event_id}/

    Description: Provides the best-performing players for a given gameweek (event).

    Key Data: Players in the dream team, their points, and positions.

17. User Data

    Endpoint: https://fantasy.premierleague.com/api/me/

    Description: Returns personal data for the currently authenticated user (requires authentication).

    Key Data: User-specific information, leagues joined, etc.

18. Player Data (Detailed)

    Endpoint: https://fantasy.premierleague.com/api/element-summary/{element_id}/

    Description: Provides detailed data about a specific player, including match-by-match data.

    Key Data: Player performance in each gameweek, season stats, upcoming fixtures.

Notes:

    The FPL API is generally publicly accessible and doesn't require authentication for most of the endpoints.

    However, to access some personalized or sensitive data (like the /me/ endpoint), you need to be authenticated, usually requiring login and session management.

    The API response is typically in JSON format, making it easy to parse and use in various applications.

# Data Source Exploration Notes

## FPL API (bootstrap-static) — Current Season
- Top-level keys: `elements` (800+ players), `teams` (20), `events` (38 GWs), `element_types` (4 positions), `chips`, `game_settings`, `game_config`, `phases`, `element_stats`
- `elements` has ~95 fields per player including expected stats (xG, xA), ICT index, per-90 metrics, transfer data, set-piece order, pricing
- `now_cost` is in 0.1m units (e.g., 145 = £14.5m)
- `element_type`: 1=GK, 2=DEF, 3=MID, 4=FWD
- Per-GW live data requires separate `/api/event/{id}/live/` calls
- Per-player history requires `/api/element-summary/{id}/` calls

## vaastav/Fantasy-Premier-League GitHub Repo — Historical Data
- 10 seasons: 2016-17 through 2025-26
- Per season: `players_raw.csv` (mirrors bootstrap-static elements), `cleaned_players.csv` (19-col subset), `fixtures.csv`, `teams.csv`, `player_idlist.csv`
- Per-GW: `gws/gw{N}.csv` (~41 cols, pre-joined with player name/position/team), `gws/merged_gw.csv`
- Per-player: `players/{Name}_{ID}/gw.csv` and `history.csv`
- Understat data (2019-20+): match-level xG, xA, npxG, xGChain, xGBuildup
- Cross-season: `cleaned_merged_seasons.csv`, `master_team_list.csv`
- Schema evolves across seasons: older seasons lack expected stats, have granular pass/tackle/dribble data; newer add `starts`, manager points

## Key ETL Challenges
- Schema evolution: columns differ across seasons (2016-17 has ~57 cols, 2024-25 has ~95)
- Type differences: `element_type` is int in raw data, text in cleaned data
- ID mapping: player IDs are not stable across seasons; need `code`/`opta_code` for cross-season linking
- Team IDs change across seasons; `master_team_list.csv` provides mapping
- GW CSVs are pre-joined (include name/team text); API data requires joining
- Understat data needs separate ID mapping (`id_dict.csv`)

# currentDate
Today's date is 2026-03-08.
