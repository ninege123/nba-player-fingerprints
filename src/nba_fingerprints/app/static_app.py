"""Generate a static interactive app from app-facing processed tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("site/app")


def generate_static_app(
    season: str = "2023-24",
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write a self-contained static HTML app and return its path."""
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    season_key = season.replace("-", "_")
    profiles = _read_csv(processed_path / f"app_player_profiles_{season_key}.csv")
    edges = _read_csv(processed_path / f"app_similarity_edges_{season_key}.csv")
    metadata = _read_csv(processed_path / f"app_feature_metadata_{season_key}.csv")

    payload = {
        "season": season,
        "profiles": _records(profiles),
        "edges": _records(edges),
        "metadata": _records(metadata),
    }
    app_path = output_path / "index.html"
    app_path.write_text(_render_html(payload), encoding="utf-8")
    return app_path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing app data file: {path}")
    return pd.read_csv(path)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _render_html(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NBA Player Fingerprints App</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #64748b;
      --line: #d9e0ea;
      --accent: #0f766e;
      --accent-2: #b45309;
      --accent-3: #2563eb;
      --danger: #be123c;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }}
    button, input, select {{ font: inherit; }}
    .shell {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
    }}
    .sidebar {{
      border-right: 1px solid var(--line);
      background: #fbfcfe;
      padding: 18px;
      overflow: auto;
      max-height: 100vh;
    }}
    .brand {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 16px;
    }}
    .brand h1 {{
      margin: 0;
      font-size: 20px;
      line-height: 1.1;
      letter-spacing: 0;
    }}
    .season {{
      color: var(--muted);
      font-weight: 700;
      font-size: 12px;
    }}
    .control {{
      display: grid;
      gap: 6px;
      margin-bottom: 12px;
    }}
    .control label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .control input, .control select {{
      width: 100%;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 6px;
      padding: 9px 10px;
      outline: none;
    }}
    .player-list {{
      display: grid;
      gap: 8px;
      margin-top: 16px;
    }}
    .player-button {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      padding: 10px;
      text-align: left;
      cursor: pointer;
      box-shadow: var(--shadow);
    }}
    .player-button.active {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.15);
    }}
    .player-name {{
      font-weight: 800;
      display: block;
    }}
    .player-meta {{
      color: var(--muted);
      font-size: 12px;
      display: block;
      margin-top: 2px;
    }}
    .main {{
      overflow: auto;
      max-height: 100vh;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(247, 248, 251, 0.95);
      border-bottom: 1px solid var(--line);
      padding: 18px 24px;
      backdrop-filter: blur(10px);
    }}
    .topbar h2 {{
      margin: 0 0 4px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .subline {{
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 8px;
      border-radius: 999px;
      background: #e8f3f1;
      color: #0f5f59;
      font-size: 12px;
      font-weight: 700;
    }}
    .content {{
      padding: 22px 24px 32px;
      display: grid;
      gap: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
      min-width: 0;
    }}
    .span-4 {{ grid-column: span 4; }}
    .span-5 {{ grid-column: span 5; }}
    .span-6 {{ grid-column: span 6; }}
    .span-7 {{ grid-column: span 7; }}
    .span-8 {{ grid-column: span 8; }}
    .span-12 {{ grid-column: span 12; }}
    .panel h3 {{
      margin: 0 0 12px;
      font-size: 15px;
      letter-spacing: 0;
    }}
    .metric-row {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
      min-height: 76px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      min-height: 34px;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 850;
      line-height: 1;
    }}
    .bar-list {{
      display: grid;
      gap: 9px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(150px, 210px) minmax(0, 1fr) 64px;
      gap: 10px;
      align-items: center;
    }}
    .bar-label {{
      color: var(--ink);
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .track {{
      height: 10px;
      border-radius: 999px;
      background: #e7edf5;
      overflow: hidden;
    }}
    .fill {{
      height: 100%;
      width: 0%;
      background: var(--accent);
      border-radius: inherit;
    }}
    .fill.orange {{ background: var(--accent-2); }}
    .fill.blue {{ background: var(--accent-3); }}
    .bar-value {{
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      text-align: right;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .coverage {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .badge {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 8px;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      background: #fff;
    }}
    .badge.ok {{
      border-color: #99d8ce;
      color: #0f766e;
      background: #edf9f6;
    }}
    .badge.missing {{
      border-color: #f3b2c2;
      color: var(--danger);
      background: #fff1f4;
    }}
    .scatter {{
      width: 100%;
      height: 320px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
    }}
    .note {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 10px;
    }}
    @media (max-width: 980px) {{
      .shell {{ grid-template-columns: 1fr; }}
      .sidebar {{ max-height: none; border-right: 0; border-bottom: 1px solid var(--line); }}
      .main {{ max-height: none; }}
      .span-4, .span-5, .span-6, .span-7, .span-8 {{ grid-column: span 12; }}
      .metric-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .bar-row {{ grid-template-columns: minmax(120px, 180px) minmax(0, 1fr) 56px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">
        <h1>NBA Player Fingerprints</h1>
        <span class="season" id="seasonLabel"></span>
      </div>
      <div class="control">
        <label for="search">Search player</label>
        <input id="search" type="search" placeholder="Stephen Curry">
      </div>
      <div class="control">
        <label for="positionFilter">Position</label>
        <select id="positionFilter"></select>
      </div>
      <div class="control">
        <label for="archetypeFilter">Archetype</label>
        <select id="archetypeFilter"></select>
      </div>
      <div id="playerList" class="player-list"></div>
    </aside>
    <main class="main">
      <header class="topbar">
        <h2 id="playerTitle"></h2>
        <div class="subline" id="playerSubline"></div>
      </header>
      <section class="content">
        <div class="grid">
          <section class="panel span-12">
            <h3>Profile</h3>
            <div class="metric-row" id="profileMetrics"></div>
          </section>
          <section class="panel span-7">
            <h3>Fingerprint Support</h3>
            <div class="bar-list" id="supportBars"></div>
            <div class="note">Highest player dimensions from the normalized fingerprint. These are the features most visibly supporting the profile shape.</div>
          </section>
          <section class="panel span-5">
            <h3>Player Labels</h3>
            <div id="labelsBlock"></div>
          </section>
          <section class="panel span-6">
            <h3>Nearest Neighbors</h3>
            <div id="neighborsTable"></div>
          </section>
          <section class="panel span-6">
            <h3>Typicality vs Peers</h3>
            <div class="bar-list" id="typicalityBars"></div>
            <div class="note">Peer typicality is percentile rank of similarity to the broad position reference within the same primary position.</div>
          </section>
          <section class="panel span-6">
            <h3>Scoring Style</h3>
            <div class="bar-list" id="styleBars"></div>
          </section>
          <section class="panel span-6">
            <h3>Shot Creation Map</h3>
            <svg id="scatter" class="scatter" role="img" aria-label="Catch-and-shoot versus pull-up scatter plot"></svg>
          </section>
          <section class="panel span-12">
            <h3>Source Coverage</h3>
            <div class="coverage" id="coverageBadges"></div>
          </section>
        </div>
      </section>
    </main>
  </div>
  <script>
    const APP_DATA = {payload_json};
  </script>
  <script>
    const profiles = APP_DATA.profiles;
    const edges = APP_DATA.edges;
    const metadata = APP_DATA.metadata;
    const byPlayerId = new Map(profiles.map(player => [String(player.player_id), player]));
    const metadataByFeature = new Map(metadata.map(row => [row.feature, row]));
    let selectedPlayer = profiles.find(player => player.player_name === "Stephen Curry") || profiles[0];

    const search = document.getElementById("search");
    const positionFilter = document.getElementById("positionFilter");
    const archetypeFilter = document.getElementById("archetypeFilter");
    const playerList = document.getElementById("playerList");

    document.getElementById("seasonLabel").textContent = APP_DATA.season;

    function number(value) {{
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : 0;
    }}

    function pct(value, digits = 0) {{
      return `${{(100 * number(value)).toFixed(digits)}}%`;
    }}

    function dec(value, digits = 3) {{
      return number(value).toFixed(digits);
    }}

    function label(feature) {{
      return metadataByFeature.get(feature)?.label || feature.replaceAll("_", " ");
    }}

    function uniqueSorted(values) {{
      return [...new Set(values.filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b)));
    }}

    function populateFilters() {{
      const positions = uniqueSorted(profiles.map(player => player.primary_position));
      const archetypes = uniqueSorted(profiles.map(player => player.top_archetype));
      positionFilter.innerHTML = `<option value="">All positions</option>` + positions.map(value => `<option value="${{value}}">${{value}}</option>`).join("");
      archetypeFilter.innerHTML = `<option value="">All archetypes</option>` + archetypes.map(value => `<option value="${{value}}">${{value.replaceAll("_", " ")}}</option>`).join("");
    }}

    function filteredPlayers() {{
      const query = search.value.trim().toLowerCase();
      const position = positionFilter.value;
      const archetype = archetypeFilter.value;
      return profiles
        .filter(player => !query || player.player_name.toLowerCase().includes(query))
        .filter(player => !position || player.primary_position === position)
        .filter(player => !archetype || player.top_archetype === archetype)
        .slice(0, 80);
    }}

    function renderPlayerList() {{
      const rows = filteredPlayers();
      playerList.innerHTML = rows.map(player => `
        <button class="player-button ${{player.player_id === selectedPlayer.player_id ? "active" : ""}}" data-player-id="${{player.player_id}}">
          <span class="player-name">${{player.player_name}}</span>
          <span class="player-meta">${{player.team_abbreviation}} · ${{player.primary_position || "NA"}} · ${{String(player.top_archetype || "").replaceAll("_", " ")}}</span>
        </button>
      `).join("");
      playerList.querySelectorAll("button").forEach(button => {{
        button.addEventListener("click", () => {{
          selectedPlayer = byPlayerId.get(button.dataset.playerId);
          render();
        }});
      }});
    }}

    function metric(labelText, valueText) {{
      return `<div class="metric"><div class="metric-label">${{labelText}}</div><div class="metric-value">${{valueText}}</div></div>`;
    }}

    function renderHeader(player) {{
      document.getElementById("playerTitle").textContent = player.player_name;
      document.getElementById("playerSubline").innerHTML = `
        <span>${{player.team_abbreviation}}</span>
        <span>${{player.position || "NA"}}</span>
        <span>${{number(player.minutes_per_game).toFixed(1)}} MPG</span>
        <span class="pill">${{String(player.top_archetype || "").replaceAll("_", " ")}}</span>
      `;
    }}

    function renderProfileMetrics(player) {{
      document.getElementById("profileMetrics").innerHTML = [
        metric("Points per 36", dec(player.points_per_36, 1)),
        metric("Usage", pct(player.usage_rate, 1)),
        metric("True shooting", pct(player.true_shooting_pct, 1)),
        metric("Peer typicality", pct(player.peer_typicality_percentile, 0)),
      ].join("");
    }}

    function renderLabels(player) {{
      document.getElementById("labelsBlock").innerHTML = `
        <table>
          <tbody>
            <tr><th>Top archetype</th><td>${{String(player.top_archetype || "").replaceAll("_", " ")}}<br><span class="player-meta">${{pct(player.top_archetype_similarity, 1)}} similarity</span></td></tr>
            <tr><th>Position reference</th><td>${{player.top_position_reference || "NA"}}<br><span class="player-meta">${{pct(player.top_position_similarity, 1)}} similarity</span></td></tr>
            <tr><th>Style summary</th><td>${{player.style_summary || "No style label available"}}</td></tr>
            <tr><th>Support</th><td>${{player.supporting_features || "NA"}}</td></tr>
            <tr><th>Gaps</th><td>${{player.gap_features || "NA"}}</td></tr>
          </tbody>
        </table>
      `;
    }}

    function normalizedFeatureRows(player, limit = 8) {{
      return metadata
        .filter(row => row.is_fingerprint_dimension === true || row.is_fingerprint_dimension === "True")
        .map(row => ({{ feature: row.feature, label: row.label, value: number(player[row.feature]), group: row.group }}))
        .filter(row => row.value > 0)
        .sort((a, b) => b.value - a.value)
        .slice(0, limit);
    }}

    function barRows(rows, color = "") {{
      return rows.map(row => `
        <div class="bar-row">
          <div class="bar-label">${{row.label}}</div>
          <div class="track"><div class="fill ${{color}}" style="width:${{Math.max(0, Math.min(100, 100 * row.value))}}%"></div></div>
          <div class="bar-value">${{row.format || pct(row.value, 0)}}</div>
        </div>
      `).join("");
    }}

    function renderSupport(player) {{
      document.getElementById("supportBars").innerHTML = barRows(normalizedFeatureRows(player), "blue");
    }}

    function renderNeighbors(player) {{
      const playerEdges = edges
        .filter(edge => String(edge.player_id) === String(player.player_id))
        .sort((a, b) => number(a.neighbor_rank) - number(b.neighbor_rank))
        .slice(0, 5);
      document.getElementById("neighborsTable").innerHTML = `
        <table>
          <thead><tr><th>Rank</th><th>Player</th><th>Team</th><th>Similarity</th></tr></thead>
          <tbody>
            ${{playerEdges.map(edge => `
              <tr>
                <td>${{edge.neighbor_rank}}</td>
                <td>${{edge.neighbor_player_name}}</td>
                <td>${{edge.neighbor_team_abbreviation}}</td>
                <td>${{pct(edge.cosine_similarity, 1)}}</td>
              </tr>
            `).join("")}}
          </tbody>
        </table>
      `;
    }}

    function renderTypicality(player) {{
      const rows = [
        {{ label: "Peer typicality percentile", value: number(player.peer_typicality_percentile), format: pct(player.peer_typicality_percentile, 0) }},
        {{ label: "Position reference similarity", value: number(player.top_position_similarity), format: pct(player.top_position_similarity, 1) }},
        {{ label: "Top neighbor similarity", value: number(player.top_neighbor_similarity), format: pct(player.top_neighbor_similarity, 1) }},
        {{ label: "Distinctiveness", value: number(player.distinctiveness_score), format: pct(player.distinctiveness_score, 1) }},
      ];
      document.getElementById("typicalityBars").innerHTML = barRows(rows, "orange");
    }}

    function renderStyle(player) {{
      const rows = [
        "isolation_frequency",
        "pnr_ball_handler_frequency",
        "spot_up_frequency",
        "off_screen_frequency",
        "cut_frequency",
        "post_up_frequency",
        "transition_frequency",
        "catch_shoot_attempt_frequency",
        "pull_up_attempt_frequency",
        "rim_attempt_rate",
      ].map(feature => ({{ label: label(feature), value: number(player[feature]), format: pct(player[feature], 1) }}))
       .sort((a, b) => b.value - a.value)
       .slice(0, 8);
      document.getElementById("styleBars").innerHTML = barRows(rows);
    }}

    function renderCoverage(player) {{
      const flags = [
        ["has_synergy_data", "Synergy"],
        ["has_shot_location_data", "Shot locations"],
        ["has_tracking_shot_data", "Tracking shots"],
        ["has_touch_tracking_data", "Touch tracking"],
      ];
      document.getElementById("coverageBadges").innerHTML = flags.map(([feature, text]) => {{
        const ok = player[feature] === true || player[feature] === "True";
        return `<span class="badge ${{ok ? "ok" : "missing"}}">${{text}}: ${{ok ? "available" : "missing"}}</span>`;
      }}).join("");
    }}

    function renderScatter(player) {{
      const svg = document.getElementById("scatter");
      const width = svg.clientWidth || 500;
      const height = svg.clientHeight || 320;
      const pad = 42;
      const maxX = Math.max(...profiles.map(p => number(p.catch_shoot_attempt_frequency)), 0.01);
      const maxY = Math.max(...profiles.map(p => number(p.pull_up_attempt_frequency)), 0.01);
      const x = value => pad + (width - pad * 1.5) * number(value) / maxX;
      const y = value => height - pad - (height - pad * 1.6) * number(value) / maxY;
      const points = profiles.map(p => {{
        const selected = String(p.player_id) === String(player.player_id);
        return `<circle cx="${{x(p.catch_shoot_attempt_frequency)}}" cy="${{y(p.pull_up_attempt_frequency)}}" r="${{selected ? 6 : 3}}" fill="${{selected ? "#b45309" : "#2563eb"}}" opacity="${{selected ? 1 : 0.32}}"></circle>`;
      }}).join("");
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = `
        <line x1="${{pad}}" y1="${{height - pad}}" x2="${{width - pad / 2}}" y2="${{height - pad}}" stroke="#94a3b8"></line>
        <line x1="${{pad}}" y1="${{height - pad}}" x2="${{pad}}" y2="${{pad / 2}}" stroke="#94a3b8"></line>
        ${{points}}
        <text x="${{pad}}" y="${{height - 10}}" font-size="12" fill="#64748b">Catch-and-shoot</text>
        <text x="12" y="${{pad}}" font-size="12" fill="#64748b" transform="rotate(-90 12 ${{pad}})">Pull-up</text>
        <text x="${{x(player.catch_shoot_attempt_frequency) + 8}}" y="${{y(player.pull_up_attempt_frequency) - 8}}" font-size="12" fill="#17202a" font-weight="700">${{player.player_name}}</text>
      `;
    }}

    function render() {{
      renderPlayerList();
      renderHeader(selectedPlayer);
      renderProfileMetrics(selectedPlayer);
      renderLabels(selectedPlayer);
      renderSupport(selectedPlayer);
      renderNeighbors(selectedPlayer);
      renderTypicality(selectedPlayer);
      renderStyle(selectedPlayer);
      renderCoverage(selectedPlayer);
      renderScatter(selectedPlayer);
    }}

    search.addEventListener("input", renderPlayerList);
    positionFilter.addEventListener("change", renderPlayerList);
    archetypeFilter.addEventListener("change", renderPlayerList);
    window.addEventListener("resize", () => renderScatter(selectedPlayer));

    populateFilters();
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the static NBA player fingerprint app")
    parser.add_argument("--season", default="2023-24")
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    app_path = generate_static_app(
        season=args.season,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
    )
    print(f"app: {app_path}")


if __name__ == "__main__":
    main()
