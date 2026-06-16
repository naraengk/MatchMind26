    const state = { bootstrap: null, tournament: null, currentPage: "welcome" };
    const pct = value => `${Number(value || 0).toFixed(1)}%`;
    const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const ratingColor = value => {
      const n = Number(value || 0);
      const hue = Math.max(45, Math.min(138, 45 + (n - 60) * 3.1));
      return `hsl(${hue} 78% 62%)`;
    };
    const clubInitials = club => {
      const ignored = new Set(["fc", "cf", "sc", "afc", "ac", "club", "de", "the"]);
      const words = String(club || "Free Agent").replace(/\([^)]*\)$/, "").match(/[A-Za-z0-9]+/g) || [];
      const useful = words.filter(word => !ignored.has(word.toLowerCase()));
      const source = useful.length ? useful : words;
      return source.slice(0, 2).map(word => word[0]).join("").toUpperCase() || "FA";
    };
    const clubColor = club => {
      const palette = ["#8f2d3e", "#235789", "#2e7d5b", "#a26122", "#704c9f", "#1f7a8c", "#a33f20", "#527a2b"];
      const hash = Array.from(String(club || "Free Agent")).reduce((sum, char) => ((sum * 31) + char.charCodeAt(0)) >>> 0, 7);
      return palette[hash % palette.length];
    };
    const teamFlag = (team, flag) => {
      if (flag === "flag-england" || team === "England") return `<span class="nation-flag flag-england" aria-label="England flag"></span>`;
      if (flag === "flag-scotland" || team === "Scotland") return `<span class="nation-flag flag-scotland" aria-label="Scotland flag"></span>`;
      return `<span aria-hidden="true">${esc(flag || "🏳️")}</span>`;
    };
    const teamWithFlag = (team, flag) => `<span class="team-with-flag">${teamFlag(team, flag)}<span>${esc(team)}</span></span>`;

    function table(rows, columns) {
      if (!rows || rows.length === 0) return `<p>No rows available.</p>`;
      return `<div class="table-wrap"><table><thead><tr>${columns.map(c => `<th>${esc(c.label)}</th>`).join("")}</tr></thead><tbody>${
        rows.map(row => `<tr>${columns.map(c => `<td>${c.render ? c.render(row[c.key], row) : esc(row[c.key])}</td>`).join("")}</tr>`).join("")
      }</tbody></table></div>`;
    }
    const resultBadge = result => {
      const value = String(result || "").toUpperCase();
      const kind = value === "W" ? "win" : value === "D" ? "draw" : "loss";
      return `<span class="result-badge result-${kind}" aria-label="${kind}">${esc(value)}</span>`;
    };
    const versusCell = row => `<span class="versus-cell">${teamWithFlag(row.home_team, row.home_flag)}<span class="versus-score">${esc(row.home_score)}-${esc(row.away_score)}</span>${teamWithFlag(row.away_team, row.away_flag)}</span>`;
    const venueCell = row => `<span class="venue-cell"><strong>${esc(row.city)}</strong><small>${esc(row.stadium)}</small></span>`;
    const recentFormPanel = team => {
      const wins = team.recent.filter(match => match.result === "W").length;
      const draws = team.recent.filter(match => match.result === "D").length;
      const losses = team.recent.filter(match => match.result === "L").length;
      const points = wins * 3 + draws;
      const rows = team.recent.map(match => `
        <div class="recent-match-row">
          <span class="recent-date">${esc(match.date)}</span>
          <div class="recent-opponent">${teamWithFlag(match.opponent, match.opponent_flag)}<span class="recent-competition">${esc(match.competition)}</span></div>
          ${resultBadge(match.result)}
          <span class="recent-score">${esc(match.score)}</span>
        </div>`).join("");
      return `<article class="panel analysis-team-panel">
        <div class="analysis-team-head"><h3>${teamWithFlag(team.name, team.flag)} Recent Form</h3><span>Last ${team.recent.length} matches</span></div>
        <div class="form-summary">
          <div><span>Record</span><strong>${wins}-${draws}-${losses}</strong></div>
          <div><span>Points</span><strong>${points}</strong></div>
          <div><span>PPM</span><strong>${team.recent.length ? (points / team.recent.length).toFixed(2) : "0.00"}</strong></div>
        </div>
        <div class="recent-match-list">${rows}</div>
      </article>`;
    };
    const emptyHeadToHead = () => `<div class="analysis-empty"><span class="analysis-empty-icon">-</span><div><strong>No recent meetings</strong><span>There is no head-to-head match in the project’s ten-year dataset.</span></div></div>`;

    function setPage(page) {
      state.currentPage = page;
      document.body.classList.remove("page-welcome", "page-group", "page-knockout");
      document.body.classList.add(`page-${page}`);
      document.querySelectorAll(".page").forEach(el => el.classList.add("hidden"));
      document.getElementById(page).classList.remove("hidden");
      document.querySelectorAll(".tab").forEach(btn => {
        const active = btn.dataset.page === page;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-current", active ? "page" : "false");
        btn.innerHTML = `<span>${esc(btn.dataset.label)}</span>${active ? "<small>Current</small>" : ""}`;
      });
      if (page === "welcome") renderWelcome();
      if (page === "group") renderGroup();
      if (page === "knockout") renderKnockout();
    }

    function renderWelcome() {
      const facts = state.bootstrap.facts;
      document.getElementById("welcome").innerHTML = `
        <section class="hero">
          <div class="hero-copy">
            <div class="kicker">FIFA World Cup 2026</div>
            <h1>WC 2026<br><span class="accent">match predictor</span></h1>
            <p>Explore every scheduled group-stage match, see the reasoning behind each prediction, and follow the full route to the World Cup final.</p>
            <div class="pill-row">
              <span class="pill">${esc(facts.hosts)}</span>
              <span class="pill">${esc(facts.dates)}</span>
            </div>
            <div class="hero-actions">
              <button class="primary-action" onclick="setPage('group')">Analyze a Match</button>
              <button class="secondary-action" onclick="setPage('knockout')">Open Tournament Projection</button>
            </div>
          </div>
          <aside class="hero-console">
            <div class="console-head">
              <div><div class="kicker" style="margin:0 0 5px">Prediction model</div><strong>World Cup 2026</strong></div>
              <span class="live-status">Ready</span>
            </div>
            <div class="tournament-map">
              <div class="map-line"><span class="map-number">01</span><div><strong>Group Stage</strong><small>${facts.fixtures} scheduled fixtures</small></div><span class="map-value">48</span></div>
              <div class="map-line"><span class="map-number">02</span><div><strong>Qualification</strong><small>Top two plus best third-place teams</small></div><span class="map-value">32</span></div>
              <div class="map-line"><span class="map-number">03</span><div><strong>Knockout Path</strong><small>Five rounds to the title</small></div><span class="map-value">1</span></div>
            </div>
            <p>Historical performance, current squad quality, recent form, club strength, experience, and venue advantage are combined into each projection.</p>
          </aside>
        </section>
        <section class="welcome-band">
          <div class="welcome-stat"><span>Teams</span><b>${facts.teams}</b></div>
          <div class="welcome-stat"><span>Groups</span><b>${facts.groups}</b></div>
          <div class="welcome-stat"><span>Group Matches</span><b>${facts.fixtures}</b></div>
          <div class="welcome-stat"><span>Training Matches</span><b>${Number(facts.training_matches || 0).toLocaleString()}</b></div>
        </section>
        <div class="section-title"><div class="kicker">How To Explore</div><h2>One model, two prediction views</h2></div>
        <section class="workflow-grid">
          <article class="workflow-card"><div class="workflow-index">01 / FIXTURE REPORT</div><h3>Group Stage Analysis</h3><p>Choose an official fixture to compare probabilities, Elo ratings, recent form, managers, squad depth, player ratings, and venue advantage.</p></article>
          <article class="workflow-card"><div class="workflow-index">02 / TOURNAMENT SIMULATION</div><h3>Knockout Projection</h3><p>Generate expected group standings, identify the 32 qualifiers, and follow projected scorelines and confidence through the final.</p></article>
          <article class="workflow-card"><div class="workflow-index">03 / EXPLAINABILITY</div><h3>Evidence Behind Results</h3><p>See all eight signals behind a prediction, how much each one counts, how it is measured, and which team has the edge.</p></article>
        </section>`;
    }

    function renderGroup() {
      const groups = state.bootstrap.groups;
      const fixtures = state.bootstrap.fixtures;
      document.getElementById("group").innerHTML = `
        <section class="stage-landing group-landing">
          <div class="group-landing-copy">
            <div class="stage-eyebrow">Group-stage fixtures / 72 scheduled matches</div>
            <h1>Group Stage</h1>
            <p>Choose an official fixture and open a complete pre-match report built from team history, current form, squad quality, managers, and venue advantage.</p>
            <div class="stage-tags"><span class="stage-tag">12 groups</span><span class="stage-tag">48 teams</span><span class="stage-tag">8 model signals</span><span class="stage-tag">26-player squads</span></div>
          </div>
          <div class="analysis-map">
            <div class="analysis-step"><span class="analysis-step-index">01</span><div><strong>Select a fixture</strong><small>Official group schedule only</small></div><span class="analysis-step-value">72</span></div>
            <div class="analysis-step"><span class="analysis-step-index">02</span><div><strong>Compare the teams</strong><small>Form, Elo, squads, and venue</small></div><span class="analysis-step-value">8 signals</span></div>
            <div class="analysis-step"><span class="analysis-step-index">03</span><div><strong>Read the forecast</strong><small>Probabilities with full reasoning</small></div><span class="analysis-step-value">3 outcomes</span></div>
          </div>
        </section>
        <section class="selectors">
          <div><label>Group</label><select id="groupSelect">${groups.map(g => `<option>${esc(g)}</option>`).join("")}</select></div>
          <div><label>Scheduled Group-Stage Match</label><select id="fixtureSelect"><option value="">Select a match</option></select></div>
        </section>
        <div id="matchReport" class="tool-intro"><h1>Select a scheduled match</h1><p>Choose a fixture above to see the prediction, squad ratings, team stats, recent form, the head-to-head record, and the reasoning behind it.</p></div>`;
      const groupSelect = document.getElementById("groupSelect");
      const fixtureSelect = document.getElementById("fixtureSelect");
      const syncFixtures = () => {
        const group = groupSelect.value;
        const visible = fixtures.filter(f => group === "All" || f.group === group);
        fixtureSelect.innerHTML = `<option value="">Select a match</option>` + visible.map(f => `<option value="${f.fixture_id}">${esc(f.label)}</option>`).join("");
      };
      groupSelect.addEventListener("change", syncFixtures);
      fixtureSelect.addEventListener("change", () => fixtureSelect.value && loadMatch(fixtureSelect.value));
      syncFixtures();
    }

    async function loadMatch(id) {
      const report = document.getElementById("matchReport");
      report.innerHTML = `<p>Loading match report...</p>`;
      let data;
      try {
        const response = await fetch(`/api/match/${id}`);
        if (!response.ok) throw new Error(`Match service returned ${response.status}`);
        data = await response.json();
      } catch (error) {
        report.innerHTML = `<h1>Match report could not load</h1><p>${esc(error.message)}. Confirm the FastAPI server is running, then choose the fixture again.</p>`;
        return;
      }
      const p = data.prediction;
      const [a, b] = data.teams;
      report.className = "";
      report.innerHTML = `
        <section class="panel">
          <div class="kicker">Group ${esc(data.fixture.group)} / ${esc(data.fixture.city)}</div>
          <div class="prediction-title">${esc(p.title)}</div>
          <p>${esc(p.summary)}</p>
          <div class="prob-grid">
            ${probCard(`${a.name} Win`, p.home_win_pct)}
            ${probCard("Draw", p.draw_pct)}
            ${probCard(`${b.name} Win`, p.away_win_pct)}
          </div>
        </section>
        <h2 class="section-title">Team Rooms</h2>
        <section class="team-grid">${teamRoom(a)}${teamRoom(b)}</section>
        <h2 class="section-title">Why This Outcome?</h2>
        <section class="evidence-grid">${data.evidence.map(evidenceCard).join("")}</section>
        <h2 class="section-title">Official 26-Player Squads</h2>
        <section class="team-grid">${squadPanel(a)}${squadPanel(b)}</section>
        <section class="deep-analysis-shell">
          <div class="analysis-heading">
            <div><div class="kicker">Team comparison</div><h2>Deep Analysis</h2></div>
            <p>Compare short-term momentum, direct meetings, and every fixture that can shape the final group table.</p>
          </div>
          <div class="team-grid">
            ${recentFormPanel(a)}
            ${recentFormPanel(b)}
          </div>
          <section class="panel analysis-detail-panel">
            <div class="analysis-block">
              <div class="analysis-block-title"><h3>Head-to-Head</h3><span>${data.head_to_head.length} recorded meeting${data.head_to_head.length === 1 ? "" : "s"}</span></div>
              <div class="analysis-table">${data.head_to_head.length ? table(data.head_to_head, [
                {key:"date", label:"Date"}, {key:"match", label:"Match", render:(value,row) => versusCell(row)}, {key:"competition", label:"Competition"}
              ]) : emptyHeadToHead()}</div>
            </div>
            <div class="analysis-block">
              <div class="analysis-block-title"><h3>Group Schedule</h3><span>Complete Group ${esc(data.fixture.group)} fixture list</span></div>
              <div class="analysis-table schedule-table">${table(data.group_schedule, [
                {key:"date", label:"Date"}, {key:"team_one", label:"Team One", render:(value,row) => teamWithFlag(value, row.team_one_flag)}, {key:"team_two", label:"Team Two", render:(value,row) => teamWithFlag(value, row.team_two_flag)}, {key:"city", label:"Venue", render:(value,row) => venueCell(row)}
              ])}</div>
            </div>
          </section>
        </section>
        ${modelEvaluation(data.metrics)}`;
    }

    function modelEvaluation(metrics) {
      if (!metrics || !metrics.split) return "";
      const metric = (label, value, note) => `<div class="evaluation-metric"><span>${esc(label)}</span><strong>${value}</strong><small>${esc(note)}</small></div>`;
      return `<section class="panel evaluation-panel">
        <div class="evaluation-head">
          <div><div class="kicker">Held-out test results</div><h3>Model Evaluation</h3></div>
          <p>These results use matches from 2025 onward. The model and draw threshold were selected using earlier seasons only.</p>
        </div>
        <div class="evaluation-grid">
          ${metric("Accuracy", pct(Number(metrics.accuracy) * 100), "All three outcomes")}
          ${metric("Draw Recall", pct(Number(metrics.draw_recall) * 100), "Actual draws detected")}
          ${metric("Macro F1", Number(metrics.macro_f1).toFixed(3), "Balanced class quality")}
          ${metric("Baseline", pct(Number(metrics.baseline_accuracy) * 100), "Always predict majority")}
          ${metric("Calibration Error", pct(Number(metrics.expected_calibration_error) * 100), "Lower is better")}
        </div>
        <p class="evaluation-note">${esc(metrics.squad_adjustment || "")} Test sample: ${esc(metrics.sample_count)} matches.</p>
      </section>`;
    }

    function probCard(label, value) {
      return `<div class="prob"><span>${esc(label)}</span><b>${pct(value)}</b><div class="prob-track"><div class="prob-fill" style="width:${value}%"></div></div></div>`;
    }

    function teamRoom(team) {
      const r = team.record, m = team.metadata;
      return `<article class="team-card">
        <div class="kicker">Group ${esc(m.group)} / ${esc(m.confederation)}</div>
        <h2>${teamWithFlag(team.name, team.flag)}</h2>
        <div class="manager">Manager: ${esc(m.manager)}</div>
        <p>${esc(m.scouting_note)}</p>
        <div class="stat-grid">
          <div class="stat"><span>Elo</span><b>${team.profile.elo}</b></div>
          <div class="stat"><span>PPM</span><b>${r.points_per_match}</b></div>
          <div class="stat"><span>GD/Match</span><b>${r.goal_diff_per_match > 0 ? "+" : ""}${r.goal_diff_per_match}</b></div>
          <div class="stat"><span>W-D-L</span><b>${r.wins}-${r.draws}-${r.losses}</b></div>
        </div>
      </article>`;
    }

    function evidenceCard(row) {
      return `<article class="panel evidence">
        <span class="weight">${row.weight}% decision weight</span>
        <h3>${esc(row.signal)}</h3>
        <p>${row.edge === "Even" ? "Both teams are even here." : `<b>${esc(row.edge)}</b> has the edge here.`}</p>
        <div class="stat-grid" style="grid-template-columns:1fr 1fr">
          <div class="stat"><span>Team One</span><b>${row.team_one}</b></div>
          <div class="stat"><span>Team Two</span><b>${row.team_two}</b></div>
        </div>
        <p>${esc(row.formula)}<br>${esc(row.scale)}</p>
      </article>`;
    }

    function squadPanel(team) {
      const rating = value => `<span class="rating" style="background:${ratingColor(value)}">${esc(value)}</span>`;
      const playerRow = (player, core = false) => {
        const number = core ? player.position : `#${String(player.shirt_number).padStart(2, "0")}`;
        const playerRating = core ? player.overall_rating : player.player_rating;
        const clubRating = core ? player.club_strength : player.club_rating;
        const detail = `${player.position} / ${player.age} years / ${player.club}`;
        return `<div class="player-row">
          <span class="shirt-number">${esc(number)}</span>
          <span class="club-badge" title="${esc(player.club)}" aria-label="${esc(player.club)}" style="background:${clubColor(player.club)}">${clubInitials(player.club)}</span>
          <div class="player-copy"><strong>${esc(player.player)}</strong><small>${esc(detail)}</small></div>
          <div class="player-ratings">
            <div class="mini-rating"><small>Player</small>${rating(playerRating)}</div>
            <div class="mini-rating club-score"><small>Club</small>${rating(clubRating)}</div>
          </div>
        </div>`;
      };
      return `<article class="panel">
        <section class="squad-section">
          <div class="squad-heading"><h3>${esc(team.flag)} ${esc(team.name)} Squad Core</h3><span>${team.squad_core.length} highest-rated players</span></div>
          <div class="player-list">${team.squad_core.map(player => playerRow(player, true)).join("")}</div>
        </section>
        <section class="squad-section">
          <div class="squad-heading"><h3>${esc(team.name)} Full Roster</h3><span>${team.roster.length} selected players</span></div>
          <div class="player-list">${team.roster.map(player => playerRow(player)).join("")}</div>
        </section>
      </article>`;
    }

    async function renderKnockout() {
      const root = document.getElementById("knockout");
      root.innerHTML = `<section class="tool-intro loading-shell"><div><div class="loader"></div><h2>Building Tournament Projection</h2><p>Calculating 72 group matches, qualification seeds, knockout ties, scorelines, and confidence levels.</p></div></section>`;
      let data = state.tournament;
      try {
        if (!data) {
          const response = await fetch("/api/tournament");
          if (!response.ok) throw new Error(`Projection service returned ${response.status}`);
          data = await response.json();
          if (!data.bracket || !data.groups || !data.qualifiers || !data.monte_carlo) throw new Error("Projection data is incomplete");
          state.tournament = data;
        }
      } catch (error) {
        root.innerHTML = `<section class="tool-intro loading-shell error"><div><h2>Tournament Projection Could Not Load</h2><p>${esc(error.message)}. Confirm the FastAPI server is running, then try again.</p><button class="retry-button" onclick="state.tournament=null; renderKnockout()">Retry Projection</button></div></section>`;
        return;
      }
      const byRound = name => data.bracket.filter(r => r.Round === name);
      const flagMap = {};
      data.bracket.forEach(row => {
        flagMap[row["Team One"]] = row["Team One Flag"];
        flagMap[row["Team Two"]] = row["Team Two Flag"];
        flagMap[row["Projected Winner"]] = row["Winner Flag"];
      });
      root.innerHTML = `
        <section class="stage-landing knockout-landing">
          <div class="knockout-copy">
            <div class="stage-eyebrow">Tournament Simulation / 32-Team Field</div>
            <h1>Knockout Stages</h1>
            <p>Follow the projected route from qualification through every knockout tie, with a projected scoreline and a confidence level for each one.</p>
            <div class="round-path"><span>R32</span><i></i><span>R16</span><i></i><span>QF</span><i></i><span>SF</span><i></i><span>Final</span></div>
          </div>
          <aside class="champion-feature">
            <div class="champion-feature-head"><div><span class="kicker">Projected Champion</span><small>The model's projected winner</small></div><span class="champion-seal" aria-label="Champion">♛</span></div>
            <h2>${teamWithFlag(data.champion, data.champion_flag)}</h2>
            <div class="champion-confidence"><span>Final win confidence</span><strong>${pct(data.final_confidence)}</strong></div>
          </aside>
        </section>
        <section class="stage-confidence-grid">${[
          {name:"Round of 32", ties:16},
          {name:"Round of 16", ties:8},
          {name:"Quarterfinals", ties:4},
          {name:"Semifinals", ties:2},
          {name:"Final", ties:1}
        ].map((stage, index) => {
          const row = data.round_confidence.find(item => item.Round === stage.name);
          const confidence = Number(row ? row["Win Confidence"] : 0);
          const tier = confidence >= 75 ? "Strong signal" : confidence >= 65 ? "Clear lean" : "Competitive";
          return `<article class="panel stage-card">
            <div class="stage-card-head">
              <div><span class="kicker">${esc(stage.name)}</span><div class="stage-matches">${stage.ties} projected ${stage.ties === 1 ? "tie" : "ties"}</div></div>
              <span class="stage-number">0${index + 1}</span>
            </div>
            <div>
              <div class="stage-confidence-value"><strong>${pct(confidence)}</strong><span>average confidence</span></div>
              <div class="stage-meter" aria-label="${pct(confidence)} average confidence"><span style="width:${confidence}%"></span></div>
              <div class="stage-tier"><span>Model certainty</span><b>${tier}</b></div>
            </div>
          </article>`;
        }).join("")}</section>
        ${monteCarloView(data.monte_carlo)}
        <h2 class="section-title">Projected Knockout Bracket</h2>
        ${bracketView(data.bracket)}
        <section class="panel"><h3>How This Is Calculated</h3><p>${esc(data.method)}</p></section>
        <section class="tournament-data">
          <div class="data-heading"><div><div class="kicker">Group Projection</div><h2>Projected Groups</h2></div><p>Expected points combine every scheduled fixture's win, draw, and loss probabilities.</p></div>
          <div class="groups-grid">${groupCards(data.groups, flagMap)}</div>
        </section>
        <section class="tournament-data">
          <div class="data-heading"><div><div class="kicker">Round of 32 Field</div><h2>Qualified Teams</h2></div><p>The top two in each group advance with the eight strongest third-place teams.</p></div>
          <div class="qualifier-grid">${qualifierCards(data.qualifiers, flagMap)}</div>
        </section>`;
    }

    function monteCarloView(simulation) {
      const contenders = simulation.probabilities.slice(0, 3);
      const final = simulation.most_common_final;
      const columns = [
        ["Round of 32", "Qualify"],
        ["Round of 16", "R16"],
        ["Quarterfinals", "QF"],
        ["Semifinals", "SF"],
        ["Final", "Final"],
        ["Champion", "Champion"]
      ];
      return `<section class="simulation-section">
        <div class="simulation-heading">
          <div><div class="kicker">Probability Forecast</div><h2>Monte Carlo Tournament Simulation</h2><p>The model runs the complete tournament ${Number(simulation.simulations).toLocaleString()} times instead of assuming the favorite always wins.</p></div>
          <span class="simulation-badge">${Number(simulation.simulations).toLocaleString()} simulated tournaments</span>
        </div>
        <div class="simulation-summary">
          <div class="contender-grid">${contenders.map((row, index) => `<article class="contender-card">
            <span class="contender-rank">Title contender 0${index + 1}</span>
            <h3>${teamWithFlag(row.Team, row.Flag)}</h3>
            <div class="contender-chance"><strong>${pct(row.Champion)}</strong><span>title probability</span></div>
            <div class="stage-meter"><span style="width:${row.Champion}%"></span></div>
          </article>`).join("")}</div>
          <aside class="final-matchup">
            <span class="kicker">Most Common Final</span>
            <h3>Most frequently simulated matchup</h3>
            <div class="final-team">${teamWithFlag(final["Team One"], final["Team One Flag"])}<span>Finalist</span></div>
            <div class="final-versus">versus</div>
            <div class="final-team">${teamWithFlag(final["Team Two"], final["Team Two Flag"])}<span>Finalist</span></div>
            <div class="final-frequency"><span>Appeared in simulations</span><strong>${pct(final.Probability)}</strong></div>
          </aside>
        </div>
        <div class="simulation-table"><table>
          <thead><tr><th>Rank</th><th>Team</th>${columns.map(column => `<th>${column[1]}</th>`).join("")}</tr></thead>
          <tbody>${simulation.probabilities.map((row, index) => `<tr>
            <td>${index + 1}</td><td>${teamWithFlag(row.Team, row.Flag)}</td>
            ${columns.map(column => `<td class="${column[0] === "Champion" ? "champion-probability" : ""}">${pct(row[column[0]])}</td>`).join("")}
          </tr>`).join("")}</tbody>
        </table></div>
        <div class="simulation-note"><strong>How it works:</strong> ${esc(simulation.method)} A fixed random seed keeps the displayed probabilities reproducible for this model version.</div>
      </section>`;
    }

    function bracketView(bracket) {
      const rows = round => bracket.filter(row => row.Round === round).sort((a, b) => a.Match - b.Match);
      const r32 = rows("Round of 32"), r16 = rows("Round of 16"), qf = rows("Quarterfinals"), sf = rows("Semifinals"), final = rows("Final");
      const column = (title, items, side, stage) => `<section class="bracket-column ${side} ${stage}"><div class="bracket-label">${title}</div><div class="bracket-stack">${items.map(matchCard).join("")}</div></section>`;
      const finalRow = final[0];
      const center = `<section class="bracket-center"><div class="bracket-label">Final</div><div class="finalists"><div class="finalist">${teamWithFlag(finalRow["Team One"], finalRow["Team One Flag"])}</div><div class="finalist">${teamWithFlag(finalRow["Team Two"], finalRow["Team Two Flag"])}</div></div><div class="winner-trophy"><span class="kicker">Winners</span><strong>${teamWithFlag(finalRow["Projected Winner"], finalRow["Winner Flag"])}</strong><em>${esc(finalRow["Projected Score"])}</em><div class="confidence">${pct(finalRow["Win Confidence"])} confidence</div></div></section>`;
      return `<div class="bracket-scroll"><div class="wc-bracket">
        ${column("Round of 32", r32.slice(0,8), "left", "r32")}
        ${column("R16", r16.slice(0,4), "left", "r16")}
        ${column("QF", qf.slice(0,2), "left", "qf")}
        ${column("SF", sf.slice(0,1), "left", "sf")}
        ${center}
        ${column("SF", sf.slice(1), "right", "sf")}
        ${column("QF", qf.slice(2), "right", "qf")}
        ${column("R16", r16.slice(4), "right", "r16")}
        ${column("Round of 32", r32.slice(8), "right", "r32")}
      </div></div>`;
    }

    function groupCards(groups, flagMap) {
      const names = [...new Set(groups.map(row => row.Group))].sort();
      return names.map(group => {
        const teams = groups.filter(row => row.Group === group).sort((a,b) => a["Group Rank"] - b["Group Rank"]);
        return `<article class="group-card"><div class="group-card-head"><strong>Group ${esc(group)}</strong><span>${teams.length} teams</span></div><div>${teams.map(row => `<div class="group-team ${row["Group Rank"] <= 2 ? "qualifying" : ""}"><span class="group-rank">${row["Group Rank"]}</span><strong>${teamWithFlag(row.Team, row.Flag || flagMap[row.Team] || "")}</strong><span class="team-points">${Number(row["Expected Points"]).toFixed(2)} pts</span><span class="team-elo">${Math.round(row.Elo)} Elo</span></div>`).join("")}</div></article>`;
      }).join("");
    }

    function qualifierCards(qualifiers, flagMap) {
      return qualifiers.map(row => `<article class="qualifier-card"><div class="qualifier-top"><span class="seed">SEED ${row.Seed}</span><span>Group ${esc(row.Group)} / #${row["Group Rank"]}</span></div><div class="qualifier-team">${teamWithFlag(row.Team, row.Flag || flagMap[row.Team] || "")}</div><div class="qualifier-meta"><span>${Number(row["Expected Points"]).toFixed(2)} pts</span><span>${Math.round(row.Elo)} Elo</span></div><span class="path-tag">${esc(row.Path)}</span></article>`).join("");
    }

    function matchCard(row) {
      const teamLine = (team, flag, score, winner) => `<div class="team-line ${winner ? "winner" : ""}">${teamWithFlag(team, flag)}<b>${esc(score)}</b></div>`;
      const [s1, s2] = String(row["Projected Score"]).split("-");
      return `<article class="match-card">
        ${teamLine(row["Team One"], row["Team One Flag"], s1, row["Projected Winner"] === row["Team One"])}
        ${teamLine(row["Team Two"], row["Team Two Flag"], s2, row["Projected Winner"] === row["Team Two"])}
        <span class="confidence">${pct(row["Win Confidence"])} confidence</span>
      </article>`;
    }

    document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", () => setPage(btn.dataset.page)));

    fetch("/api/bootstrap")
      .then(r => r.json())
      .then(data => { state.bootstrap = data; renderWelcome(); })
      .catch(err => {
        document.getElementById("welcome").innerHTML = `<section class="panel error"><h2>App failed to load</h2><p>${esc(err.message)}</p></section>`;
      });
