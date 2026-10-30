(function () {
  "use strict";

  const root = document.getElementById("app-root");
  const params = new URLSearchParams(window.location.search);
  let activeView = params.get("view") === "pilot" ? "pilot" : "rater";
  let activeMode = activeView === "pilot" ? "pilot" : "live";

  // Detail visibility threshold: 3 native tiles (832px video + 1px border each side = 834px)
  // + 2*12px gaps + 2*14px stage padding = 2554px. Same breakpoint as the wide-screen text scale in styles.css.
  const NATIVE_3UP_MIN_WIDTH = 2554;

  // Inter-trial 50% grey adaptation curtain duration (700 ms per DESIGN-cosmos3-quant-human-test-2026-10-30.md:247)
  const INTER_TRIAL_CURTAIN_MS = 700;

  // Map Shift+1-3 (both key and code) to 0-based clip indices for Worst selection in 3-Up mode
  const SHIFT_DIGIT_MAP = {
    "!": 0,
    "@": 1,
    "#": 2,
    Digit1: 0,
    Digit2: 1,
    Digit3: 2,
  };

  // Layout override from URL (`?layout=flip` or `?layout=3up` or `?layout=native`) or automatic
  let manualLayoutOverride = params.get("layout") || null; // null = auto width detection
  let activeFlipLetter = params.get("flip") || "A";
  // "auto" follows the layout: native 3-up tiles are already 1:1, every other layout fits. "fit" | "1:1" | "2x" are manual.
  let activeZoom = params.get("zoom") || "auto";
  let playbackSpeed = 1.0;
  let loopMode = "loop"; // "loop" | "bounce"

  // Session & Draft State
  let serverState = null;
  let currentGlobalIndex = null;

  // Per-item draft answers preserved across layout switches and resizes
  let draftBest = null;
  let draftWorst = null;
  let draftChoice = null;
  let draftQ2 = {};
  let draftLikert = {};
  let draftNotes = "";
  let firstPlayCompleted = false;
  let maxWatchedTime = 0;
  let videoElements = {};
  let isPlaying = true;
  let syncInterval = null;
  const BREAK_RECOMMENDED_SECONDS = 600; // 10:00 recommended rest on the break cards
  let breakSecondsRemaining = BREAK_RECOMMENDED_SECONDS;
  let breakTimerId = null;

  function formatBreakClock(totalSeconds) {
    const m = Math.floor(totalSeconds / 60);
    const s = totalSeconds % 60;
    return `${String(m).padStart(2, "0")} : ${String(s).padStart(2, "0")}`;
  }

  // Theme management
  let currentTheme = localStorage.getItem("blindpanel_theme") || params.get("theme") || "light";
  document.documentElement.setAttribute("data-theme", currentTheme);

  function toggleTheme() {
    currentTheme = currentTheme === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", currentTheme);
    localStorage.setItem("blindpanel_theme", currentTheme);
    updateThemeButtons();
  }

  function updateThemeButtons() {
    const btns = document.querySelectorAll("[data-action='toggle-theme']");
    btns.forEach((b) => {
      b.textContent = currentTheme === "light" ? "Light / Dark" : "Dark / Light";
    });
  }

  function windowFitsNative3Up() {
    return window.innerWidth >= NATIVE_3UP_MIN_WIDTH;
  }

  // Three tiles side by side are native-size (1:1) whenever the window fits them; the scaled 3-up grid is only
  // for narrower windows that asked for 3-up, and the flip is the automatic fallback there.
  function computeEffective3ClipLayout() {
    if (manualLayoutOverride === "flip") return "flip";
    if (manualLayoutOverride === "3up") return windowFitsNative3Up() ? "native" : "3up";
    if (manualLayoutOverride === "native") return "native";
    return windowFitsNative3Up() ? "native" : "flip";
  }

  function computeEffectiveZoom(effLayout) {
    if (activeZoom !== "auto") return activeZoom;
    return effLayout === "native" ? "1:1" : "fit";
  }

  function escapeHtml(str) {
    return String(str == null ? "" : str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function apiGet(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }
    return res.json();
  }

  async function apiPost(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
  }

  async function loadViewData() {
    activeMode = activeView === "pilot" ? "pilot" : "live";
    const reqIdx = params.get("item") != null ? `&global_index=${encodeURIComponent(params.get("item"))}` : "";
    const st = await apiGet(`/api/state?mode=${activeMode}${reqIdx}`);
    applyNewServerState(st);
  }

  function applyNewServerState(st) {
    const itemChanged = currentGlobalIndex !== st.global_index || (serverState && serverState.interstitial !== st.interstitial);
    serverState = st;

    if (itemChanged) {
      currentGlobalIndex = st.global_index;
      draftBest = null;
      draftWorst = null;
      draftChoice = st.choice_outcome || null;
      draftQ2 = Object.assign({}, st.q2_by_letter || {});
      draftLikert = Object.assign({}, st.likert_by_letter || {});
      draftNotes = st.notes || "";
      activeFlipLetter = (st.letters && st.letters[0]) || "A";

      // Restore Best/Worst draft if item already has ranks
      if (st.ranks_by_letter) {
        for (const [letKey, rnk] of Object.entries(st.ranks_by_letter)) {
          if (rnk === 1) draftBest = letKey;
          if (rnk === st.k) draftWorst = letKey;
        }
      }

      firstPlayCompleted = !st.require_full_play || Boolean(st.completed);
      maxWatchedTime = 0;
    }

    renderRaterOrPilotScreen();
  }

  // ------------------------------------------------------------------
  // Rater & Pilot Screens (Interstitials + Active Rating Stage)
  // ------------------------------------------------------------------
  function renderRaterOrPilotScreen() {
    const st = serverState;
    if (!st) return;

    if (st.interstitial === "session_start") {
      renderSessionStartInterstitial(st);
      return;
    }
    if (st.interstitial === "rest_break" || st.interstitial === "block_break") {
      renderRestBreakInterstitial(st);
      return;
    }
    if (st.interstitial === "session_complete") {
      renderSessionCompleteInterstitial(st);
      return;
    }
    renderActiveRatingStage(st);
  }

  function renderPilotBannerIfNeeded(st) {
    if (st.mode !== "pilot") return "";
    return `
      <div class="pilot-banner" role="region" aria-label="Pilot Mode Isolated Recording Notice">
        <div class="pilot-banner-left">
          <span class="pilot-pill">⚠ PILOT MODE</span>
          <span>VERIFICATION PASS — RECORDED APART FROM LIVE ANSWERS</span>
        </div>
        <div class="pilot-banner-right mono">
          <span>Pilot Log Isolated</span>
          <span>|</span>
          <span>Live Session Untouched</span>
        </div>
      </div>`;
  }

  function renderSessionStartInterstitial(st) {
    const totalInBlock = st.block_total_items;
    const pct = Math.max(2, Math.round((1 / totalInBlock) * 100));
    root.innerHTML = `
      ${renderPilotBannerIfNeeded(st)}
      <header class="top-bar" role="banner">
        <div class="progress-cluster" aria-label="Session Progress">
          <span class="brand-title">blindpanel</span>
          <span class="badge mono">${escapeHtml(st.counter_label)}</span>
          <div class="progress-track" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="${escapeHtml(st.progress_aria_label)}">
            <div class="progress-fill" style="width: ${pct}%;"></div>
          </div>
        </div>
        <div class="brand-group">
          <button type="button" class="btn btn-sm" data-action="toggle-theme">Light / Dark</button>
        </div>
      </header>

      <main class="content-single-card" role="main">
        <section class="panel-card" aria-labelledby="start-heading">
          <div style="display: flex; flex-direction: column; gap: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
              <span class="badge badge-pass">Session Start</span>
              <span class="mono" style="font-size: calc(12px * var(--ui-scale)); color: var(--text-secondary);">${escapeHtml(st.counter_label)}</span>
            </div>

            <h2 id="start-heading">Practice items first</h2>

            <p style="color: var(--text-secondary); font-size: calc(13px * var(--ui-scale)); line-height: 1.48;">
              The opening practice items let you adjust to the video clips and playback controls, followed by the rest of the <strong>${totalInBlock} items</strong> in this session.
            </p>

            <div class="stat-grid">
              <div class="stat-box">
                <div class="stat-value">${totalInBlock} Items</div>
                <div class="stat-label">Total in this session</div>
              </div>
              <div class="stat-box">
                <div class="stat-value">${st.k} Clips (${st.letters[0]}–${st.letters[st.letters.length - 1]})</div>
                <div class="stat-label">Synchronized playback</div>
              </div>
              <div class="stat-box">
                <div class="stat-value">1 Full Play</div>
                <div class="stat-label">Required before picking</div>
              </div>
            </div>

            <div style="background: var(--bg-elevated); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 10px 12px; display: flex; flex-direction: column; gap: 6px; font-size: calc(12.5px * var(--ui-scale)); color: var(--text-secondary);">
              <div style="color: var(--text-primary); font-weight: 700;">What to expect during rating:</div>
              <div>• Watch the synchronized clips through at least one full loop before voting unlocks.</div>
              <div>• Pick the <strong>Best</strong> and <strong>Worst</strong> clip; on 3-clip items, the remaining clip is ranked second automatically.</div>
              <div>• On robot manipulation items, also answer how far the robot gets with the action within the 2.5 s clip (<strong>Yes / Partly / No</strong>).</div>
            </div>
          </div>

          <div style="display: flex; align-items: center; justify-content: space-between; border-top: 1px solid var(--border-subtle); padding-top: 10px;">
            <div class="progress-cluster">
              <div class="progress-track" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="${escapeHtml(st.progress_aria_label)}">
                <div class="progress-fill" style="width: ${pct}%;"></div>
              </div>
              <span style="font-size: calc(12px * var(--ui-scale)); color: var(--text-secondary);">${escapeHtml(st.counter_label)} ready</span>
            </div>
            <button type="button" class="btn btn-primary" id="btn-begin-session" ${st.live_start_blocked ? "disabled" : ""}>
              ${st.live_start_blocked ? "🔒 Session Locked" : "Begin Session ↵"}
            </button>
          </div>
        </section>
      </main>
    `;
    updateThemeButtons();
    root.querySelectorAll("[data-action='toggle-theme']").forEach((b) => b.addEventListener("click", toggleTheme));
    const btnBegin = document.getElementById("btn-begin-session");
    if (btnBegin && !st.live_start_blocked) {
      btnBegin.addEventListener("click", async () => {
        const nextSt = await apiPost("/api/session/start", { mode: activeMode });
        applyNewServerState(nextSt);
      });
    }
  }

  function renderRestBreakInterstitial(st) {
    const isBlockTransition = st.interstitial === "block_break";
    // At a block transition the finished block's total comes from the server (the same total the counter
    // showed for that block); mid-block, the items saved so far out of the current block's total.
    const prevBlockNumber = st.block_number - 1;
    const savedCount = isBlockTransition ? st.prev_block_total_items : st.item_number - 1;
    const savedTotal = isBlockTransition ? st.prev_block_total_items : st.block_total_items;

    // Recommended rest: a real 10-minute countdown (the previous card showed a fixed "08 : 42" that never moved).
    if (breakTimerId) clearInterval(breakTimerId);
    breakSecondsRemaining = BREAK_RECOMMENDED_SECONDS;

    root.innerHTML = `
      ${renderPilotBannerIfNeeded(st)}
      <header class="top-bar" role="banner">
        <div class="progress-cluster" aria-label="Session Progress">
          <span class="brand-title">blindpanel</span>
          <span class="badge mono">Item ${savedCount} / ${savedTotal}</span>
        </div>
        <div class="brand-group">
          <button type="button" class="btn btn-sm" data-action="toggle-theme">Light / Dark</button>
        </div>
      </header>

      <main class="content-single-card" role="main">
        <section class="panel-card" aria-labelledby="break-heading">
          <div style="display: flex; flex-direction: column; gap: 12px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
              <span class="badge badge-warn">Rest Break</span>
              <span class="mono" style="font-size: calc(12px * var(--ui-scale)); color: var(--text-secondary);">Item ${savedCount} / ${savedTotal} Saved</span>
            </div>

            <h2 id="break-heading">
              ${isBlockTransition ? `Block ${prevBlockNumber} Complete — Take a Break (Block ${st.block_number} Optional)` : "Mid-Block Rest Break — Take a Short Break"}
            </h2>

            <p style="color: var(--text-secondary); font-size: calc(13px * var(--ui-scale)); line-height: 1.48;">
              ${
                isBlockTransition
                  ? `All <strong>${savedTotal} items</strong> of Block ${prevBlockNumber} are complete and saved on disk. Take a short break to rest your eyes. <strong>Block ${st.block_number} (${st.block_total_items} side-by-side pairs) is optional</strong> — you may save and close now, or continue to Block ${st.block_number} after resting.`
                  : `Items 1–${savedCount} are saved on disk. Take a short break to rest your eyes before continuing with Item ${st.item_number} / ${st.block_total_items}.`
              }
            </p>

            <div style="background: var(--bg-elevated); border: 1.5px solid var(--border-strong); border-radius: 6px; padding: 12px 14px; display: flex; align-items: center; justify-content: space-between;">
              <div>
                <div style="font-size: calc(12px * var(--ui-scale)); font-weight: 700; text-transform: uppercase; color: var(--text-secondary);">
                  Recommended Rest Remaining
                </div>
                <div class="mono" id="break-timer-display" style="font-size: calc(28px * var(--ui-scale)); font-weight: 800; letter-spacing: 0.03em; margin: 2px 0;">
                  ${formatBreakClock(breakSecondsRemaining)}
                </div>
                <div style="font-size: calc(12px * var(--ui-scale)); color: var(--text-secondary);">
                  Look at a distant object ≥ 6 m away to rest your eyes.
                </div>
              </div>
              <span class="badge badge-pass">Progress Saved</span>
            </div>
          </div>

          <div style="display: flex; align-items: center; justify-content: space-between; border-top: 1px solid var(--border-subtle); padding-top: 10px; gap: 8px;">
            ${
              isBlockTransition
                ? `<button type="button" class="btn" id="btn-save-close">Save &amp; Close</button>
                   <button type="button" class="btn btn-primary" id="btn-resume-block2">Resume Block ${st.block_number} ↵</button>`
                : `<span></span>
                   <button type="button" class="btn btn-primary" id="btn-resume-break">Continue Session ↵</button>`
            }
          </div>
        </section>
      </main>
    `;
    updateThemeButtons();
    root.querySelectorAll("[data-action='toggle-theme']").forEach((b) => b.addEventListener("click", toggleTheme));

    breakTimerId = setInterval(() => {
      const disp = document.getElementById("break-timer-display");
      if (!disp) {
        clearInterval(breakTimerId);
        breakTimerId = null;
        return;
      }
      breakSecondsRemaining = Math.max(0, breakSecondsRemaining - 1);
      disp.textContent = formatBreakClock(breakSecondsRemaining);
      if (breakSecondsRemaining === 0) {
        clearInterval(breakTimerId);
        breakTimerId = null;
      }
    }, 1000);

    const btnResumeBreak = document.getElementById("btn-resume-break");
    if (btnResumeBreak) {
      btnResumeBreak.addEventListener("click", async () => {
        const nextSt = await apiPost("/api/session/resume_break", { mode: activeMode });
        applyNewServerState(nextSt);
      });
    }
    const btnResumeBlock2 = document.getElementById("btn-resume-block2");
    if (btnResumeBlock2) {
      btnResumeBlock2.addEventListener("click", async () => {
        const nextSt = await apiPost("/api/session/start_block2", { mode: activeMode });
        applyNewServerState(nextSt);
      });
    }
    const btnSaveClose = document.getElementById("btn-save-close");
    if (btnSaveClose) {
      btnSaveClose.addEventListener("click", async () => {
        const nextSt = await apiPost("/api/session/decline_block2", { mode: activeMode });
        applyNewServerState(nextSt);
      });
    }
  }

  function renderSessionCompleteInterstitial(st) {
    // Card tm (2): The rater export is removed entirely. The Session Complete view only thanks the rater and says the session is saved.
    // The counts are the server's list of fully rated blocks, each with the same total the session counter showed
    // for it ("Item N / 43" during Block 1 ends as "43 / 43"); a declined optional block is simply not listed.
    const doneBlocks = Array.isArray(st.completed_blocks) && st.completed_blocks.length > 0
      ? st.completed_blocks
      : [{ block_number: st.block_number, total_items: st.block_total_items }];
    const lastBlock = doneBlocks[doneBlocks.length - 1];
    const totalDone = lastBlock.total_items;
    const perBlockSummary = doneBlocks.map((b) => `Block ${b.block_number}: ${b.total_items} / ${b.total_items}`).join(" · ");
    const ratedHeading =
      doneBlocks.length === 1
        ? `${totalDone} / ${totalDone} items rated`
        : `${perBlockSummary} items rated`;
    const allSavedLine =
      doneBlocks.length === 1
        ? `All ${totalDone} items saved and verified`
        : `All items saved and verified (${perBlockSummary})`;
    root.innerHTML = `
      ${renderPilotBannerIfNeeded(st)}
      <header class="top-bar" role="banner">
        <div class="brand-group">
          <span class="brand-title">blindpanel</span>
          <span class="badge mono" id="counter-badge-complete">Item ${totalDone} / ${totalDone}</span>
          <span class="badge">Session Complete</span>
        </div>
        <div class="brand-group">
          <button type="button" class="btn btn-sm" data-action="toggle-theme">Light / Dark</button>
        </div>
      </header>

      <main class="content-single-card" role="main">
        <section class="panel-card" aria-labelledby="rater-complete-heading">
          <div style="display: flex; flex-direction: column; gap: 14px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
              <span class="badge badge-pass">Session Complete</span>
              <span class="mono" style="font-size: calc(12px * var(--ui-scale)); color: var(--text-secondary);">Item ${totalDone} / ${totalDone}</span>
            </div>

            <h2 id="rater-complete-heading">Session Complete — ${ratedHeading}</h2>

            <div class="progress-cluster">
              <div class="progress-track" style="width: 160px;" role="progressbar" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100" aria-label="Item ${totalDone} of ${totalDone} complete">
                <div class="progress-fill" style="width: 100%;"></div>
              </div>
              <span style="font-size: calc(12.5px * var(--ui-scale)); font-weight: 700;">100% Complete (Item ${totalDone} / ${totalDone})</span>
            </div>

            <p id="session-saved-message" style="color: var(--text-secondary); font-size: calc(13.5px * var(--ui-scale)); line-height: 1.5;">
              Thank you for completing the evaluation. Your session is saved on disk; you may close this window safely.
            </p>
          </div>

          <div style="display: flex; align-items: center; justify-content: space-between; border-top: 1px solid var(--border-subtle); padding-top: 10px;">
            <span style="font-size: calc(12px * var(--ui-scale)); color: var(--text-secondary);">
              ${allSavedLine}
            </span>
          </div>
        </section>
      </main>
    `;
    updateThemeButtons();
    root.querySelectorAll("[data-action='toggle-theme']").forEach((b) => b.addEventListener("click", toggleTheme));
  }

  // ------------------------------------------------------------------
  // Active Rating Screen (Block 1 3-Clip BWS, Block 2 2AFC, BWS, Pairwise, MOS, ABX)
  // ------------------------------------------------------------------
  function renderActiveRatingStage(st) {
    const pct = Math.max(2, Math.round((st.item_number / st.block_total_items) * 100));
    const methodBadgeMap = {
      bws_3: "Pick Best & Worst",
      bws: "Pick Best & Worst",
      "2afc": "Side-by-Side Comparison",
      pairwise: "Pairwise Comparison",
      mos_likert: "Quality Rating (1–5)",
      abx: "ABX Reference Match",
    };
    const methodBadge = methodBadgeMap[st.method] || "Comparison";

    const is3Clip = st.k >= 3 && st.method !== "abx";
    const effLayout = is3Clip ? computeEffective3ClipLayout() : "2up";

    let gridClass = "video-grid-3";
    if (st.method === "2afc" || st.method === "pairwise" || st.k === 2) {
      gridClass = "video-grid-2-centered";
    } else if (is3Clip) {
      if (effLayout === "native") gridClass = "video-grid-native-3";
      else if (effLayout === "flip") gridClass = "video-grid-flip-single";
      else gridClass = "video-grid-3";
    }

    // Determine middle letter for bws_3 when both Best and Worst are picked
    let derivedMiddle = null;
    if (st.method === "bws_3" && draftBest && draftWorst && draftBest !== draftWorst) {
      const rem = st.letters.filter((l) => l !== draftBest && l !== draftWorst);
      if (rem.length === 1) derivedMiddle = rem[0];
    }

    // Build video tiles
    const clipsToRender =
      st.method === "abx"
        ? [
            { letter: "A", url: st.reference_clips.A },
            { letter: "B", url: st.reference_clips.B },
            { letter: "X", url: st.reference_clips.X },
          ]
        : st.clips;

    const tilesHtml = clipsToRender
      .map((clip) => {
        const letKey = clip.letter;
        let pickClass = "";
        let pickBadgeHtml = "";
        let ariaDesc = `Clip ${letKey}`;

        if (st.method === "bws_3" || st.method === "bws") {
          if (draftBest === letKey) {
            pickClass = "picked-best";
            pickBadgeHtml = `<div class="tile-pick-badge tile-pick-best">★ BEST</div>`;
            ariaDesc = `Clip ${letKey} — Selected as Best`;
          } else if (draftWorst === letKey) {
            pickClass = "picked-worst";
            pickBadgeHtml = `<div class="tile-pick-badge tile-pick-worst">✕ WORST</div>`;
            ariaDesc = `Clip ${letKey} — Selected as Worst`;
          } else if (derivedMiddle === letKey) {
            pickClass = "picked-middle";
            pickBadgeHtml = `<div class="tile-pick-badge tile-pick-middle">2nd (Auto)</div>`;
            ariaDesc = `Clip ${letKey} — Ranked 2nd Automatically`;
          }
        } else if (st.method === "2afc" || st.method === "pairwise" || st.method === "abx") {
          if (draftChoice === letKey) {
            pickClass = "picked-best";
            pickBadgeHtml = `<div class="tile-pick-badge tile-pick-best">★ SELECTED</div>`;
            ariaDesc = `Clip ${letKey} — Selected`;
          }
        }

        const flipHiddenClass =
          effLayout === "flip" && is3Clip && letKey !== activeFlipLetter ? "flip-inactive" : "";
        const effZoom = computeEffectiveZoom(effLayout);
        const zoomClass =
          effZoom === "1:1" ? "zoom-1to1" : effZoom === "2x" ? "zoom-2x" : "";

        return `
          <div class="video-tile ${pickClass} ${flipHiddenClass} ${zoomClass}" data-tile-letter="${letKey}" role="group" aria-label="${ariaDesc}">
            <video
              id="video-clip-${letKey}"
              data-clip-letter="${letKey}"
              src="${escapeHtml(clip.url)}"
              muted
              playsinline
              preload="auto"
            ></video>
            <div class="tile-letter-badge">${letKey}</div>
            ${pickBadgeHtml}
          </div>`;
      })
      .join("");

    // Layout & Flip Controls for 3-Clip detail visibility
    const flipSwitcherHtml = is3Clip
      ? `
        <div class="segmented-control" role="group" aria-label="View Layout Mode">
          <button type="button" class="seg-btn ${effLayout === "flip" ? "active" : ""}" data-layout-mode="flip" title="1:1 Single-Clip Flip Mode (keys 1 / 2 / 3)">1:1 Flip</button>
          <button type="button" class="seg-btn ${effLayout === "3up" || effLayout === "native" ? "active" : ""}" data-layout-mode="3up" title="3-Up Side-by-Side Grid">3-Up</button>
        </div>
        ${
          effLayout === "flip"
            ? `<div class="segmented-control" role="group" aria-label="Active Flip Clip">
                ${st.letters
                  .map(
                    (l, idx) =>
                      `<button type="button" class="seg-btn ${activeFlipLetter === l ? "active" : ""}" data-flip-letter="${l}" title="Show Clip ${l} (Key ${idx + 1})">${l} (${idx + 1})</button>`
                  )
                  .join("")}
               </div>`
            : ""
        }
      `
      : "";

    // Question 2 Row (Manipulation M1-M4 items only)
    let q2Html = "";
    if (st.ask_q2 && st.question2) {
      const q2Locked = !firstPlayCompleted;
      const groupsHtml = st.letters
        .map((letKey) => {
          const curAns = draftQ2[letKey] || "";
          const btnsHtml = st.question2.options
            .map((opt) => {
              const isSel = curAns === opt.id;
              const selClass = isSel ? `selected-${opt.id}` : "";
              const prefix = isSel ? "✓ " : "";
              return `<button type="button" class="action-btn ${selClass}" data-q2-letter="${letKey}" data-q2-val="${escapeHtml(opt.id)}" ${q2Locked ? "disabled" : ""}>${prefix}${escapeHtml(opt.label)}</button>`;
            })
            .join("");
          return `
            <div class="action-clip-group" role="group" aria-label="Clip ${letKey} Action Progress">
              <span class="action-clip-label">Clip ${letKey}:</span>
              ${btnsHtml}
            </div>`;
        })
        .join("");

      q2Html = `
        <div class="action-row" role="group" aria-label="Action Progress Question">
          <div class="action-row-header">
            <span class="action-question-title">${escapeHtml(st.question2.prompt)}</span>
            <span class="action-question-hint">${
              q2Locked ? "Unlocks after 1st full loop completes" : `Select one per clip (${st.letters.join(", ")})`
            }</span>
          </div>
          <div class="action-grid-3">${groupsHtml}</div>
        </div>`;
    }

    // Decision Row (Method-driven)
    const voteHtml = buildVoteClusterHtml(st, derivedMiddle);
    const effZoomForButtons = computeEffectiveZoom(effLayout);

    root.innerHTML = `
      ${renderPilotBannerIfNeeded(st)}
      <header class="top-bar" role="banner">
        <div class="progress-cluster" aria-label="Session Progress">
          <span class="brand-title">blindpanel</span>
          <span class="badge mono" id="counter-badge">${escapeHtml(st.counter_label)}</span>
          <div class="progress-track" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="${escapeHtml(st.progress_aria_label)}">
            <div class="progress-fill" style="width: ${pct}%;"></div>
          </div>
          <span class="badge">${escapeHtml(methodBadge)}</span>
        </div>

        <div class="brand-group">
          ${st.can_undo ? `<button type="button" class="btn btn-sm" id="btn-undo" title="Undo previous item rating">↶ Undo</button>` : ""}
          <button type="button" class="btn btn-sm" id="btn-shortcuts" title="Keyboard shortcuts: 1-3 / A-C pick Best (or flip clip in 1:1 Flip mode), Shift+1-3 / Shift+A-C pick Worst, Y/P/N action progress, Space play/pause, Left/Right step frame, Enter submit, F flag flawed" aria-label="Keyboard Shortcuts">
            ? Shortcuts
          </button>
          <button type="button" class="btn btn-sm" data-action="toggle-theme">Light / Dark</button>
        </div>
      </header>

      <div class="prompt-strip" role="region" aria-label="Current Prompt and Visual Cue">
        <div class="prompt-main">
          <span class="prompt-id mono">${escapeHtml(st.prompt_id)}</span>
          <span class="prompt-text">${escapeHtml(st.prompt)}</span>
        </div>
        <div class="prompt-moment" id="cue-strip">
          <strong>Look at:</strong> ${escapeHtml(st.rater_cue)}
        </div>
      </div>

      <main class="video-stage" role="main" aria-label="Video Comparison Stage">
        <div id="grey-gap-curtain" class="grey-gap-curtain hidden" aria-hidden="true"></div>
        <div id="video-grid-container" class="${gridClass}" data-effective-layout="${effLayout}">
          ${tilesHtml}
        </div>
      </main>

      <section class="bottom-hud" aria-label="Playback and Voting Controls">
        <div class="transport-row">
          <div class="transport-group">
            <button type="button" class="btn btn-sm" id="btn-play-pause">⏸ Pause</button>
            <button type="button" class="btn btn-sm" id="btn-restart" title="Restart from Frame 0">↺ Restart</button>
            <div class="segmented-control" role="group" aria-label="Frame Stepping">
              <button type="button" class="seg-btn" id="btn-step-back" title="Step back 1 frame">−1f</button>
              <span class="seg-btn mono" id="frame-readout" style="cursor: default; color: var(--text-primary);">Frame 1 / 61</span>
              <button type="button" class="seg-btn" id="btn-step-fwd" title="Step forward 1 frame">+1f</button>
            </div>
          </div>

          <div class="scrubber-box" aria-label="Synchronized Scrubber">
            <div class="scrubber-bar" id="scrubber-bar" role="slider" aria-label="Playback Timeline" aria-valuenow="1" aria-valuemin="0" aria-valuemax="61" tabindex="0">
              <div class="scrubber-fill" id="scrubber-fill" style="width: 0%;"></div>
            </div>
          </div>

          <div class="transport-group">
            ${flipSwitcherHtml}
            <div class="segmented-control" role="group" aria-label="Loop Mode">
              <button type="button" class="seg-btn ${loopMode === "loop" ? "active" : ""}" data-loop-mode="loop">Loop</button>
              <button type="button" class="seg-btn ${loopMode === "bounce" ? "active" : ""}" data-loop-mode="bounce">Bounce</button>
            </div>
            <div class="segmented-control" role="group" aria-label="Playback Speed">
              <button type="button" class="seg-btn ${playbackSpeed === 0.25 ? "active" : ""}" data-speed="0.25">0.25×</button>
              <button type="button" class="seg-btn ${playbackSpeed === 0.5 ? "active" : ""}" data-speed="0.5">0.5×</button>
              <button type="button" class="seg-btn ${playbackSpeed === 1.0 ? "active" : ""}" data-speed="1.0">1.0×</button>
            </div>
            <div class="segmented-control" role="group" aria-label="Viewport Zoom">
              <button type="button" class="seg-btn ${effZoomForButtons === "fit" ? "active" : ""}" data-zoom="fit">Fit</button>
              <button type="button" class="seg-btn ${effZoomForButtons === "1:1" ? "active" : ""}" data-zoom="1:1">1:1</button>
              <button type="button" class="seg-btn ${effZoomForButtons === "2x" ? "active" : ""}" data-zoom="2x">2×</button>
            </div>
          </div>
        </div>

        ${q2Html}

        <div class="decision-row">
          <div class="vote-cluster" id="vote-cluster">
            ${voteHtml}
          </div>

          <div class="notes-cluster">
            <input
              type="text"
              id="notes-input"
              class="notes-input"
              placeholder="Notes (optional)…"
              value="${escapeHtml(draftNotes)}"
              aria-label="Item notes"
            >
            <button type="button" class="btn btn-sm btn-flawed ${st.flawed ? "active-flawed" : ""}" id="btn-flag-flawed" title="Flag this item if flawed or action cannot be shown (Key F)">
              ⚑ ${st.flawed ? "Flagged Flawed" : "Flag flawed item"}
            </button>
          </div>
        </div>
      </section>
    `;

    updateThemeButtons();
    bindActiveStageEvents(st);
    setupSynchronizedVideos(st);
  }

  function buildVoteClusterHtml(st, derivedMiddle) {
    const locked = !firstPlayCompleted;
    const lockBadge = locked
      ? `<span class="badge badge-warn mono" id="first-play-lock-badge" style="font-weight: 800; padding: 4px 8px;">🔒 LOCKED UNTIL 1 FULL PLAY</span>`
      : "";

    const q2Complete =
      !st.ask_q2 ||
      !st.question2 ||
      st.letters.every((l) => Boolean(draftQ2[l]));

    if (st.method === "bws_3" || st.method === "bws") {
      const targetLetters = st.method === "bws" ? st.remaining_letters : st.letters;
      const bestBtns = targetLetters
        .map((l) => {
          const isSel = draftBest === l;
          return `<button type="button" class="pick-btn ${isSel ? "selected-best" : ""}" data-pick-role="best" data-pick-letter="${l}" ${locked ? "disabled" : ""}>${isSel ? `★ ${l}` : l}</button>`;
        })
        .join("");

      const worstBtns = targetLetters
        .map((l) => {
          const isSel = draftWorst === l;
          return `<button type="button" class="pick-btn ${isSel ? "selected-worst" : ""}" data-pick-role="worst" data-pick-letter="${l}" ${locked ? "disabled" : ""}>${isSel ? `✕ ${l}` : l}</button>`;
        })
        .join("");

      const orderBadge =
        st.method === "bws_3" && draftBest && draftWorst && derivedMiddle
          ? `<span class="badge mono" id="order-summary-badge" aria-label="Complete Ranking Order">Order: ${draftBest} ≻ ${derivedMiddle} ≻ ${draftWorst}</span>`
          : `<span class="badge mono" id="order-summary-badge" aria-label="Complete Ranking Order">Pick Best &amp; Worst</span>`;

      const canSubmit = !locked && draftBest && draftWorst && draftBest !== draftWorst && q2Complete;

      return `
        ${lockBadge}
        <div class="pick-group" role="group" aria-label="Pick Best of 3">
          <span class="pick-label">Best:</span>
          ${bestBtns}
        </div>
        <div class="pick-group" role="group" aria-label="Pick Worst of 3">
          <span class="pick-label">Worst:</span>
          ${worstBtns}
        </div>
        ${orderBadge}
        <button type="button" class="btn btn-primary" id="btn-submit-item" ${canSubmit ? "" : "disabled"}>
          Submit &amp; Next ↵
        </button>`;
    }

    if (st.method === "2afc" || st.method === "pairwise") {
      const leftL = st.letters[0] || "A";
      const rightL = st.letters[1] || "B";
      const unsureBtn =
        st.method === "pairwise" && st.allow_unsure
          ? `<button type="button" class="pick-btn ${draftChoice === "unsure" ? "selected-best" : ""}" data-choice="unsure" ${locked ? "disabled" : ""}>${draftChoice === "unsure" ? "★ Unsure" : "Unsure"}</button>`
          : "";
      const canSubmit = !locked && Boolean(draftChoice) && q2Complete;
      return `
        ${lockBadge}
        <div class="pick-group" role="group" aria-label="Side-by-Side Selection">
          <span class="pick-label">Better overall quality:</span>
          <button type="button" class="pick-btn ${draftChoice === leftL ? "selected-best" : ""}" data-choice="${leftL}" ${locked ? "disabled" : ""}>${draftChoice === leftL ? "★ " : ""}Prefer Left (${leftL})</button>
          <button type="button" class="pick-btn ${draftChoice === rightL ? "selected-best" : ""}" data-choice="${rightL}" ${locked ? "disabled" : ""}>${draftChoice === rightL ? "★ " : ""}Prefer Right (${rightL})</button>
          ${unsureBtn}
        </div>
        <button type="button" class="btn btn-primary" id="btn-submit-item" ${canSubmit ? "" : "disabled"}>
          Submit Pair ↵
        </button>`;
    }

    if (st.method === "abx") {
      const canSubmit = !locked && (draftChoice === "A" || draftChoice === "B") && q2Complete;
      return `
        ${lockBadge}
        <div class="pick-group" role="group" aria-label="ABX Selection">
          <span class="pick-label">Clip X matches:</span>
          <button type="button" class="pick-btn ${draftChoice === "A" ? "selected-best" : ""}" data-choice="A" ${locked ? "disabled" : ""}>${draftChoice === "A" ? "★ " : ""}Reference A</button>
          <button type="button" class="pick-btn ${draftChoice === "B" ? "selected-best" : ""}" data-choice="B" ${locked ? "disabled" : ""}>${draftChoice === "B" ? "★ " : ""}Reference B</button>
        </div>
        <button type="button" class="btn btn-primary" id="btn-submit-item" ${canSubmit ? "" : "disabled"}>
          Submit ABX ↵
        </button>`;
    }

    if (st.method === "mos_likert") {
      const scale = st.likert_scale || [
        { score: 5, label: "5" },
        { score: 4, label: "4" },
        { score: 3, label: "3" },
        { score: 2, label: "2" },
        { score: 1, label: "1" },
      ];
      const groups = st.letters
        .map((letKey) => {
          const curScore = draftLikert[letKey];
          const btns = scale
            .map((pt) => {
              const isSel = Number(curScore) === Number(pt.score);
              return `<button type="button" class="pick-btn ${isSel ? "selected-best" : ""}" data-likert-letter="${letKey}" data-likert-score="${pt.score}" ${locked ? "disabled" : ""}>${pt.score}</button>`;
            })
            .join("");
          return `<div class="pick-group" role="group" aria-label="MOS Score Clip ${letKey}">
            <span class="pick-label">Clip ${letKey}:</span>
            ${btns}
          </div>`;
        })
        .join("");
      const allScored = st.letters.every((l) => draftLikert[l] != null);
      const canSubmit = !locked && allScored && q2Complete;
      return `
        ${lockBadge}
        ${groups}
        <button type="button" class="btn btn-primary" id="btn-submit-item" ${canSubmit ? "" : "disabled"}>
          Submit Rating ↵
        </button>`;
    }

    return "";
  }

  // Update selection UI in-place without recreating <video> elements (preserves exact playback frame!)
  function refreshSelectionAndLayoutInPlace() {
    const st = serverState;
    if (!st) return;

    const is3Clip = st.k >= 3 && st.method !== "abx";
    const effLayout = is3Clip ? computeEffective3ClipLayout() : "2up";
    const gridEl = document.getElementById("video-grid-container");
    if (gridEl && is3Clip) {
      gridEl.className =
        effLayout === "native"
          ? "video-grid-native-3"
          : effLayout === "flip"
          ? "video-grid-flip-single"
          : "video-grid-3";
      gridEl.setAttribute("data-effective-layout", effLayout);
    }
    const effZoom = computeEffectiveZoom(effLayout);

    let derivedMiddle = null;
    if (st.method === "bws_3" && draftBest && draftWorst && draftBest !== draftWorst) {
      const rem = st.letters.filter((l) => l !== draftBest && l !== draftWorst);
      if (rem.length === 1) derivedMiddle = rem[0];
    }

    // Update each video tile's classes and badge without touching the <video> element
    const tiles = document.querySelectorAll(".video-tile[data-tile-letter]");
    tiles.forEach((tile) => {
      const letKey = tile.getAttribute("data-tile-letter");
      tile.classList.remove(
        "picked-best",
        "picked-worst",
        "picked-middle",
        "flip-inactive",
        "zoom-1to1",
        "zoom-2x"
      );
      if (effLayout === "flip" && is3Clip && letKey !== activeFlipLetter) {
        tile.classList.add("flip-inactive");
      }
      if (effZoom === "1:1") tile.classList.add("zoom-1to1");
      if (effZoom === "2x") tile.classList.add("zoom-2x");

      const oldBadge = tile.querySelector(".tile-pick-badge");
      if (oldBadge) oldBadge.remove();

      if (st.method === "bws_3" || st.method === "bws") {
        if (draftBest === letKey) {
          tile.classList.add("picked-best");
          tile.setAttribute("aria-label", `Clip ${letKey} — Selected as Best`);
          tile.insertAdjacentHTML("beforeend", `<div class="tile-pick-badge tile-pick-best">★ BEST</div>`);
        } else if (draftWorst === letKey) {
          tile.classList.add("picked-worst");
          tile.setAttribute("aria-label", `Clip ${letKey} — Selected as Worst`);
          tile.insertAdjacentHTML("beforeend", `<div class="tile-pick-badge tile-pick-worst">✕ WORST</div>`);
        } else if (derivedMiddle === letKey) {
          tile.classList.add("picked-middle");
          tile.setAttribute("aria-label", `Clip ${letKey} — Ranked 2nd Automatically`);
          tile.insertAdjacentHTML("beforeend", `<div class="tile-pick-badge tile-pick-middle">2nd (Auto)</div>`);
        } else {
          tile.setAttribute("aria-label", `Clip ${letKey}`);
        }
      } else if (draftChoice === letKey) {
        tile.classList.add("picked-best");
        tile.setAttribute("aria-label", `Clip ${letKey} — Selected`);
        tile.insertAdjacentHTML("beforeend", `<div class="tile-pick-badge tile-pick-best">★ SELECTED</div>`);
      }
    });

    // Update Q2 buttons in place
    if (st.ask_q2 && st.question2) {
      const q2Locked = !firstPlayCompleted;
      const hintEl = document.querySelector(".action-question-hint");
      if (hintEl) {
        hintEl.textContent = q2Locked
          ? "Unlocks after 1st full loop completes"
          : `Select one per clip (${st.letters.join(", ")})`;
      }
      document.querySelectorAll("[data-q2-letter]").forEach((btn) => {
        const l = btn.getAttribute("data-q2-letter");
        const val = btn.getAttribute("data-q2-val");
        const opt = st.question2.options.find((o) => o.id === val);
        const isSel = draftQ2[l] === val;
        btn.disabled = q2Locked;
        btn.className = `action-btn ${isSel ? `selected-${val}` : ""}`;
        btn.textContent = `${isSel ? "✓ " : ""}${opt ? opt.label : val}`;
      });
    }

    // Re-render vote cluster and rebind vote buttons
    const voteCluster = document.getElementById("vote-cluster");
    if (voteCluster) {
      voteCluster.innerHTML = buildVoteClusterHtml(st, derivedMiddle);
      bindVoteClusterEvents(st);
    }

    // Update layout & flip toolbar buttons
    document.querySelectorAll("[data-layout-mode]").forEach((btn) => {
      const m = btn.getAttribute("data-layout-mode");
      btn.classList.toggle("active", (m === "flip" && effLayout === "flip") || (m === "3up" && effLayout !== "flip"));
    });
    document.querySelectorAll("[data-flip-letter]").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-flip-letter") === activeFlipLetter);
    });
    document.querySelectorAll("[data-zoom]").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-zoom") === effZoom);
    });
  }

  function bindVoteClusterEvents(st) {
    document.querySelectorAll("[data-pick-role]").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (!firstPlayCompleted) return;
        const role = btn.getAttribute("data-pick-role");
        const letter = btn.getAttribute("data-pick-letter");
        if (role === "best") {
          draftBest = letter;
          if (draftWorst === letter) draftWorst = null;
        } else {
          draftWorst = letter;
          if (draftBest === letter) draftBest = null;
        }
        refreshSelectionAndLayoutInPlace();
      });
    });

    document.querySelectorAll("[data-choice]").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (!firstPlayCompleted) return;
        draftChoice = btn.getAttribute("data-choice");
        refreshSelectionAndLayoutInPlace();
      });
    });

    document.querySelectorAll("[data-likert-letter]").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (!firstPlayCompleted) return;
        const l = btn.getAttribute("data-likert-letter");
        const sc = parseFloat(btn.getAttribute("data-likert-score"));
        draftLikert[l] = sc;
        refreshSelectionAndLayoutInPlace();
      });
    });

    const btnSubmit = document.getElementById("btn-submit-item");
    if (btnSubmit) {
      btnSubmit.addEventListener("click", () => submitCurrentItem());
    }
  }

  function bindActiveStageEvents(st) {
    root.querySelectorAll("[data-action='toggle-theme']").forEach((b) => b.addEventListener("click", toggleTheme));
    bindVoteClusterEvents(st);

    document.querySelectorAll("[data-q2-letter]").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (!firstPlayCompleted) return;
        const l = btn.getAttribute("data-q2-letter");
        const val = btn.getAttribute("data-q2-val");
        draftQ2[l] = val;
        refreshSelectionAndLayoutInPlace();
      });
    });

    document.querySelectorAll("[data-layout-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        manualLayoutOverride = btn.getAttribute("data-layout-mode");
        // Re-render transport row if switching to/from flip so flip A/B/C buttons appear, while preserving currentTime
        preserveVideoTimeAcrossRerender(() => renderActiveRatingStage(serverState));
      });
    });

    document.querySelectorAll("[data-flip-letter]").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeFlipLetter = btn.getAttribute("data-flip-letter");
        refreshSelectionAndLayoutInPlace();
      });
    });

    document.querySelectorAll("[data-zoom]").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeZoom = btn.getAttribute("data-zoom");
        refreshSelectionAndLayoutInPlace();
      });
    });

    document.querySelectorAll("[data-speed]").forEach((btn) => {
      btn.addEventListener("click", () => {
        playbackSpeed = parseFloat(btn.getAttribute("data-speed"));
        Object.values(videoElements).forEach((v) => {
          v.playbackRate = playbackSpeed;
        });
        document.querySelectorAll("[data-speed]").forEach((b) => {
          b.classList.toggle("active", parseFloat(b.getAttribute("data-speed")) === playbackSpeed);
        });
      });
    });

    document.querySelectorAll("[data-loop-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        loopMode = btn.getAttribute("data-loop-mode");
        document.querySelectorAll("[data-loop-mode]").forEach((b) => {
          b.classList.toggle("active", b.getAttribute("data-loop-mode") === loopMode);
        });
      });
    });

    const notesInp = document.getElementById("notes-input");
    if (notesInp) {
      notesInp.addEventListener("input", () => {
        draftNotes = notesInp.value;
      });
    }

    const btnFlag = document.getElementById("btn-flag-flawed");
    if (btnFlag) {
      btnFlag.addEventListener("click", async () => {
        const nextSt = await apiPost("/api/flag_flawed", {
          mode: activeMode,
          global_index: st.global_index,
          flawed: !st.flawed,
          reason: draftNotes || "Flagged by rater",
        });
        serverState.flawed = nextSt.flawed;
        btnFlag.classList.toggle("active-flawed", nextSt.flawed);
        btnFlag.textContent = `⚑ ${nextSt.flawed ? "Flagged Flawed" : "Flag flawed item"}`;
      });
    }

    const btnUndo = document.getElementById("btn-undo");
    if (btnUndo) {
      btnUndo.addEventListener("click", async () => {
        const nextSt = await apiPost("/api/undo", { mode: activeMode });
        applyNewServerState(nextSt);
      });
    }
  }

  function preserveVideoTimeAcrossRerender(fn) {
    const lead = Object.values(videoElements)[0];
    const savedTime = lead ? lead.currentTime : 0;
    const savedPaused = lead ? lead.paused : false;
    fn();
    Object.values(videoElements).forEach((v) => {
      const restore = () => {
        try {
          v.currentTime = savedTime;
          if (savedPaused) v.pause();
          else v.play().catch(() => {});
        } catch (_) {}
      };
      if (v.readyState >= 1) restore();
      else v.addEventListener("loadedmetadata", restore, { once: true });
    });
  }

  function setupSynchronizedVideos(st) {
    if (syncInterval) clearInterval(syncInterval);
    videoElements = {};
    const vids = Array.from(document.querySelectorAll("video[data-clip-letter]"));
    if (vids.length === 0) return;

    vids.forEach((v) => {
      const letKey = v.getAttribute("data-clip-letter");
      videoElements[letKey] = v;
      v.playbackRate = playbackSpeed;
    });

    const lead = vids[0];
    const playPauseBtn = document.getElementById("btn-play-pause");
    const restartBtn = document.getElementById("btn-restart");
    const stepBackBtn = document.getElementById("btn-step-back");
    const stepFwdBtn = document.getElementById("btn-step-fwd");
    const scrubberBar = document.getElementById("scrubber-bar");
    const scrubberFill = document.getElementById("scrubber-fill");
    const frameReadout = document.getElementById("frame-readout");

    function syncFollowersToLead() {
      const t = lead.currentTime || 0;
      for (let i = 1; i < vids.length; i++) {
        if (Math.abs((vids[i].currentTime || 0) - t) > 0.04) {
          try {
            vids[i].currentTime = t;
          } catch (_) {}
        }
      }
    }

    function updateScrubberAndUnlock() {
      const dur = lead.duration && isFinite(lead.duration) ? lead.duration : 2.54;
      const cur = lead.currentTime || 0;
      if (cur > maxWatchedTime) maxWatchedTime = cur;

      const totalFrames = 61;
      const frameNum = Math.min(totalFrames, Math.max(1, Math.round((cur / dur) * totalFrames) + 1));
      if (frameReadout) frameReadout.textContent = `Frame ${frameNum} / ${totalFrames}`;
      if (scrubberFill) scrubberFill.style.width = `${Math.min(100, (cur / dur) * 100).toFixed(1)}%`;
      if (scrubberBar) scrubberBar.setAttribute("aria-valuenow", String(frameNum));

      if (!firstPlayCompleted && (cur >= dur - 0.12 || maxWatchedTime >= dur - 0.12)) {
        firstPlayCompleted = true;
        refreshSelectionAndLayoutInPlace();
      }

      if (cur >= dur - 0.04 && !lead.paused) {
        vids.forEach((v) => {
          try {
            v.currentTime = 0;
            v.play().catch(() => {});
          } catch (_) {}
        });
      }
    }

    lead.addEventListener("ended", () => {
      if (!firstPlayCompleted) {
        firstPlayCompleted = true;
        refreshSelectionAndLayoutInPlace();
      }
      vids.forEach((v) => {
        try {
          v.currentTime = 0;
          v.play().catch(() => {});
        } catch (_) {}
      });
    });

    lead.addEventListener("timeupdate", () => {
      syncFollowersToLead();
      updateScrubberAndUnlock();
    });

    syncInterval = setInterval(() => {
      if (lead && !lead.paused) {
        syncFollowersToLead();
        updateScrubberAndUnlock();
      }
    }, 80);

    // Start all clips in lockstep once ready
    Promise.all(
      vids.map(
        (v) =>
          new Promise((resolve) => {
            if (v.readyState >= 2) resolve();
            else {
              v.addEventListener("canplay", () => resolve(), { once: true });
              setTimeout(resolve, 600);
            }
          })
      )
    ).then(() => {
      const seekParam = params.get("t");
      if (seekParam != null) {
        const targetT = parseFloat(seekParam);
        vids.forEach((v) => {
          try {
            v.currentTime = targetT;
          } catch (_) {}
        });
        isPlaying = false;
        if (playPauseBtn) playPauseBtn.textContent = "▶ Play";
        updateScrubberAndUnlock();
      } else {
        isPlaying = true;
        vids.forEach((v) => v.play().catch(() => {}));
      }
    });

    if (playPauseBtn) {
      playPauseBtn.addEventListener("click", () => {
        if (lead.paused) {
          isPlaying = true;
          vids.forEach((v) => v.play().catch(() => {}));
          playPauseBtn.textContent = "⏸ Pause";
        } else {
          isPlaying = false;
          vids.forEach((v) => v.pause());
          syncFollowersToLead();
          playPauseBtn.textContent = "▶ Play";
        }
      });
    }

    if (restartBtn) {
      restartBtn.addEventListener("click", () => {
        vids.forEach((v) => {
          try {
            v.currentTime = 0;
          } catch (_) {}
        });
        updateScrubberAndUnlock();
      });
    }

    function stepFrames(deltaFrames) {
      vids.forEach((v) => v.pause());
      if (playPauseBtn) playPauseBtn.textContent = "▶ Play";
      const dur = lead.duration && isFinite(lead.duration) ? lead.duration : 2.54;
      const nextT = Math.max(0, Math.min(dur, (lead.currentTime || 0) + deltaFrames * (1 / 24)));
      vids.forEach((v) => {
        try {
          v.currentTime = nextT;
        } catch (_) {}
      });
      if (nextT >= dur - 0.12 && !firstPlayCompleted) {
        firstPlayCompleted = true;
        refreshSelectionAndLayoutInPlace();
      }
      updateScrubberAndUnlock();
    }

    if (stepBackBtn) stepBackBtn.addEventListener("click", () => stepFrames(-1));
    if (stepFwdBtn) stepFwdBtn.addEventListener("click", () => stepFrames(1));

    if (scrubberBar) {
      scrubberBar.addEventListener("click", (ev) => {
        const rect = scrubberBar.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
        const dur = lead.duration && isFinite(lead.duration) ? lead.duration : 2.54;
        const targetT = ratio * dur;
        vids.forEach((v) => {
          try {
            v.currentTime = targetT;
          } catch (_) {}
        });
        if (ratio >= 0.95 && !firstPlayCompleted) {
          firstPlayCompleted = true;
          refreshSelectionAndLayoutInPlace();
        }
        updateScrubberAndUnlock();
      });
    }
  }

  async function submitCurrentItem() {
    const st = serverState;
    if (!st || !firstPlayCompleted) return;
    const payload = {
      mode: activeMode,
      global_index: st.global_index,
      best_letter: draftBest,
      worst_letter: draftWorst,
      choice: draftChoice,
      q2_by_letter: draftQ2,
      likert_by_letter: draftLikert,
      notes: draftNotes,
      flawed: st.flawed,
    };

    try {
      const nextSt = await apiPost("/api/submit", payload);
      // Show 700 ms #808080 grey adaptation curtain between trials (DESIGN-cosmos3-quant-human-test-2026-10-30.md:247)
      const curtain = document.getElementById("grey-gap-curtain");
      if (curtain && nextSt.global_index !== st.global_index && !nextSt.interstitial) {
        curtain.classList.remove("hidden");
        await new Promise((r) => setTimeout(r, INTER_TRIAL_CURTAIN_MS));
      }
      applyNewServerState(nextSt);
    } catch (err) {
      alert(err.message);
    }
  }

  // Responsive layout switch on window resize without losing video time or draft selections
  window.addEventListener("resize", () => {
    if (!serverState || serverState.interstitial) return;
    const is3Clip = serverState.k >= 3 && serverState.method !== "abx";
    if (!is3Clip) return;
    const gridEl = document.getElementById("video-grid-container");
    const prevLayout = gridEl ? gridEl.getAttribute("data-effective-layout") : null;
    const nextLayout = computeEffective3ClipLayout();
    if (prevLayout !== nextLayout) {
      preserveVideoTimeAcrossRerender(() => renderActiveRatingStage(serverState));
    }
  });

  // Global keyboard shortcuts (Card tm item 5: Y/P/N for Q2, Shift+1-3 for Worst in 3-up, F for flag flawed)
  window.addEventListener("keydown", (ev) => {
    if (!serverState || serverState.interstitial) {
      if (ev.key === "Enter" && serverState) {
        const beginBtn = document.getElementById("btn-begin-session");
        if (beginBtn && !beginBtn.disabled) beginBtn.click();
      }
      return;
    }
    if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA")) return;

    const st = serverState;
    const is3Clip = st.k >= 3 && st.method !== "abx";
    const effLayout = is3Clip ? computeEffective3ClipLayout() : "2up";

    // Shift+1 / Shift+2 / Shift+3 pick Worst in 3-Up mode (Card tm item 5)
    if (ev.shiftKey && (ev.key in SHIFT_DIGIT_MAP || ev.code in SHIFT_DIGIT_MAP)) {
      const idx = ev.key in SHIFT_DIGIT_MAP ? SHIFT_DIGIT_MAP[ev.key] : SHIFT_DIGIT_MAP[ev.code];
      const letKey = st.letters[idx];
      if (!letKey || !firstPlayCompleted) return;
      if (st.method === "bws_3" || st.method === "bws") {
        ev.preventDefault();
        draftWorst = letKey;
        if (draftBest === letKey) draftBest = null;
        refreshSelectionAndLayoutInPlace();
      }
      return;
    }

    // Keys 1 / 2 / 3 flip active clip in 1:1 Flip mode (or pick Best/Choice in 2-up / 3-up)
    if (["1", "2", "3"].includes(ev.key) && !ev.shiftKey) {
      const idx = parseInt(ev.key, 10) - 1;
      const letKey = st.letters[idx];
      if (!letKey) return;
      if (is3Clip && effLayout === "flip") {
        ev.preventDefault();
        activeFlipLetter = letKey;
        refreshSelectionAndLayoutInPlace();
        return;
      }
      if (firstPlayCompleted) {
        ev.preventDefault();
        if (st.method === "bws_3" || st.method === "bws") {
          draftBest = letKey;
          if (draftWorst === letKey) draftWorst = null;
        } else if (st.method === "2afc" || st.method === "pairwise" || st.method === "abx") {
          draftChoice = letKey;
        }
        refreshSelectionAndLayoutInPlace();
      }
      return;
    }

    const upper = ev.key.toUpperCase();

    // Keys A / B / C pick Best (or Shift+A/B/C pick Worst)
    if (st.letters.includes(upper) && firstPlayCompleted) {
      ev.preventDefault();
      if (st.method === "bws_3" || st.method === "bws") {
        if (ev.shiftKey) {
          draftWorst = upper;
          if (draftBest === upper) draftBest = null;
        } else {
          draftBest = upper;
          if (draftWorst === upper) draftWorst = null;
        }
      } else if (st.method === "2afc" || st.method === "pairwise" || st.method === "abx") {
        draftChoice = upper;
      }
      refreshSelectionAndLayoutInPlace();
      return;
    }

    // Keys Y / P / N answer Question 2 (Action Completion: Yes / Partly / No) for the active or next unanswered clip
    const q2ShortcutMap = { "Y": "yes", "P": "partly", "N": "no" };
    if (upper in q2ShortcutMap && st.ask_q2 && st.question2 && firstPlayCompleted) {
      ev.preventDefault();
      const q2Val = q2ShortcutMap[upper];
      let targetLetter = null;
      if (effLayout === "flip" && st.letters.includes(activeFlipLetter)) {
        targetLetter = activeFlipLetter;
      } else {
        targetLetter = st.letters.find((l) => !draftQ2[l]) || st.letters[0];
      }
      if (targetLetter) {
        draftQ2[targetLetter] = q2Val;
        refreshSelectionAndLayoutInPlace();
      }
      return;
    }

    // Key F toggles Flag Flawed item
    if (upper === "F" && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
      ev.preventDefault();
      const flagBtn = document.getElementById("btn-flag-flawed");
      if (flagBtn) flagBtn.click();
      return;
    }

    if (ev.key === " ") {
      ev.preventDefault();
      const pp = document.getElementById("btn-play-pause");
      if (pp) pp.click();
    } else if (ev.key === "ArrowLeft") {
      ev.preventDefault();
      const sb = document.getElementById("btn-step-back");
      if (sb) sb.click();
    } else if (ev.key === "ArrowRight") {
      ev.preventDefault();
      const sf = document.getElementById("btn-step-fwd");
      if (sf) sf.click();
    } else if (ev.key === "Enter") {
      const sub = document.getElementById("btn-submit-item");
      if (sub && !sub.disabled) {
        ev.preventDefault();
        sub.click();
      }
    }
  });

  // __BLINDPANEL_TEST_HOOKS_SLOT__

  loadViewData().catch((err) => {
    root.innerHTML = `<div style="padding: 20px; color: red;">Failed to load session: ${escapeHtml(err.message)}</div>`;
  });
})();
