#!/usr/bin/env python3
"""
smart_audit.py — Machine-readable audit tool for Zelda RL training runs.

Designed to be run by a coding agent or human. Outputs a structured plain-text
report with severity-tagged findings, quantified behavioral patterns, and
concrete actionable recommendations. No matplotlib required.

Usage:
    python smart_audit.py <run_folder> [--tail N] [--out report.txt]

Example:
    python smart_audit.py run_sword_v2
    python smart_audit.py run_sword_v2 --tail 500   # analyse only last 500 episodes
"""
import os
import csv
import json
import math
import argparse
import statistics
from pathlib import Path
from collections import Counter, defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_episode_csv(path):
    """Load episode_stats.csv. Returns list of dicts (one per episode)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Cast numeric columns
            for key in row:
                v = row[key]
                if v in ("True", "False"):
                    row[key] = v == "True"
                else:
                    try:
                        row[key] = int(v)
                    except ValueError:
                        try:
                            row[key] = float(v)
                        except ValueError:
                            pass
            rows.append(row)
    return rows


def load_debug_jsonl(path):
    """Load debug_stats.jsonl. Returns list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_monitor_csv(run_dir):
    """Load VecMonitor monitor.csv if present. Returns (rewards, lengths, timesteps)."""
    p = os.path.join(run_dir, "monitor.csv")
    if not os.path.exists(p):
        return [], [], []
    rewards, lengths, timesteps = [], [], []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("r,"):
                continue
            parts = line.split(",")
            try:
                rewards.append(float(parts[0]))
                lengths.append(float(parts[1]))
                timesteps.append(float(parts[2]))
            except (ValueError, IndexError):
                continue
    return rewards, lengths, timesteps


# ──────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ──────────────────────────────────────────────────────────────────────────────

def safe_mean(lst):
    return statistics.mean(lst) if lst else 0.0


def safe_median(lst):
    return statistics.median(lst) if lst else 0.0


def safe_stdev(lst):
    return statistics.stdev(lst) if len(lst) >= 2 else 0.0


def linear_trend(xs, ys):
    """Return (slope, r_squared) of simple linear regression."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx, my = safe_mean(xs), safe_mean(ys)
    ss_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    ss_xx = sum((x - mx) ** 2 for x in xs)
    ss_yy = sum((y - my) ** 2 for y in ys)
    if ss_xx == 0:
        return 0.0, 0.0
    slope = ss_xy / ss_xx
    r_sq = (ss_xy ** 2 / (ss_xx * ss_yy)) if ss_yy > 0 else 0.0
    return slope, r_sq


def pct_true(lst):
    if not lst:
        return 0.0
    return 100.0 * sum(1 for v in lst if v) / len(lst)


def rolling_mean(lst, window):
    """Return rolling mean series."""
    result = []
    buf = []
    for v in lst:
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        result.append(safe_mean(buf))
    return result


def trend_arrow(slope):
    if slope > 1e-6:
        return "↑ growing"
    elif slope < -1e-6:
        return "↓ declining"
    return "→ flat"


# ──────────────────────────────────────────────────────────────────────────────
# Severity tagging
# ──────────────────────────────────────────────────────────────────────────────

ALERTS = []


def alert(severity, message):
    """severity: CRITICAL | WARNING | OK | INFO"""
    ALERTS.append((severity, message))


def _pad(s, width):
    return s + " " * max(0, width - len(s))


# ──────────────────────────────────────────────────────────────────────────────
# Report sections
# ──────────────────────────────────────────────────────────────────────────────

def section(title):
    w = 72
    return f"\n{'=' * w}\n  {title}\n{'=' * w}"


def subsection(title):
    return f"\n--- {title} ---"


def fmt(v, decimals=2):
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def table_row(cols, widths):
    return " | ".join(_pad(str(c), w) for c, w in zip(cols, widths))


def build_report(run_dir, tail=None):
    ALERTS.clear()
    lines = []

    # ── Locate data files ────────────────────────────────────────────────────
    ep_csv_path = os.path.join(run_dir, "debug", "episode_stats.csv")
    debug_jsonl_path = os.path.join(run_dir, "debug", "debug_stats.jsonl")
    has_ep_csv = os.path.exists(ep_csv_path)
    has_debug_jsonl = os.path.exists(debug_jsonl_path)

    episodes = load_episode_csv(ep_csv_path) if has_ep_csv else []
    debug_rows = load_debug_jsonl(debug_jsonl_path) if has_debug_jsonl else []
    mon_rewards, mon_lengths, mon_timesteps = load_monitor_csv(run_dir)

    if tail and episodes:
        episodes = episodes[-tail:]
    if tail and debug_rows:
        debug_rows = debug_rows[-tail:]

    n_ep = len(episodes)
    n_debug = len(debug_rows)

    total_ts = (
        max((r.get("timesteps", 0) for r in debug_rows), default=0)
        if debug_rows else
        (max(mon_timesteps) if mon_timesteps else 0)
    )

    # ── Header ───────────────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append("  ZELDA RL SMART AUDIT REPORT")
    lines.append(f"  Run    : {run_dir}")
    lines.append(f"  Tail   : {'all episodes' if not tail else f'last {tail} episodes'}")
    lines.append(f"  Episodes (ep CSV): {n_ep}")
    lines.append(f"  Debug snapshots  : {n_debug}")
    lines.append(f"  Total timesteps  : {total_ts:,}")
    lines.append("=" * 72)

    # ── Primary goal ─────────────────────────────────────────────────────────
    lines.append(section("PRIMARY GOAL — SWORD ACQUISITION"))
    if episodes:
        sword_eps = [e for e in episodes if e.get("has_sword")]
        shield_eps = [e for e in episodes if e.get("has_shield")]
        sword_rate = pct_true([e.get("has_sword") for e in episodes])
        shield_rate = pct_true([e.get("has_shield") for e in episodes])

        lines.append(f"  Sword acquired   : {len(sword_eps)} / {n_ep} episodes  ({sword_rate:.1f}%)")
        lines.append(f"  Shield acquired  : {len(shield_eps)} / {n_ep} episodes  ({shield_rate:.1f}%)")

        if sword_rate == 0:
            alert("CRITICAL", f"Sword NEVER acquired in {n_ep} episodes (0.0%)")
        elif sword_rate < 5:
            alert("WARNING", f"Sword acquired only {sword_rate:.1f}% of episodes — very rare")
        else:
            alert("OK", f"Sword acquisition rate: {sword_rate:.1f}%")

        if shield_rate < 30:
            alert("WARNING", f"Shield rate {shield_rate:.1f}% — agent struggling early game")
        elif shield_rate > 70:
            alert("OK", f"Shield rate {shield_rate:.1f}% — early game solid")
    else:
        lines.append("  [no episode CSV data]")
        alert("INFO", "No episode_stats.csv found. Run training first or upgrade callback.")

    # ── Reward analysis ──────────────────────────────────────────────────────
    lines.append(section("REWARD ANALYSIS"))
    if episodes:
        rewards = [e.get("reward_sum", 0) for e in episodes]
        explore_r = [e.get("reward_explore", 0) for e in episodes]
        event_r = [e.get("reward_event", 0) for e in episodes]
        kill_r = [e.get("reward_kill", 0) for e in episodes]
        stuck_r = [e.get("reward_stuck", 0) for e in episodes]
        fight_r = [e.get("reward_fight", 0) for e in episodes]

        mean_r = safe_mean(rewards)
        total_r_sum = sum(abs(v) for v in rewards) or 1e-9

        def pct_of_total(vals):
            s = sum(v for v in vals)
            return 100.0 * s / sum(rewards) if sum(rewards) != 0 else 0.0

        widths = [15, 10, 10, 12]
        lines.append(table_row(["Component", "Mean/ep", "% Total", "Trend"], widths))
        lines.append("-" * 52)

        xs = list(range(n_ep))

        def row(name, vals):
            slope, _ = linear_trend(xs, vals)
            return table_row(
                [name, fmt(safe_mean(vals)), fmt(pct_of_total(vals)) + "%", trend_arrow(slope)],
                widths
            )

        lines.append(row("explore", explore_r))
        lines.append(row("event", event_r))
        lines.append(row("kill", kill_r))
        lines.append(row("fight", fight_r))
        lines.append(row("stuck", stuck_r))
        lines.append(row("TOTAL", rewards))

        # Farming flag
        explore_pct = pct_of_total(explore_r)
        if explore_pct > 75:
            alert("CRITICAL", f"Explore reward dominates at {explore_pct:.1f}% of total — farming likely")
        elif explore_pct > 55:
            alert("WARNING", f"Explore reward at {explore_pct:.1f}% — monitor for farming")
        else:
            alert("OK", f"Explore reward at {explore_pct:.1f}% — within healthy range")

        # Reward trend
        slope_r, r2_r = linear_trend(xs, rewards)
        lines.append(f"\n  Reward trend : slope={slope_r:.4f}/ep, R²={r2_r:.3f}")
        if slope_r > 0 and r2_r > 0.1:
            alert("OK", f"Reward trending upward (slope={slope_r:.4f}, R²={r2_r:.3f})")
        elif slope_r < 0:
            alert("WARNING", f"Reward DECLINING (slope={slope_r:.4f}) — possible regression")
        else:
            alert("INFO", f"Reward trend flat/noisy (slope={slope_r:.4f}, R²={r2_r:.3f})")

        # Plateau check: last 10% vs first 10%
        n10 = max(1, n_ep // 10)
        early_mean = safe_mean(rewards[:n10])
        late_mean = safe_mean(rewards[-n10:])
        lines.append(f"  Early mean (first {n10} ep): {early_mean:.3f}")
        lines.append(f"  Late  mean (last  {n10} ep): {late_mean:.3f}")
        if late_mean <= early_mean * 1.02 and n_ep > 50:
            alert("WARNING", f"Possible plateau: late mean ({late_mean:.2f}) ≈ early mean ({early_mean:.2f})")

    else:
        lines.append("  [no episode CSV data]")

    # ── Farming detection ────────────────────────────────────────────────────
    lines.append(section("FARMING DETECTION"))
    if episodes:
        # High visited_locations + low event reward = farming pattern
        farming_eps = [
            e for e in episodes
            if e.get("visited_locations", 0) > 150
            and e.get("reward_event", 0) < 1.0
        ]
        farming_pct = 100.0 * len(farming_eps) / n_ep if n_ep else 0
        lines.append(f"  Episodes with >150 visited tiles but event_reward<1.0: {len(farming_eps)} / {n_ep} ({farming_pct:.1f}%)")

        # Cross-episode farming: lifetime_visited growth stall
        lv = [e.get("lifetime_visited", 0) for e in episodes]
        if len(lv) > 10:
            lv_growth = [lv[i] - lv[i-1] for i in range(1, len(lv))]
            mean_growth = safe_mean(lv_growth)
            lines.append(f"  Mean lifetime_visited growth per episode: {mean_growth:.1f} new tiles")
            if mean_growth < 5 and len(episodes) > 20:
                alert("WARNING", f"lifetime_visited barely growing ({mean_growth:.1f} new tiles/ep) — agent revisiting same area")
            else:
                alert("OK", f"lifetime_visited growing at {mean_growth:.1f} tiles/ep")

        if farming_pct > 30:
            alert("CRITICAL", f"Farming detected in {farming_pct:.1f}% of episodes — reward structure issue")
        elif farming_pct > 15:
            alert("WARNING", f"Moderate farming in {farming_pct:.1f}% of episodes")
        else:
            alert("OK", f"Low farming incidence ({farming_pct:.1f}%)")
    else:
        lines.append("  [no episode CSV data]")

    # ── Stuck analysis ───────────────────────────────────────────────────────
    lines.append(section("STUCK ANALYSIS"))
    if episodes:
        stuck_vals = [e.get("stuck_steps_final", 0) for e in episodes]
        med_stuck = safe_median(stuck_vals)
        high_stuck = [e for e in episodes if e.get("stuck_steps_final", 0) > 200]
        high_stuck_pct = 100.0 * len(high_stuck) / n_ep if n_ep else 0

        lines.append(f"  Median stuck_steps at episode end: {med_stuck:.0f}")
        lines.append(f"  Episodes ending with stuck_steps > 200: {len(high_stuck)} / {n_ep} ({high_stuck_pct:.1f}%)")

        # World distribution when stuck
        if high_stuck:
            world_counts = Counter(e.get("world_at_end", -1) for e in high_stuck)
            lines.append("  World distribution when stuck:")
            for world, count in world_counts.most_common(5):
                w_name = "overworld" if world == 1 else f"world {world}"
                lines.append(f"    World {world} ({w_name}): {count} stuck episodes")

        if high_stuck_pct > 20:
            alert("CRITICAL", f"{high_stuck_pct:.1f}% of episodes terminate stuck — increase stuck penalty or add state skipping")
        elif high_stuck_pct > 10:
            alert("WARNING", f"{high_stuck_pct:.1f}% of episodes end stuck")
        else:
            alert("OK", f"Stuck rate low ({high_stuck_pct:.1f}%)")
    else:
        lines.append("  [no episode CSV data]")

    # ── Curriculum analysis ──────────────────────────────────────────────────
    lines.append(section("CURRICULUM ANALYSIS"))
    if episodes:
        start_groups = defaultdict(list)
        for e in episodes:
            start_groups[e.get("start_state", "unknown")].append(e)

        widths = [20, 8, 10, 12, 12]
        lines.append(table_row(["start_state", "# eps", "% total", "shield_rate", "mean_reward"], widths))
        lines.append("-" * 66)
        for state, eps in sorted(start_groups.items(), key=lambda x: -len(x[1])):
            sr = pct_true([e.get("has_shield") for e in eps])
            mr = safe_mean([e.get("reward_sum", 0) for e in eps])
            pct = 100.0 * len(eps) / n_ep
            lines.append(table_row(
                [os.path.basename(state), str(len(eps)), f"{pct:.1f}%", f"{sr:.1f}%", f"{mr:.2f}"],
                widths
            ))

        # Detect imbalance
        if len(start_groups) > 1:
            counts = [len(v) for v in start_groups.values()]
            max_pct = 100.0 * max(counts) / n_ep
            if max_pct > 85:
                alert("WARNING", f"Curriculum heavily skewed: one state used in {max_pct:.1f}% of episodes")
    else:
        lines.append("  [no episode CSV data]")

    # ── Episode length analysis ──────────────────────────────────────────────
    lines.append(section("EPISODE LENGTH ANALYSIS"))
    if episodes:
        ep_lens = [e.get("episode_length", 0) for e in episodes]
        lines.append(f"  Mean length  : {safe_mean(ep_lens):.0f} steps")
        lines.append(f"  Median length: {safe_median(ep_lens):.0f} steps")
        lines.append(f"  Min / Max    : {min(ep_lens)} / {max(ep_lens)}")

        short_eps = [e for e in episodes if e.get("episode_length", 0) < 500]
        if len(short_eps) > 0.2 * n_ep:
            alert("WARNING", f"{100*len(short_eps)/n_ep:.1f}% of episodes end before 500 steps — agent dying or bad reset?")
        else:
            alert("OK", f"Episode length distribution looks healthy (mean={safe_mean(ep_lens):.0f})")
    else:
        lines.append("  [no episode CSV data]")

    # ── Top 5 best episodes ──────────────────────────────────────────────────
    lines.append(section("TOP 5 EPISODES BY REWARD"))
    if episodes:
        top5 = sorted(episodes, key=lambda e: e.get("reward_sum", 0), reverse=True)[:5]
        widths = [6, 10, 8, 7, 7, 10, 8, 20]
        lines.append(table_row(["ep", "timestep", "reward", "shield", "sword", "visited", "world", "start_state"], widths))
        lines.append("-" * 76)
        for e in top5:
            lines.append(table_row([
                str(e.get("episode", "?")),
                str(e.get("timestep", "?")),
                fmt(e.get("reward_sum", 0)),
                "✓" if e.get("has_shield") else "✗",
                "✓" if e.get("has_sword") else "✗",
                str(e.get("visited_locations", "?")),
                str(e.get("world_at_end", "?")),
                os.path.basename(str(e.get("start_state", "?"))),
            ], widths))
    else:
        lines.append("  [no episode CSV data]")

    # ── Debug snapshot analysis ──────────────────────────────────────────────
    lines.append(section("STEP-LEVEL DEBUG SNAPSHOT ANALYSIS"))
    if debug_rows:
        d_rewards = [r.get("reward_sum", 0) for r in debug_rows]
        d_coverage = [r.get("coverage_ratio", 0) for r in debug_rows]
        d_lifetime = [r.get("lifetime_visited", 0) for r in debug_rows]
        lines.append(f"  Snapshots         : {n_debug}")
        lines.append(f"  Mean reward_sum   : {safe_mean(d_rewards):.3f}")
        lines.append(f"  Mean coverage_ratio: {safe_mean(d_coverage):.4f}")
        lines.append(f"  Max lifetime_visited: {max(d_lifetime) if d_lifetime else 0}")

        # Speed estimate
        times = [r.get("episode_seconds", 0) for r in debug_rows if r.get("episode_seconds")]
        if times:
            mean_ep_s = safe_mean(times)
            lines.append(f"  Mean episode time : {mean_ep_s:.1f}s")
            steps_per_ep = safe_mean([r.get("step", 0) for r in debug_rows])
            if mean_ep_s > 0:
                steps_per_sec = steps_per_ep / mean_ep_s
                lines.append(f"  Throughput est.   : {steps_per_sec:.0f} steps/sec/env")
                alert("INFO", f"Throughput estimate: {steps_per_sec:.0f} steps/sec per env")
    else:
        lines.append("  [no debug_stats.jsonl found]")

    # ── Alert summary ────────────────────────────────────────────────────────
    lines.append(section("ALERT SUMMARY"))
    for severity, msg in ALERTS:
        prefix = {
            "CRITICAL": "[CRITICAL]",
            "WARNING":  "[WARNING ]",
            "OK":       "[   OK   ]",
            "INFO":     "[ INFO   ]",
        }.get(severity, "[???]")
        lines.append(f"  {prefix}  {msg}")

    # ── Recommendations ──────────────────────────────────────────────────────
    lines.append(section("RECOMMENDATIONS"))
    recs = []

    critical = [msg for sev, msg in ALERTS if sev == "CRITICAL"]
    warnings = [msg for sev, msg in ALERTS if sev == "WARNING"]

    if any("Sword NEVER" in m for m in critical):
        recs.append("[HIGH] Agent never got sword. Check saved.state starts — is the sword reachable? Consider increasing sword reward or creating an intermediate saved.state closer to the sword spawn.")
    if any("farming" in m.lower() for m in critical + warnings):
        recs.append("[HIGH] Farming detected. Consider: (1) reducing explore reward, (2) raising stuck penalty, (3) adding an explore_bonus_cap so single-room farming is bounded.")
    if any("stuck" in m.lower() for m in critical + warnings):
        recs.append("[HIGH] High stuck rate. Increase stuck penalty magnitude or reduce stuck_threshold from 64 to 32 so penalties fire faster.")
    if any("lifetime_visited barely growing" in m for m in warnings):
        recs.append("[MED] Agent is cycling the same tiles across episodes. The lifetime_visited reward is not pulling it forward. Consider adding a new saved.state from a more advanced position.")
    if any("Reward DECLINING" in m for m in warnings):
        recs.append("[HIGH] Reward is regressing — the loaded checkpoint may be overwriting good weights. Try a lower learning rate or increase checkpoint frequency.")
    if any("plateau" in m.lower() for m in warnings):
        recs.append("[MED] Reward has plateaued. Consider: (1) adjusting curriculum weights, (2) annealing ent_coef, (3) increasing n_envs for more diverse experience.")
    if not recs:
        recs.append("[LOW] No critical issues detected. Continue monitoring.")

    for rec in recs:
        lines.append(f"  {rec}")

    lines.append("\n" + "=" * 72)
    lines.append("  END OF REPORT")
    lines.append("=" * 72 + "\n")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Machine-readable Zelda RL run audit. Plain-text output, no plots."
    )
    parser.add_argument("run_folder", help="Path to run folder (e.g. run_sword_v2)")
    parser.add_argument(
        "--tail", type=int, default=None,
        help="Analyse only the last N episodes (default: all)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Write report to this file in addition to stdout"
    )
    args = parser.parse_args()

    report = build_report(args.run_folder, tail=args.tail)
    print(report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {args.out}")
