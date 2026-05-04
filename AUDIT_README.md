# Zelda RL Training Audit Script

Automated audit tool for analyzing Zelda RL training runs. Generates comprehensive statistics, plots, and comparison reports.

## Features

- **Episode Statistics**: mean/median/min/max rewards, standard deviation, total timesteps
- **Exploration Metrics**: coverage ratio, visited locations, unique worlds
- **Item & Combat Tracking**: sword pickups, shield pickups, monsters killed, health
- **Event Analysis**: number of events triggered per run
- **Visualizations**: 
  - Reward curve per episode
  - Coverage ratio progression
  - Visited locations per episode
  - Reward vs Coverage scatter plot
  - **Coverage heatmaps by world** (NEW!) — shows spatial distribution of exploration
  - Cross-run comparison bar charts

## Usage

### Single Run Audit
```bash
python audit_run.py run_10m_fixed
```

Output:
- Console report with all metrics
- Plots saved to `audit_output/run_10m_fixed/`:
  - `reward_curve.png` — Episode rewards over training
  - `coverage_curve.png` — Coverage ratio progression
  - `visited_locations.png` — Exploration breadth per episode
  - `reward_vs_coverage.png` — Correlation scatter plot
  - `coverage_heatmap_world_*.png` — Spatial exploration per world (heat = visit frequency)
  - `coverage_heatmap_all_worlds.png` — 4-panel comparison of all worlds explored

### Compare Multiple Runs
```bash
python audit_run.py run_10m_fixed --compare run_5m run_20m_long
```

Output: (same as above) + `comparison_runs.png` with side-by-side bar charts

### Custom Output Directory
```bash
python audit_run.py run_10m_fixed --output my_audits/
```

## Report Metrics Explained

| Metric | Meaning |
|--------|---------|
| Mean Reward | Average episode reward (higher is better) |
| Median Reward | 50th percentile reward (robust to outliers) |
| Coverage Ratio | Avg fraction of game grid visited (0-1, higher = more exploration) |
| Visited Locations | Avg unique cells explored per episode |
| Sword Pickups | Episodes where agent found/equipped sword |
| Shield Pickups | Episodes where agent found/equipped shield |
| Monsters Killed | Total combat victories |
| Events Triggered | Episodes where game events (milestones) were triggered |
| Unique Worlds | Number of distinct game worlds explored |

## Example Workflow

```bash
# Run training to completion
tmux new-session -d -s training_long
tmux send-keys -t training_long ".venv/bin/python Train.py --log_dir run_20m_long ..." Enter

# Later, audit all runs
python audit_run.py run_5m
python audit_run.py run_10m_fixed
python audit_run.py run_20m_long --compare run_5m run_10m_fixed

# View plots
open audit_output/run_20m_long/reward_curve.png
open comparison_runs.png
```

## Output Structure

```
audit_output/
├── run_5m/
│   ├── reward_curve.png
│   ├── coverage_curve.png
│   ├── visited_locations.png
│   ├── reward_vs_coverage.png
│   ├── coverage_heatmap_world_0.png
│   ├── coverage_heatmap_world_1.png
│   └── coverage_heatmap_all_worlds.png
├── run_10m_fixed/
│   ├── reward_curve.png
│   └── ...
└── run_20m_long/
    └── ...
comparison_runs.png  (only if --compare used)
```

## Notes

- Requires `matplotlib` and `numpy` (included in project dependencies)
- Works with any run folder containing `monitor.csv` and `debug/debug_stats.jsonl`
- Heatmaps extract player X/Y coordinates from debug logs — requires your env to log `player_x` and `player_y`
- Comparison plot shows: mean reward, mean coverage, sword pickups across all runs
- Script handles missing files gracefully (skips unavailable metrics)
