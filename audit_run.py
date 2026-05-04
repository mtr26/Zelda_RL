#!/usr/bin/env python3
"""
Automated audit script for Zelda RL training runs.
Usage: python audit_run.py <run_folder> [--compare <other_run>]
"""
import os
import json
import csv
import statistics
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_monitor_csv(monitor_path):
    """Load monitor.csv and extract rewards per episode."""
    rewards = []
    timesteps = []
    with open(monitor_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                parts = line.split(',')
                rewards.append(float(parts[0]))
                timesteps.append(float(parts[2]))
            except (ValueError, IndexError):
                continue
    return rewards, timesteps


def load_debug_stats(debug_path):
    """Load debug_stats.jsonl and extract per-episode metrics."""
    data = {
        'rewards': [],
        'coverage': [],
        'visited_locations': [],
        'sword_pickups': 0,
        'monsters_killed': 0,
        'events_triggered': 0,
        'worlds_visited': [],
        'health': [],
        'timesteps': [],
        'has_shield': 0,
        'max_reward_episode': None,
        'max_coverage_episode': None,
        'player_positions': defaultdict(list),  # world -> [(x, y), ...]
    }
    max_reward = -float('inf')
    max_coverage = -float('inf')
    
    with open(debug_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            reward = j.get('reward_sum', 0)
            coverage = j.get('coverage_ratio', 0)
            visited = j.get('visited_locations', 0)
            
            data['rewards'].append(reward)
            data['coverage'].append(coverage)
            data['visited_locations'].append(visited)
            data['timesteps'].append(j.get('timesteps', 0))
            data['health'].append(j.get('health', 0))
            
            if j.get('has_sword'):
                data['sword_pickups'] += 1
            if j.get('has_shield'):
                data['has_shield'] += 1
            
            data['monsters_killed'] += j.get('killed_monster', 0) or 0
            if j.get('reward_event', 0):
                data['events_triggered'] += 1
            
            world = j.get('world', 0)
            data['worlds_visited'].append(world)
            
            # Track player positions for heatmap
            px = j.get('player_x', 0)
            py = j.get('player_y', 0)
            if px and py:
                data['player_positions'][world].append((px, py))
            
            if reward > max_reward:
                max_reward = reward
                data['max_reward_episode'] = j
            if coverage > max_coverage:
                max_coverage = coverage
                data['max_coverage_episode'] = j
    
    return data


def compute_stats(run_folder):
    """Compute comprehensive statistics for a run."""
    monitor_path = os.path.join(run_folder, 'monitor.csv')
    debug_path = os.path.join(run_folder, 'debug', 'debug_stats.jsonl')
    
    stats = {'name': run_folder, 'exists': True}
    
    # Monitor stats
    if os.path.isfile(monitor_path):
        rewards, timesteps = load_monitor_csv(monitor_path)
        stats.update({
            'episodes': len(rewards),
            'mean_reward': statistics.mean(rewards) if rewards else None,
            'median_reward': statistics.median(rewards) if rewards else None,
            'min_reward': min(rewards) if rewards else None,
            'max_reward': max(rewards) if rewards else None,
            'stdev_reward': statistics.stdev(rewards) if len(rewards) > 1 else 0,
            'total_timesteps': max(timesteps) if timesteps else 0,
        })
        stats['monitor_rewards'] = rewards
        stats['monitor_timesteps'] = timesteps
    else:
        stats['episodes'] = 0
    
    # Debug stats
    if os.path.isfile(debug_path):
        debug = load_debug_stats(debug_path)
        stats.update({
            'debug_entries': len(debug['rewards']),
            'debug_mean_reward': statistics.mean(debug['rewards']) if debug['rewards'] else None,
            'debug_max_reward': max(debug['rewards']) if debug['rewards'] else None,
            'mean_coverage': statistics.mean(debug['coverage']) if debug['coverage'] else None,
            'max_coverage': max(debug['coverage']) if debug['coverage'] else None,
            'mean_visited_locations': statistics.mean(debug['visited_locations']) if debug['visited_locations'] else None,
            'max_visited_locations': max(debug['visited_locations']) if debug['visited_locations'] else None,
            'sword_pickups': debug['sword_pickups'],
            'monsters_killed': debug['monsters_killed'],
            'events_triggered': debug['events_triggered'],
            'shield_episodes': debug['has_shield'],
            'mean_health': statistics.mean(debug['health']) if debug['health'] else None,
            'unique_worlds': len(set(debug['worlds_visited'])),
        })
        stats['debug'] = debug
    else:
        stats['debug_entries'] = 0
    
    return stats


def format_report(stats):
    """Generate a formatted text report."""
    report = []
    report.append("=" * 80)
    report.append(f"AUDIT REPORT: {stats['name']}")
    report.append("=" * 80)
    report.append("")
    
    # Episode stats
    report.append("EPISODE STATISTICS:")
    report.append(f"  Total Episodes: {stats.get('episodes', 0)}")
    report.append(f"  Total Timesteps: {stats.get('total_timesteps', 0):,.0f}")
    report.append(f"  Mean Reward: {stats.get('mean_reward', 'N/A'):.2f}" if stats.get('mean_reward') else "  Mean Reward: N/A")
    report.append(f"  Median Reward: {stats.get('median_reward', 'N/A'):.2f}" if stats.get('median_reward') else "  Median Reward: N/A")
    report.append(f"  Min/Max Reward: {stats.get('min_reward', 'N/A'):.2f} / {stats.get('max_reward', 'N/A'):.2f}" if stats.get('max_reward') else "  Min/Max Reward: N/A")
    report.append(f"  StdDev: {stats.get('stdev_reward', 0):.2f}")
    report.append("")
    
    # Coverage stats
    report.append("EXPLORATION STATISTICS:")
    report.append(f"  Mean Coverage Ratio: {stats.get('mean_coverage', 'N/A'):.4f}" if stats.get('mean_coverage') else "  Mean Coverage Ratio: N/A")
    report.append(f"  Max Coverage Ratio: {stats.get('max_coverage', 'N/A'):.4f}" if stats.get('max_coverage') else "  Max Coverage Ratio: N/A")
    report.append(f"  Mean Visited Locations: {stats.get('mean_visited_locations', 'N/A'):.0f}" if stats.get('mean_visited_locations') else "  Mean Visited Locations: N/A")
    report.append(f"  Max Visited Locations: {stats.get('max_visited_locations', 'N/A'):.0f}" if stats.get('max_visited_locations') else "  Max Visited Locations: N/A")
    report.append(f"  Unique Worlds Visited: {stats.get('unique_worlds', 0)}")
    report.append("")
    
    # Item/Combat stats
    report.append("ITEM & COMBAT STATISTICS:")
    report.append(f"  Sword Pickups: {stats.get('sword_pickups', 0)} episodes")
    report.append(f"  Shield Pickups: {stats.get('shield_episodes', 0)} episodes")
    report.append(f"  Monsters Killed: {stats.get('monsters_killed', 0)}")
    report.append(f"  Mean Health: {stats.get('mean_health', 'N/A'):.1f}" if stats.get('mean_health') else "  Mean Health: N/A")
    report.append("")
    
    # Event stats
    report.append("EVENT STATISTICS:")
    report.append(f"  Events Triggered: {stats.get('events_triggered', 0)} episodes")
    report.append("")
    
    # Debug entries
    report.append("DEBUG STATISTICS:")
    report.append(f"  Debug Entries: {stats.get('debug_entries', 0)}")
    report.append(f"  Debug Mean Reward: {stats.get('debug_mean_reward', 'N/A'):.2f}" if stats.get('debug_mean_reward') else "  Debug Mean Reward: N/A")
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def plot_metrics(stats, output_dir):
    """Generate plots for the run."""
    os.makedirs(output_dir, exist_ok=True)
    
    debug = stats.get('debug', {})
    monitor_rewards = stats.get('monitor_rewards', [])
    
    # 1. Reward over episodes (from monitor.csv)
    if monitor_rewards:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(monitor_rewards, alpha=0.6, linewidth=0.8)
        ax.axhline(statistics.mean(monitor_rewards), color='r', linestyle='--', label=f"Mean: {statistics.mean(monitor_rewards):.2f}")
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title(f"{stats['name']} - Reward per Episode")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reward_curve.png'), dpi=100)
        plt.close()
    
    # 2. Coverage over episodes
    coverage = debug.get('coverage', [])
    if coverage:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(coverage, alpha=0.6, linewidth=0.8, color='green')
        ax.axhline(statistics.mean(coverage), color='r', linestyle='--', label=f"Mean: {statistics.mean(coverage):.4f}")
        ax.set_xlabel('Debug Entry')
        ax.set_ylabel('Coverage Ratio')
        ax.set_title(f"{stats['name']} - Coverage Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'coverage_curve.png'), dpi=100)
        plt.close()
    
    # 3. Visited locations over time
    visited = debug.get('visited_locations', [])
    if visited:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(visited, alpha=0.6, linewidth=0.8, color='purple')
        ax.axhline(statistics.mean(visited), color='r', linestyle='--', label=f"Mean: {statistics.mean(visited):.0f}")
        ax.set_xlabel('Debug Entry')
        ax.set_ylabel('Visited Locations')
        ax.set_title(f"{stats['name']} - Locations Explored per Episode")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'visited_locations.png'), dpi=100)
        plt.close()
    
    # 4. Reward vs Coverage scatter
    if debug.get('rewards') and coverage:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(coverage, debug['rewards'], alpha=0.5, s=20)
        ax.set_xlabel('Coverage Ratio')
        ax.set_ylabel('Reward')
        ax.set_title(f"{stats['name']} - Reward vs Coverage")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reward_vs_coverage.png'), dpi=100)
        plt.close()


def plot_coverage_heatmaps(stats, output_dir):
    """Generate coverage heatmaps showing where agent explored by world."""
    os.makedirs(output_dir, exist_ok=True)
    
    debug = stats.get('debug', {})
    player_positions = debug.get('player_positions', {})
    
    if not player_positions:
        return
    
    # Create heatmap for each world
    for world_id in sorted(player_positions.keys()):
        positions = player_positions[world_id]
        if not positions:
            continue
        
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        # Create 2D histogram (heatmap)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Game grid is typically 160x128 or similar, but we'll infer from data
        hist, xedges, yedges = np.histogram2d(
            xs, ys, bins=[32, 32], range=[[0, 256], [0, 256]]
        )
        
        im = ax.imshow(hist.T, origin='lower', cmap='hot', aspect='auto')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f"{stats['name']} - Coverage Heatmap (World {world_id})")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Visit Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'coverage_heatmap_world_{world_id}.png'), dpi=100)
        plt.close()
    
    # Create combined heatmap for all worlds
    if len(player_positions) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, world_id in enumerate(sorted(player_positions.keys())[:4]):
            positions = player_positions[world_id]
            if not positions:
                continue
            
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            
            hist, xedges, yedges = np.histogram2d(
                xs, ys, bins=[32, 32], range=[[0, 256], [0, 256]]
            )
            
            im = axes[idx].imshow(hist.T, origin='lower', cmap='hot', aspect='auto')
            axes[idx].set_xlabel('X')
            axes[idx].set_ylabel('Y')
            axes[idx].set_title(f"World {world_id}")
            plt.colorbar(im, ax=axes[idx])
        
        # Hide unused subplots
        for idx in range(len(player_positions), 4):
            axes[idx].set_visible(False)
        
        plt.suptitle(f"{stats['name']} - Coverage Heatmaps by World")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'coverage_heatmap_all_worlds.png'), dpi=100)
        plt.close()


def compare_runs(stats_list):
    """Generate comparison plot across runs."""
    names = [s['name'] for s in stats_list]
    mean_rewards = [s.get('mean_reward', 0) for s in stats_list]
    mean_coverage = [s.get('mean_coverage', 0) for s in stats_list]
    sword_pickups = [s.get('sword_pickups', 0) for s in stats_list]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].bar(names, mean_rewards, color='steelblue')
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Mean Reward Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(names, mean_coverage, color='green')
    axes[1].set_ylabel('Mean Coverage Ratio')
    axes[1].set_title('Mean Coverage Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    
    axes[2].bar(names, sword_pickups, color='orange')
    axes[2].set_ylabel('Sword Pickups')
    axes[2].set_title('Sword Pickups Comparison')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('comparison_runs.png', dpi=100)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit Zelda RL training runs.')
    parser.add_argument('run_folder', help='Path to run folder (e.g., run_10m_fixed)')
    parser.add_argument('--compare', nargs='+', help='Compare with other run folders')
    parser.add_argument('--output', default='audit_output', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Audit main run
    print(f"Auditing {args.run_folder}...")
    stats = compute_stats(args.run_folder)
    
    if stats.get('episodes', 0) == 0 and stats.get('debug_entries', 0) == 0:
        print(f"ERROR: No valid data found in {args.run_folder}")
        exit(1)
    
    # Print report
    report = format_report(stats)
    print(report)
    
    # Generate plots
    output_dir = os.path.join(args.output, stats['name'])
    print(f"\nGenerating plots to {output_dir}...")
    plot_metrics(stats, output_dir)
    print(f"✓ Plots saved: reward_curve.png, coverage_curve.png, visited_locations.png, reward_vs_coverage.png")
    
    # Generate coverage heatmaps
    plot_coverage_heatmaps(stats, output_dir)
    print(f"✓ Coverage heatmaps saved: coverage_heatmap_*.png")
    
    # Comparison
    if args.compare:
        print(f"\nComparing with {args.compare}...")
        compare_stats = [stats] + [compute_stats(r) for r in args.compare]
        compare_runs(compare_stats)
        print(f"✓ Comparison plot saved: comparison_runs.png")
    
    print(f"\n✓ Audit complete!")
