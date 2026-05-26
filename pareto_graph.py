#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from simulation_logic import System

BOLD_RED = '\033[1;31m'
RESET = '\033[0m'
X_LABEL = 'agents/user/sec'
Y_LABEL = 'agents/sec'
CONCURRENCY_X_LABEL = 'concurrency'
FIELDNAMES = [
    'config_label', 'num_inflight_agents', 'iterations', 'minimal_inflight',
    'actual_agents_per_second', 'agents_per_second_per_inflight', 'hit_rate', 'T',
]
CONF_FILES = [
    'limited_hbm_only.json',
    'limited_hbm_and_cmx_waterfall.json',
    'limited_hbm_and_cmx_neighbours.json',
    'cmx_only.json',
    'inifinite_storage.json',
    'recompute.json',
]
def params_from_config(config, num_inflight_agents):
    params = {
        'blocks_buffers': config['blocks_buffers'],
        'storage_tiers': config['storage_tiers'],
        'allow_holes_recalculation': config.get('allow_holes_recalculation', 1),
        'random_placement_on_miss': config.get('random_placement_on_miss', 0),
        'evict_on_miss': config.get('evict_on_miss', 1),
        'num_inflight_agents': num_inflight_agents,
        'steps': config['steps'],
        'iterations': config['iterations'],
        'time_between_steps': config['time_between_steps'],
        'total_gpus': config.get('total_gpus', 1),
        'step_time_in_gpu': config.get('step_time_in_gpu', 1),
        'is_linear_step_time': config['is_linear_step_time'],
        'force_hit_ratio': config.get('force_hit_ratio', 0),
        'scheduling_strategy': config.get('scheduling_strategy', 'shared_storage_least_busy'),
        'is_use_theoretical_agents': config.get('is_use_theoretical_agents', 0),
        'debug_full_kvc_each_step': config.get('debug_full_kvc_each_step', False),
        'debug_fixed_between_steps_time': config.get('debug_fixed_between_steps_time', False),
        'monitor_interval_virtual_time': 0,
    }
    for i, tier in enumerate(params['storage_tiers']):
        t = tier['type']
        if t == 'memos' and tier['num_blocks'] % tier['num_ranges'] != 0:
            raise ValueError(
                f"storage_tiers[{i}] num_blocks ({tier['num_blocks']}) must be divisible by num_ranges ({tier['num_ranges']})"
            )
        if t not in ('lru', 'memos', 'allocrelease'):
            raise ValueError(f"storage_tiers[{i}] unknown type {t!r}")
    return params


def load_series(data_file):
    series = {}
    with open(data_file) as f:
        for row in csv.DictReader(f):
            series.setdefault(row['config_label'], []).append(row)
    return series


def annotate_concurrency(xs, ys, rows):
    for x, y, row in zip(xs, ys, rows):
        plt.annotate(str(row['num_inflight_agents']), (x, y), textcoords='offset points', xytext=(4, 4), fontsize=7)


def plot_pareto(series, output, min_x=None):
    plt.figure(figsize=(10, 6))
    for label in sorted(series):
        rows = sorted(series[label], key=lambda r: int(r['num_inflight_agents']))
        if min_x is not None:
            rows = [r for r in rows if float(r['agents_per_second_per_inflight']) >= min_x]
        if not rows:
            continue
        xs = [float(r['agents_per_second_per_inflight']) for r in rows]
        ys = [float(r['actual_agents_per_second']) for r in rows]
        plt.plot(xs, ys, marker='o', linewidth=2, label=label)
        annotate_concurrency(xs, ys, rows)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title('Pareto: throughput vs per-agent throughput')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Plot saved as {output}")


def plot_concurrency(series, output):
    plt.figure(figsize=(10, 6))
    for label in sorted(series):
        rows = sorted(series[label], key=lambda r: int(r['num_inflight_agents']))
        xs = [int(r['num_inflight_agents']) for r in rows]
        ys = [float(r['actual_agents_per_second']) for r in rows]
        plt.plot(xs, ys, marker='o', linewidth=2, label=label)
        for x, y, row in zip(xs, ys, rows):
            hr = row.get('hit_rate')
            if hr not in (None, ''):
                plt.annotate(f"{float(hr) * 100:.1f}%", (x, y), textcoords='offset points', xytext=(4, 4), fontsize=7)
    plt.xlabel(CONCURRENCY_X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title('Throughput vs concurrency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Plot saved as {output}")


def plot_from_csv(data_file, output, concurrency_output=None, min_x=None):
    series = load_series(data_file)
    plot_pareto(series, output, min_x)
    if concurrency_output:
        plot_concurrency(series, concurrency_output)


def concurrency_output_path(output):
    stem, ext = os.path.splitext(output)
    return stem + '_concurrency' + ext


def _pareto_sweep_task(task):
    return pareto_sweep(*task)


def pareto_sweep(config_file, label, n_sweeps, inflight_factor):
    with open(config_file) as f:
        config = json.load(f)
    label = label or os.path.splitext(os.path.basename(config_file))[0]
    probe = System(params_from_config(config, 1))
    minimal = probe.minimal_inflight_agents_for_max_possible_agents_per_second
    max_inflight = inflight_factor * minimal
    inflights = [max(1, int(round(x))) for x in np.linspace(1, max_inflight, n_sweeps)]
    rows, xs, ys = [], [], []
    print(f"{BOLD_RED}=== pareto label={label} config={config_file} points={len(inflights)} ==={RESET}")
    for i, n in enumerate(inflights, 1):
        params = params_from_config(config, n)
        params['iterations'] = n * 1000
        print(f"{BOLD_RED}=== sweep {i}/{len(inflights)} [{label}] ==={RESET}")
        print(json.dumps(params, indent=2))
        system = System(params)
        system.simulate()
        actual = params['iterations'] / system.T
        x = actual / n
        total = system.hits + system.misses
        hit_rate = system.hits / total if total else 0
        xs.append(x)
        ys.append(actual)
        rows.append({
            'config_label': label,
            'num_inflight_agents': n,
            'iterations': params['iterations'],
            'minimal_inflight': minimal,
            'actual_agents_per_second': actual,
            'agents_per_second_per_inflight': x,
            'hit_rate': hit_rate,
            'T': system.T,
        })
        print(f"[{label}] num_inflight_agents={n} actual_agents_per_second={actual:.6f} x={x:.6f} hit_rate={hit_rate * 100:.1f}%")
    return label, rows, xs, ys


def write_csv(data_file, all_rows, fresh):
    if fresh and os.path.exists(data_file):
        os.remove(data_file)
    write_header = fresh or not os.path.exists(data_file)
    with open(data_file, 'w' if fresh else 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerows(all_rows)


def resolve_labels(config_files, labels):
    if not labels:
        return [None] * len(config_files)
    if len(labels) == 1:
        return labels * len(config_files)
    if len(labels) != len(config_files):
        raise ValueError(f"expected 0, 1, or {len(config_files)} labels, got {len(labels)}")
    return labels


def setup_run_dir(results_root, config_files, sweeps, inflight_factor, jobs, run_label=None, min_x=None):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f"{stamp}_{run_label}" if run_label else stamp
    run_dir = os.path.join(results_root, name)
    os.makedirs(run_dir, exist_ok=True)
    seen = {}
    copied = []
    for cfg in config_files:
        base = os.path.basename(cfg)
        if base in seen:
            seen[base] += 1
            stem, ext = os.path.splitext(base)
            dest_name = f"{stem}_{seen[base]}{ext}"
        else:
            seen[base] = 0
            dest_name = base
        shutil.copy2(cfg, os.path.join(run_dir, dest_name))
        copied.append(dest_name)
    with open(os.path.join(run_dir, 'run_meta.json'), 'w') as f:
        json.dump({
            'sweeps': sweeps,
            'inflight_factor': inflight_factor,
            'jobs': jobs,
            'run_label': run_label,
            'min_x': min_x,
            'config_files': copied,
        }, f, indent=2)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file", action="append", default=None)
    parser.add_argument("-o", "--output", type=str, default="pareto_graph.png")
    parser.add_argument("-d", "--data-file", type=str, default="pareto_results.csv")
    parser.add_argument("-l", "--label", action="append", default=None)
    parser.add_argument("-j", "--jobs", type=int, default=None)
    parser.add_argument("--sweeps", type=int, default=10, help="number of inflight points from 1 to factor*minimal")
    parser.add_argument("--inflight-factor", type=float, default=2.0)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--run-label", type=str, default=None, help="optional suffix for results run directory")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--concurrency", action="store_true", help="also plot concurrency -> agents/sec")
    parser.add_argument("--concurrency-output", type=str, default=None)
    parser.add_argument("--fresh-csv", action="store_true")
    parser.add_argument("--min-x", type=float, default=None, help="plot pareto points with agents/user/sec >= k")
    args = parser.parse_args()

    conc_out = args.concurrency_output or (concurrency_output_path(args.output) if args.concurrency else None)

    if args.plot_only:
        plot_from_csv(args.data_file, args.output, conc_out, args.min_x)
        raise SystemExit

    if not args.config_file:
        if args.plot_only:
            parser.error("at least one -f/--config_file is required unless --plot-only")
        args.config_file = list(CONF_FILES)
    labels = resolve_labels(args.config_file, args.label)
    jobs = args.jobs or min(len(args.config_file), os.cpu_count() or 1)
    run_dir = setup_run_dir(args.results_dir, args.config_file, args.sweeps, args.inflight_factor, jobs, args.run_label, args.min_x)
    data_file = os.path.join(run_dir, os.path.basename(args.data_file))
    output = os.path.join(run_dir, os.path.basename(args.output))
    print(f"Run directory: {run_dir}")
    tasks = [
        (cfg, lbl, args.sweeps, args.inflight_factor)
        for cfg, lbl in zip(args.config_file, labels)
    ]

    conc_out = os.path.join(run_dir, os.path.basename(conc_out)) if conc_out else None

    if len(tasks) == 1:
        _, rows, xs, ys = pareto_sweep(*tasks[0])
        all_rows = rows
        write_csv(data_file, all_rows, True)
        print(f"Data written to {data_file} (label={rows[0]['config_label']})")
        series = {rows[0]['config_label']: rows}
        plot_pareto(series, output, args.min_x)
        if conc_out:
            plot_concurrency(series, conc_out)
    else:
        print(f"Running {len(tasks)} configs with {jobs} parallel workers")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_pareto_sweep_task, tasks))
        all_rows = [row for _, rows, _, _ in results for row in rows]
        write_csv(data_file, all_rows, True)
        print(f"Data written to {data_file} ({len(results)} configs, {len(all_rows)} rows)")
        plot_from_csv(data_file, output, conc_out, args.min_x)
