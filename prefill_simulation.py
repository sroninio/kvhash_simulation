#!/usr/bin/env python3

import argparse
import pandas as pd
from datetime import datetime
from simulation_logic import System, Disk

def main(disk_size_in_blocks, allow_holes_recalculation, random_placement_on_miss, evict_on_miss, agents_list, steps_list, ranges_list, sim_ratio, iterations, time_between_steps, total_gpus, step_time_in_gpu, context_window_size, force_hit_ratio, is_shared_storage, is_use_theoretical_agents, output_file):
    disk = Disk(disk_size_in_blocks)
    first_conv_id = 0
    results = []
    for agents in agents_list:
        for steps in steps_list:
            for ranges_val in ranges_list:
                system = System(
                    disk_size_in_blocks=disk_size_in_blocks // sim_ratio,
                    steps=steps,
                    allow_holes_recalculation=allow_holes_recalculation,
                    num_inflight_agents=agents // sim_ratio,
                    iterations=iterations,
                    random_placement_on_miss=random_placement_on_miss,
                    ranges=ranges_val,
                    evict_on_miss=evict_on_miss,
                    disk=disk,
                    first_conv_id=first_conv_id,
                    time_between_steps=time_between_steps,
                    total_gpus=total_gpus,
                    step_time_in_gpu=step_time_in_gpu,
                    context_window_size=context_window_size if context_window_size > 0 else steps,
                    force_hit_ratio=force_hit_ratio,
                    is_shared_storage=is_shared_storage,
                    is_use_theoretical_agents=is_use_theoretical_agents
                )
                hit_rate, total_time, total_iterations, theoretical_rate, minimal_agent_max_bw, actual_rate = system.simulate()
                first_conv_id = system.conversation_manager.conv_id + 10
                
                # Collect results
                results.append({
                    'agents': agents,
                    'steps': steps,
                    'ranges': ranges_val,
                    'disk_size_in_blocks': disk_size_in_blocks,
                    'disk_usage' : 1 /(disk_size_in_blocks / (agents * steps)),
                    'hit_rate': hit_rate,
                    'total_time': total_time,
                    'total_iterations': total_iterations,
                    'minimal_agent_max_rate' : minimal_agent_max_bw,
                    'theoretical_rate_req_sec' : theoretical_rate,
                    'actual_rate_req_sec' : actual_rate,
                    'TTFT' : agents / actual_rate
                })
    
    # Write results to Excel
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"\nResults written to {output_file}")
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--disk_size_in_blocks",
        type=int,
        default=0,
        help="Disk size in blocks (default: 0)"
    )
    
    parser.add_argument(
        "--allow_holes_recalculation",
        type=int,
        default=1,
        help="Allow holes recalculation (default: 1)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=1500,
        help="Number of iterations (default: 1500) Total number of conversations to simulate"
    )
    
    parser.add_argument(
        "--random_placement_on_miss",
        type=int,
        default=0,
        help="Random placement on miss (default: 0)"
    )
    
    parser.add_argument(
        "--evict_on_miss",
        type=int,
        default=1,
        help="Evict on miss (default: 1)"
    )
    
    parser.add_argument(
        "--agents_list",
        type=int,
        nargs='+',
        default=[1000, 2000, 4000, 8000, 16000, 32000, 64000],
        help="List of inflight conversations to test (default: 1000 2000 4000 8000 16000 32000 64000)"
    )
    
    parser.add_argument(
        "--steps_list",
        type=int,
        nargs='+',
        default=[10, 50, 100, 150],
        help="List of step values to test (default: 10 50 100 150)"
    )
    
    parser.add_argument(
        "--ranges_list",
        type=int,
        nargs='+',
        default=[1],
        help="List of ranges values to test (default: 1 4 10)"
    )
    
    parser.add_argument(
        "--sim_ratio",
        type=int,
        default=1,
        help="Simulation ratio divider (default: 10)"
    )
    
    parser.add_argument(
        "--time_between_steps",
        type=float,
        required=True,
        help="Time between steps"
    )
    
    parser.add_argument(
        "--total_gpus",
        type=int,
        default=1,
        help="Total number of GPUs (default: 1)"
    )
    
    parser.add_argument(
        "--step_time_in_gpu",
        type=float,
        required=True,
        help="Step time in GPU"
    )
    
    parser.add_argument(
        "--context_window_size",
        type=int,
        default=0,
        help="Context window size in kv blocks (default: 0)"
    )
    
    parser.add_argument(
        "--force_hit_ratio",
        type=float,
        default=0.0,
        help="Force hit ratio, value between 0.0 and 1.0 (default: 0.0)"
    )
    
    parser.add_argument(
        "--is_shared_storage",
        type=int,
        default=1,
        help="Use shared GPU storage (1) or non-shared (0) (default: 1)"
    )
    
    parser.add_argument(
        "--is_use_theoretical_agents",
        type=int,
        default=0,
        help="Use theoretical agents count (1) or actual agents (0) (default: 0)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        help="Output file path (default: simulation_results_<timestamp>.xlsx)"
    )
    
    args = parser.parse_args()
    
    main(
        disk_size_in_blocks=args.disk_size_in_blocks,
        allow_holes_recalculation=args.allow_holes_recalculation,
        random_placement_on_miss=args.random_placement_on_miss,
        evict_on_miss=args.evict_on_miss,
        agents_list=args.agents_list,
        steps_list=args.steps_list,
        ranges_list=args.ranges_list,
        sim_ratio=args.sim_ratio,
        iterations=args.iterations,
        time_between_steps=args.time_between_steps,
        total_gpus=args.total_gpus,
        step_time_in_gpu=args.step_time_in_gpu,
        context_window_size=args.context_window_size,
        force_hit_ratio=args.force_hit_ratio,
        is_shared_storage=args.is_shared_storage,
        is_use_theoretical_agents=args.is_use_theoretical_agents,
        output_file=args.output_file
    )
