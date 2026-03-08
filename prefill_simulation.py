#!/usr/bin/env python3

import argparse
import asyncio
import pandas as pd
import json
from datetime import datetime
from simulation_logic import System, Disk

async def main(disk_size_in_blocks, allow_holes_recalculation, random_placement_on_miss, evict_on_miss, agents_list, steps_list, ranges_list, sim_ratio, iterations, time_between_steps, total_gpus, step_time_in_gpu, context_window_size, force_hit_ratio, scheduling_strategy, is_use_theoretical_agents, print_statistics, storage_blocks_per_second, output_file):
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
                    scheduling_strategy=scheduling_strategy,
                    is_use_theoretical_agents=is_use_theoretical_agents,
                    print_statistics=print_statistics,
                    storage_blocks_per_second=storage_blocks_per_second
                )
                hit_rate, total_time, total_iterations, theoretical_rate, minimal_agent_max_bw, actual_rate = await system.simulate()
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
    
    df = pd.DataFrame(results)
    
    if output_file.endswith('.xlsx'):
        output_file_excel = output_file
        output_file_csv = output_file.replace('.xlsx', '.csv')
    elif output_file.endswith('.csv'):
        output_file_csv = output_file
        output_file_excel = output_file.replace('.csv', '.xlsx')
    else:
        output_file_excel = output_file + '.xlsx'
        output_file_csv = output_file + '.csv'
    
    df.to_excel(output_file_excel, index=False)
    df.to_csv(output_file_csv, index=False)
    print(f"\nResults written to {output_file_excel} and {output_file_csv}")
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-f", "--config_file",
        type=str,
        required=True,
        help="Path to JSON config file"
    )
    
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    params = {
        'disk_size_in_blocks': config.get('disk_size_in_blocks', 1),
        'allow_holes_recalculation': config.get('allow_holes_recalculation', 1),
        'random_placement_on_miss': config.get('random_placement_on_miss', 0),
        'evict_on_miss': config.get('evict_on_miss', 1),
        'agents_list': config['agents_list'],
        'steps_list': config['steps_list'],
        'ranges_list': config['ranges_list'],
        'sim_ratio': config.get('sim_ratio', 1),
        'iterations': config['iterations'],
        'time_between_steps': config.get('time_between_steps', 1),
        'total_gpus': config.get('total_gpus', 1),
        'step_time_in_gpu': config.get('step_time_in_gpu', 1),
        'context_window_size': config.get('context_window_size', 0),
        'force_hit_ratio': config.get('force_hit_ratio', 0),
        'scheduling_strategy': config.get('scheduling_strategy', 'shared_storage_least_busy'),
        'is_use_theoretical_agents': config.get('is_use_theoretical_agents', 0),
        'print_statistics': config.get('print_statistics', 1),
        'storage_blocks_per_second': config.get('storage_blocks_per_second', 0.0),
        'output_file': config.get('output_file', f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    }
    
    asyncio.run(main(**params))
