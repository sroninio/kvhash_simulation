#!/usr/bin/env python3

import argparse
import random
from collections import defaultdict


def main(num_blocks_in_storage, kvcache_len_blocks, num_requests, iterations):    
    reqs_to_blocks = defaultdict(lambda: [-1] * kvcache_len_blocks)
    block_to_request = defaultdict(lambda: -1)
    hit_counter = 0
    for i in range(iterations):
        curr_request = i % num_requests
        for block in reqs_to_blocks[curr_request]:
            if block_to_request[block] == curr_request:
                hit_counter += 1 
        reqs_to_blocks[curr_request] = random.choices(range(num_blocks_in_storage), k=kvcache_len_blocks)
        for block in reqs_to_blocks[curr_request]:
            block_to_request[block] = curr_request
    
    cache_hit_ratio = hit_counter / (iterations * kvcache_len_blocks)
    print(f"Cache Hit Ratio: {cache_hit_ratio:.4f} ({hit_counter}/{iterations * kvcache_len_blocks})")
     

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Simulate KV cache behavior with configurable parameters"
    )
    
    parser.add_argument(
        "--num_blocks_in_storage",
        type=int,
        required=True,
        help="Number of blocks available in storage"
    )
    
    parser.add_argument(
        "--kvcache_len_blocks",
        type=int,
        required=True,
        help="Length of KV cache in blocks"
    )
    
    parser.add_argument(
        "--num_requests",
        type=int,
        required=True,
        help="Number of inflight requests, this simulates the number of alive requests"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="Number of iterations to simulate"
    )
    
    args = parser.parse_args()
    main(
        num_blocks_in_storage=args.num_blocks_in_storage,
        kvcache_len_blocks=args.kvcache_len_blocks,
        num_requests=args.num_requests,
        iterations=args.iterations
    )

