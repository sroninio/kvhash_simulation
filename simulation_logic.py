#!/usr/bin/env python3

import heapq
import random
from collections import deque


class Block:
    def __init__(self, offset):
        self.owner_conv_id = -1
        self.owner_pos_in_conv = -1
        self.offset = offset
    
    def is_belongs_to(self, conv_id, pos_in_conv):
        return (self.owner_conv_id == conv_id) and (self.owner_pos_in_conv == pos_in_conv)
    
    def take_ownership(self, conv_id, pos_in_conv):
        self.owner_conv_id = conv_id
        self.owner_pos_in_conv = pos_in_conv
            


class Conversation:
    def __init__(self, conv_id, conversation_length, disk_size_in_blocks, context_window_len):
        self.conv_id = conv_id
        self.conversation_length = conversation_length 
        self.finished_steps = 0
        self.context_window_len = context_window_len
        self.kvs = deque()

    def is_finished(self):
        return (self.finished_steps >= self.conversation_length)
    

class Disk:
    def __init__(self, size):
        self.disk = [Block(indx) for indx in range(size)] 


class System:
    def __init__(self, disk_size_in_blocks, steps, allow_holes_recalculation, \
        num_inflight_agents, iterations, random_placement_on_miss, ranges, evict_on_miss, disk, first_conv_id, \
        time_between_steps, total_gpus, step_time_in_gpu, to_check_cache_hit_rate, context_window_size, force_hit_ratio):
        if disk_size_in_blocks % ranges != 0:
            print(f"Error: disk_size_in_blocks ({disk_size_in_blocks}) must be divisible by ranges ({ranges})")
            exit(1)
        
        if not (0.0 <= force_hit_ratio <= 1.0):
            print(f"Error: force_hit_ratio ({force_hit_ratio}) must be between 0.0 and 1.0")
            exit(1)

        
        
        self.disk_size_in_blocks = disk_size_in_blocks
        self.steps = steps
        self.allow_holes_recalculation = allow_holes_recalculation
        self.num_inflight_agents = num_inflight_agents
        self.iterations = iterations
        self.random_placement_on_miss = random_placement_on_miss
        self.ranges = ranges
        self.evict_on_miss = evict_on_miss
        self.range_len = self.disk_size_in_blocks // self.ranges
        self.time_between_steps = time_between_steps
        self.total_gpus = total_gpus
        self.step_time_in_gpu = step_time_in_gpu
        self.to_check_cache_hit_rate = to_check_cache_hit_rate
        self.context_window_size = context_window_size
        self.force_hit_ratio = force_hit_ratio
        

        self.prev_conv_id = first_conv_id
        self.events = []  # Min heap for events
        self.disk = disk
        self.finished_conversations = 0
        self.inflights = 0
        self.T = 0
        self.misses = 0
        self.hits = 0
        self.conversations_queue = deque()
        self.free_gpus = self.total_gpus
        self.event_counter = 0  # Tie-breaker for heap events
        
        total_time_spent_between_steps = (self.steps + 1) * time_between_steps
        total_time_in_gpu = self.steps * self.step_time_in_gpu
        gpu_requests_per_second = self.total_gpus * (1 / total_time_in_gpu) 
        total_agent_time = total_time_in_gpu + total_time_spent_between_steps
        minimal_agent_max_bw = gpu_requests_per_second * total_agent_time

        
        print("\033[1;33m\n=============================\033[0m")
        print("CACHE PERF ANALISS")        
        print(f"disk_size_in_blocks: {self.disk_size_in_blocks}")
        print(f"allow_holes_recalculation: {self.allow_holes_recalculation}")
        print(f"num_inflight_agents: {self.num_inflight_agents}")
        print(f"iterations: {self.iterations}")
        print(f"random_placement_on_miss: {self.random_placement_on_miss}")
        print(f"ranges: {self.ranges}")
        print(f"evict_on_miss: {self.evict_on_miss}")
        print(f"context_window_size: {self.context_window_size}")
        print(f"disk to data set ratio: {self.disk_size_in_blocks / (self.num_inflight_agents * self.steps)}")
        print("BW PERF ANALISS")        
        print(f"total_gpus: {self.total_gpus}")
        print(f"step_time_in_gpu: {self.step_time_in_gpu}")
        print(f"steps: {self.steps}")
        print(f"time_between_steps: {self.time_between_steps}") 
        print(f"\033[1;31mnum_inflight_agents: {self.num_inflight_agents}\033[0m")
        print("BW THEORETICAL NUMBERS")        
        print(f"\033[1;34mmaximal possible gpu requests per second: {gpu_requests_per_second:.8f}\033[0m")
        print(f"\033[1;31mminimal_agent_max_bw: {minimal_agent_max_bw:.2f}\033[0m")
        


    def alloc_block(self, block):
        prev_range_idx = block.offset // self.range_len if block else -1
        while True:
            range_idx = random.randrange(self.ranges)
            if range_idx != prev_range_idx or self.ranges == 1:
                break
        offset_in_range = random.randrange(self.range_len) if ((not block) or (self.random_placement_on_miss)) else (block.offset % self.range_len)
        block_offset = range_idx * self.range_len + offset_in_range 
        return self.disk.disk[block_offset] 


    def handle_conversation_return_from_sleep_event(self, conv):
        conv.phase_in_step = 0
        conv.disable_all = False
        conv.produced_new_block_in_this_step = False
        self.free_gpus -= 1 
        self.handle_conversation_return_from_gpu_event(conv)

    def handle_conversation_return_from_gpu_event(self, conv):
        self.free_gpus += 1
        if len(self.conversations_queue) > 0 and self.free_gpus > 0:
            waiting_conv = self.conversations_queue.popleft()
            self.enter_conv_to_gpu(waiting_conv)       
        while conv.phase_in_step < len(conv.kvs):
            valid_kv = True
            if self.to_check_cache_hit_rate:
                kv, indx_in_conversation = conv.kvs.popleft()
                valid_kv = kv.is_belongs_to(conv.conv_id, indx_in_conversation) 
                conv.disable_all = conv.disable_all or ((not self.allow_holes_recalculation) and (not valid_kv))
                if (not conv.disable_all) and valid_kv:
                    self.hits += 1
                else:
                    self.misses += 1
                if not valid_kv and self.evict_on_miss:
                    kv = self.alloc_block(kv)
                    kv.take_ownership(conv.conv_id, indx_in_conversation)
                conv.kvs.append((kv, indx_in_conversation)) 
            conv.phase_in_step += 1
            to_recalculate = (random.random() > self.force_hit_ratio) if self.force_hit_ratio > 0 else (not valid_kv)
            if to_recalculate:
                self.enter_conv_to_gpu(conv)
                return
        if not conv.produced_new_block_in_this_step:
            conv.produced_new_block_in_this_step = True
            if self.to_check_cache_hit_rate:
                kv = self.alloc_block(None)
                kv.take_ownership(conv.conv_id, conv.finished_steps)
                conv.kvs.append((kv, conv.finished_steps))
                if len(conv.kvs) > conv.context_window_len:
                    conv.kvs.popleft()
            self.enter_conv_to_gpu(conv)
            return
        conv.finished_steps += 1
        self.enter_conv_to_sleep(conv)
            
        
    def handle_statistic_event(self):
        self.push_event(self.T + 0.1, {'type': 'stat'})

    def get_unique_id(self):
        self.prev_conv_id += 1
        return self.prev_conv_id
    
    def push_event(self, time, event_dict):
        """Helper function to push events to heap with tie-breaker counter."""
        self.event_counter += 1
        heapq.heappush(self.events, (time, self.event_counter, event_dict))
    
    def enter_conv_to_gpu(self, conv):
        if self.free_gpus > 0:
            self.free_gpus -= 1
            self.push_event(self.T + random.random() * 2 * self.step_time_in_gpu, {'type': 'gpu_finished', 'conv': conv})
        else:
            self.conversations_queue.append(conv)
    
    def enter_conv_to_sleep(self, conv):
        if not conv.is_finished():
            self.push_event(self.T + random.random() * 2 * self.time_between_steps, {'type': 'conv', 'conv': conv})
        else:
            self.finished_conversations += 1
            self.inflights -= 1


    def simulate(self):
        self.push_event(self.T + 0.1, {'type': 'stat'})
        while self.finished_conversations < self.iterations:
            t, counter, args = heapq.heappop(self.events)
            self.T = t
            if args['type'] == "stat":
                self.handle_statistic_event()
            elif args['type'] == 'conv':
                conv = args['conv']
                self.handle_conversation_return_from_sleep_event(conv)
            else: #gpu finished
                conv = args['conv'] 
                self.handle_conversation_return_from_gpu_event(conv)
            while self.inflights < self.num_inflight_agents:
                conv = Conversation(self.get_unique_id(), self.steps, self.disk_size_in_blocks, self.context_window_size)
                self.inflights += 1
                self.enter_conv_to_sleep(conv)
                
        
        # Print cache statistics
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        print("CACHE RESULTS")        
        print(f"Cache Hit Rate: {hit_rate:.2f}% ({self.hits}/{total})")
        print(f"Hits: {self.hits}, Misses: {self.misses}")
        print("BW RESULTS")        
        print(f"\033[1;34mSimulation Requests Per Second: {self.iterations / self.T:.8f}\033[0m")
        print("\033[1;33m=============================\033[0m")
        return hit_rate, self.T, self.iterations

