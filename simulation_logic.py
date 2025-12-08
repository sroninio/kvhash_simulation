#!/usr/bin/env python3

import heapq
import random
import math
from collections import deque, defaultdict
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
import asyncio


class IOType(Enum):
    READ = 0
    WRITE = 1

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
            
class Disk:
    def __init__(self, size):
        self.disk = [Block(indx) for indx in range(size)] 

class Conversation:
    def __init__(self, conv_id, conversation_length, context_window_len):
        self.conv_id = conv_id
        self.conversation_length = conversation_length 
        self.finished_steps = 0
        self.context_window_len = context_window_len
        self.kvs = deque()


class ConversationManager:
    def __init__(self, system, sleep_time_between_steps, first_conv_id, steps, context_window_size):
        self.sleep_time_between_steps = sleep_time_between_steps
        self.system = system
        self.conv_id = first_conv_id
        self.steps = steps
        self.context_window_size = context_window_size

    def get_unique_id(self):
        self.conv_id += 1
        return self.conv_id
    
    def create_conversation(self):
        conv = Conversation(self.get_unique_id(), self.steps, self.context_window_size)
        return conv

class async_server(ABC):
    def __init__(self, system, num_servers, serve_time):
        self.system = system
        self.num_servers = num_servers
        self.serve_time = serve_time
        self.inflights = 0
        self.num_completed = 0
    
    @abstractmethod
    async def _enter(self, uid):
        pass
    
    async def enter(self, uid):
        self.inflights += 1
        await self._enter(uid)
        self.inflights -= 1
        self.num_completed += 1

class MMC(async_server):
    def __init__(self, system, num_servers, serve_time):
        super().__init__(system, num_servers, serve_time)
        self.free_servers = asyncio.Semaphore(num_servers)
    
    async def _enter(self, uid):
        async with self.free_servers:
            event_future = asyncio.Future()
            self.system.push_event(self.system.T + random.expovariate(1.0 / self.serve_time), event_future)
            await event_future

class C_MM1(async_server):    
    def __init__(self, system, num_servers, serve_time):
        super().__init__(system, num_servers, serve_time)
        self.servers = [MMC(self.system, 1, self.serve_time) for _ in range(self.num_servers)]
        self.uid_to_server = defaultdict(lambda: random.choice(self.servers))
    
    async def _enter(self, uid):
        server = self.uid_to_server[uid]
        await server._enter(uid)

class System:
    @staticmethod
    def calculate_theoretical_bw(steps, time_between_steps, step_time_in_gpu, total_gpus):
        """Calculate theoretical bandwidth metrics"""
        total_time_spent_between_steps = (steps + 1) * time_between_steps
        total_time_in_gpu = steps * step_time_in_gpu
        gpu_requests_per_second = total_gpus * (1 / total_time_in_gpu) 
        total_agent_time = total_time_in_gpu + total_time_spent_between_steps
        minimal_agent_max_bw = gpu_requests_per_second * total_agent_time
        return gpu_requests_per_second, minimal_agent_max_bw
    

    
    def __init__(self, disk_size_in_blocks, steps, allow_holes_recalculation, \
        num_inflight_agents, iterations, random_placement_on_miss, ranges, evict_on_miss, disk, first_conv_id, \
        time_between_steps, total_gpus, step_time_in_gpu, context_window_size, force_hit_ratio, is_shared_storage, is_use_theoretical_agents, \
        print_statistics, storage_blocks_per_second):
        if disk_size_in_blocks % ranges != 0:
            print(f"Error: disk_size_in_blocks ({disk_size_in_blocks}) must be divisible by ranges ({ranges})")
            exit(1)
        
        if not (0.0 <= force_hit_ratio <= 1.0):
            print(f"Error: force_hit_ratio ({force_hit_ratio}) must be between 0.0 and 1.0")
            exit(1)

        gpu_requests_per_second, minimal_agent_max_bw = self.calculate_theoretical_bw(steps, time_between_steps, step_time_in_gpu, total_gpus) 
        
        self.disk_size_in_blocks = disk_size_in_blocks
        self.allow_holes_recalculation = allow_holes_recalculation
        self.num_inflight_agents = num_inflight_agents if not is_use_theoretical_agents else int(minimal_agent_max_bw)
        self.iterations = iterations
        self.random_placement_on_miss = random_placement_on_miss
        self.ranges = ranges
        self.evict_on_miss = evict_on_miss
        self.range_len = self.disk_size_in_blocks // self.ranges
        self.force_hit_ratio = force_hit_ratio

        self.gpu_requests_per_second = gpu_requests_per_second 
        self.minimal_agent_max_bw = minimal_agent_max_bw


        self.events = []  # Min heap for events
        self.disk = disk
        self.T = 0
        self.misses = 0
        self.hits = 0
        self.event_counter = 0  # Tie-breaker for heap events
        self.terminate = False
        self.print_statistics = print_statistics
        self.storage_blocks_per_second = storage_blocks_per_second

        self.agent_outside_service = MMC(self, self.num_inflight_agents, time_between_steps)
        self.gpus = MMC(self, total_gpus, step_time_in_gpu) if is_shared_storage else C_MM1(self, total_gpus, step_time_in_gpu)
        self.storage = MMC(self, 1, 1 / self.storage_blocks_per_second) if self.storage_blocks_per_second else None

        self.conversation_manager = ConversationManager(self, time_between_steps, first_conv_id, steps, context_window_size)

        self.completed_conversations = 0
        self.inflight_conversation_count = 0

        self.print_input_params(context_window_size, steps, total_gpus, is_shared_storage, is_use_theoretical_agents,
                            step_time_in_gpu, time_between_steps)
        


    def alloc_block(self, block):
        prev_range_idx = block.offset // self.range_len if block else -1
        while True:
            range_idx = random.randrange(self.ranges)
            if range_idx != prev_range_idx or self.ranges == 1:
                break
        offset_in_range = random.randrange(self.range_len) if ((not block) or (self.random_placement_on_miss)) else (block.offset % self.range_len)
        block_offset = range_idx * self.range_len + offset_in_range 
        return self.disk.disk[block_offset] 

    async def async_handle_conversation(self, conv):
        for step in range(conv.conversation_length):
            disable_all = False
            for _ in range(len(conv.kvs)):
                kv, indx_in_conversation = conv.kvs.popleft()
                valid_kv = kv.is_belongs_to(conv.conv_id, indx_in_conversation) if not self.force_hit_ratio else (random.random() < self.force_hit_ratio)
                disable_all = disable_all or ((not self.allow_holes_recalculation) and (not valid_kv))
                if (not disable_all) and (valid_kv):
                    self.hits += 1
                else:
                    self.misses += 1
                if not valid_kv and self.evict_on_miss:
                    kv = self.alloc_block(kv)
                    kv.take_ownership(conv.conv_id, indx_in_conversation)
                conv.kvs.append((kv, indx_in_conversation)) 
                if not valid_kv:
                    await self.gpus.enter(conv.conv_id)
                elif self.storage:
                    await self.storage.enter(conv.conv_id)
            kv = self.alloc_block(None)
            kv.take_ownership(conv.conv_id, step)
            conv.kvs.append((kv, step))
            if len(conv.kvs) > conv.context_window_len:
                conv.kvs.popleft()
            await self.gpus.enter(conv.conv_id)    
            await self.agent_outside_service.enter(conv.conv_id)
        
    
    def push_event(self, time, event_dict):
        """Helper function to push events to heap with tie-breaker counter."""
        self.event_counter += 1
        heapq.heappush(self.events, (time, self.event_counter, event_dict))
    
    async def simulate(self):
        if self.print_statistics:
            monitor_task = asyncio.create_task(self.monitor_gpus())
        
        def on_conversation_done(task):
            self.inflight_conversation_count -= 1
            self.completed_conversations += 1
            if self.completed_conversations % 10000 == 0:
                print(f"Done {self.completed_conversations}")
        
        while self.completed_conversations < self.iterations:
            while self.inflight_conversation_count < min(self.num_inflight_agents,  self.iterations - self.completed_conversations):
                conv = self.conversation_manager.create_conversation()
                task = asyncio.create_task(self.async_handle_conversation(conv))
                task.add_done_callback(on_conversation_done)
                self.inflight_conversation_count += 1                 
            if not self.events:
                await asyncio.sleep(0)
                continue   
            t, counter, future = heapq.heappop(self.events)
            self.T = t
            future.set_result(None)  # Notify the future
            await asyncio.sleep(0)  # Yield to let the notified task run
      
        self.terminate = True
        
        pending = asyncio.all_tasks() - {asyncio.current_task()}
        if pending:
            await asyncio.gather(*pending)
        
        hit_rate = self.print_output_params()
        
        return hit_rate, self.T, self.iterations, self.gpu_requests_per_second, self.minimal_agent_max_bw, self.iterations / self.T
        
    def print_input_params(self, context_window_size, steps, total_gpus, is_shared_storage, is_use_theoretical_agents, 
                        step_time_in_gpu, time_between_steps):
        """Print simulation parameters"""
        print("\033[1;33m\n=============================\033[0m")
        print("CACHE PERF ANALISS")        
        print(f"disk_size_in_blocks: {self.disk_size_in_blocks}")
        print(f"allow_holes_recalculation: {self.allow_holes_recalculation}")
        print(f"num_inflight_agents: {self.num_inflight_agents}")
        print(f"iterations: {self.iterations}")
        print(f"random_placement_on_miss: {self.random_placement_on_miss}")
        print(f"ranges: {self.ranges}")
        print(f"evict_on_miss: {self.evict_on_miss}")
        print(f"context_window_size: {context_window_size}")
        print(f"disk to data set ratio: {self.disk_size_in_blocks / (self.num_inflight_agents * steps)}")
        print("BW PERF ANALISS")        
        print(f"total_gpus: {total_gpus}")
        print(f"is_shared_storage: {is_shared_storage}")
        print(f"is_use_theoretical_agents: {is_use_theoretical_agents}")
        print(f"step_time_in_gpu: {step_time_in_gpu}")
        print(f"steps: {steps}")
        print(f"time_between_steps: {time_between_steps}") 
        print(f"\033[1;31mnum_inflight_agents: {self.num_inflight_agents}\033[0m")
        print("BW THEORETICAL NUMBERS")        
        print(f"\033[1;34mmaximal possible gpu requests per second: {self.gpu_requests_per_second:.8f}\033[0m")
        print(f"\033[1;31mminimal_agent_max_bw: {self.minimal_agent_max_bw:.2f}\033[0m")
    
    def print_output_params(self):
        """Print simulation output"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        print("CACHE RESULTS")        
        print(f"Cache Hit Rate: {hit_rate:.2f}% ({self.hits}/{total})")
        print(f"Hits: {self.hits}, Misses: {self.misses}")
        print("BW RESULTS")
        print(f"Total Simulation Time (T): {self.T:.2f}")
        print(f"\033[1;34mSimulation Requests Per Second: {self.iterations / self.T:.8f}\033[0m")
        print("\033[1;33m=============================\033[0m")
        return hit_rate

    async def monitor_gpus(self):
        total_free_servers = 0
        sample_count = 0      
        while not self.terminate:
            await asyncio.sleep(0.1)
            if hasattr(self.gpus, 'free_servers'):
                free = self.gpus.free_servers._value
                capacity = self.gpus.num_servers
                queue_size = len(self.gpus.free_servers._waiters)
                total_free_servers += free
                sample_count += 1
                avg = total_free_servers / sample_count
                print(f"T={self.T:.2f} - Free: {free}/{capacity}, Queue: {queue_size}, Avg Free: {avg:.2f}")
            else:
                # For NonSharedGpus, show each GPU's free servers
                free_counts = [gpu.free_servers._value for gpu in self.gpus.gpus]
                queue_sizes = [len(gpu.free_servers._waiters) for gpu in self.gpus.gpus]
                avg_free = sum(free_counts) / len(free_counts)
                total_free_servers += avg_free
                sample_count += 1
                overall_avg = total_free_servers / sample_count
                print(f"T={self.T:.2f} - Free: {free_counts}, Queue: {queue_sizes}, Avg Free: {overall_avg:.2f}")
