#!/usr/bin/env python3

import heapq
import random
import math
from collections import deque, defaultdict, Counter
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
        self.num_completed = 0
        self.samples = 0
        self.total = 0
        self.real_works = 0
        self.works = 0
        self.total_queued = 0
    
    @abstractmethod
    async def _enter(self, uid, works, is_real_work):
        pass


    
    def get_busy_reallyBusy_totalQueued(self):
        return self.works, self.real_works, self.total_queued

    async def enter(self, uid, works = 1, is_real_work = True):
        self.total_queued += 1
        await self._enter(uid, works, is_real_work)
        self.num_completed += 1

class MMC(async_server):
    def __init__(self, system, num_servers, serve_time, father = None):
        super().__init__(system, num_servers, serve_time)
        self.free_servers = asyncio.Semaphore(num_servers)
        self.father = father
    
    async def _enter(self, uid, works, is_real_work):
        base_queue = self if not self.father else self.father
        async with self.free_servers:
            base_queue.total_queued -= 1
            if is_real_work:
                base_queue.real_works += 1
            base_queue.works += 1
            event_future = asyncio.Future()
            self.system.push_event(self.system.T + works * (random.expovariate(1.0 / self.serve_time)), event_future)
            await event_future
            if is_real_work:
                base_queue.real_works -= 1
            base_queue.works -= 1


class C_MM1(async_server):    
    def __init__(self, system, num_servers, serve_time, scheduling_strategy):
        super().__init__(system, num_servers, serve_time)
        self.queue = deque()
        self.counter = Counter()
        self.scheduling_strategy = scheduling_strategy
        self.servers = [MMC(self.system, 1, self.serve_time, self) for _ in range(self.num_servers)]
    
    def get_least_busy_server(self):
        server = min(self.servers, key=lambda s: ((1 - s.free_servers._value) + len(s.free_servers._waiters), self.servers.index(s)))
        return server, self.servers.index(server)
        
    async def _enter(self, uid, works, is_real_work):
        if self.scheduling_strategy == "local_storage_sticky":
            server_idx = random.randrange(len(self.servers))
            server = self.servers[server_idx]
            
            self.queue.append(server_idx)
            self.counter[server_idx] += 1
            
            if len(self.queue) > len(self.servers):
                x = self.queue.popleft()
                self.counter[x] -= 1
                if self.counter[x] == 0:
                    del self.counter[x]
            self.samples += 1
            self.total += len(self.counter)
        elif self.scheduling_strategy == "local_storage_least_busy":
            server, server_idx = self.get_least_busy_server()
        else:
            raise ValueError(f"ERROR: unexpected scheduling strategy: {self.scheduling_strategy}")
        await server._enter(uid, works, is_real_work)

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
        
    @staticmethod
    def calculate_theoretical_bw2(steps, time_between_steps, step_time_in_gpu, total_gpus):
        """Calculate theoretical bandwidth metrics"""
        gpu_requests_per_second = total_gpus / (steps * step_time_in_gpu)
        minimal_agent_max_bw = total_gpus * (1 + time_between_steps / step_time_in_gpu)
        return gpu_requests_per_second, minimal_agent_max_bw 
    
    def __init__(self, disk_size_in_blocks, steps, allow_holes_recalculation, \
        num_inflight_agents, iterations, random_placement_on_miss, ranges, evict_on_miss, disk, first_conv_id, \
        time_between_steps, total_gpus, step_time_in_gpu, context_window_size, force_hit_ratio, scheduling_strategy, is_use_theoretical_agents, \
        print_statistics, storage_blocks_per_second):
        if disk_size_in_blocks % ranges != 0:
            print(f"Error: disk_size_in_blocks ({disk_size_in_blocks}) must be divisible by ranges ({ranges})")
            exit(1)
        
        if not (0.0 <= force_hit_ratio <= 1.0):
            print(f"Error: force_hit_ratio ({force_hit_ratio}) must be between 0.0 and 1.0")
            exit(1)

        gpu_requests_per_second, minimal_agent_max_bw = self.calculate_theoretical_bw2(steps, time_between_steps, step_time_in_gpu, total_gpus) 
        
        self.disk_size_in_blocks = disk_size_in_blocks
        self.allow_holes_recalculation = allow_holes_recalculation
        self.num_inflight_agents = num_inflight_agents if not is_use_theoretical_agents else int(minimal_agent_max_bw)
        self.iterations = iterations
        self.random_placement_on_miss = random_placement_on_miss
        self.ranges = ranges
        self.evict_on_miss = evict_on_miss
        self.range_len = self.disk_size_in_blocks // self.ranges
        self.force_hit_ratio = force_hit_ratio
        self.scheduling_strategy = scheduling_strategy

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
        if scheduling_strategy == "shared_storage_least_busy":
            self.gpus = MMC(self, total_gpus, step_time_in_gpu)
        else:
            self.gpus = C_MM1(self, total_gpus, step_time_in_gpu, scheduling_strategy)
        self.storage = MMC(self, 1, 1 / self.storage_blocks_per_second) if self.storage_blocks_per_second else None

        self.conversation_manager = ConversationManager(self, time_between_steps, first_conv_id, steps, context_window_size)

        self.completed_conversations = 0
        self.inflight_conversation_count = 0
        self.final_T = 1
        self.final_completed_count = 0

        self.print_input_params(context_window_size, steps, total_gpus, scheduling_strategy, is_use_theoretical_agents,
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
            disable_all, to_read, to_calc = False, 0, 0
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
                    to_calc += 1
                elif self.storage:
                    to_read += 1       
            if self.storage and to_read > 0:
                await self.storage.enter(conv.conv_id, to_read)
            if self.scheduling_strategy == 'local_storage_least_busy':
                _, server_idx = self.gpus.get_least_busy_server() 
                if (server_idx != (conv.conv_id % self.gpus.num_servers)) and (step > 0):
                    to_calc = len(conv.kvs)
            if self.gpus and to_calc > 0:
                await self.gpus.enter(conv.conv_id, works=to_calc, is_real_work=False)
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
            if (self.iterations - self.completed_conversations < self.num_inflight_agents) and (self.final_completed_count == 0):
                self.final_T = self.T        
                self.final_completed_count = self.completed_conversations
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
        
    def print_input_params(self, context_window_size, steps, total_gpus, scheduling_strategy, is_use_theoretical_agents, 
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
        print(f"scheduling_strategy: {scheduling_strategy}")
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
        print(f"Total Simulation Time (T): {self.final_T:.2f}")
        print(f"\033[1;34mSimulation Requests Per Second: {self.final_completed_count / self.final_T:.8f}, Max Possible Reqs Per Second: {self.gpu_requests_per_second:.8f}\033[0m")
        print("\033[1;33m=============================\033[0m")
        return hit_rate

    async def monitor_gpus(self):
        total_busy_servers = 0
        total_really_busy_servers = 0
        total_queue_len = 0
        total_sleepers = 0
        sample_count = 0

        while not self.terminate:
            await asyncio.sleep(0.1)
            busy, really_busy, total_queued = self.gpus.get_busy_reallyBusy_totalQueued()
            total_busy_servers += busy
            total_really_busy_servers += really_busy
            total_queue_len += total_queued
            sleepers = self.agent_outside_service.num_servers - self.agent_outside_service.free_servers._value
            total_sleepers += sleepers
            sample_count += 1 
            
            curr_busy_ratio = busy / self.gpus.num_servers
            curr_really_busy_ratio = really_busy / self.gpus.num_servers
            avg_busy_ratio = total_busy_servers / sample_count / self.gpus.num_servers 
            avg_really_busy_ratio = total_really_busy_servers / sample_count / self.gpus.num_servers
            avg_queue_len = total_queue_len / sample_count 
            avg_sleepers = total_sleepers / sample_count



            avg_busy_servers = self.gpus.total / self.gpus.samples if self.gpus.samples > 0 else "NOT_IMPLEMENTED"
            print(f"T={self.T:.2f} - Curr: Busy={curr_busy_ratio:.4f}, ReallyBusy={curr_really_busy_ratio:.4f}, Queue={total_queued:.0f}, Sleepers={sleepers:.0f}")
            print(f"T={self.T:.2f} - Avg:  Busy={avg_busy_ratio:.4f}, ReallyBusy={avg_really_busy_ratio:.4f}, Queue={avg_queue_len:.2f}, Sleepers={avg_sleepers:.2f}")
            print(f"T={self.T:.2f} - Avg Busy Servers: {avg_busy_servers}")
            print("-" * 80)