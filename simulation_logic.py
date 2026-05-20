#!/usr/bin/env python3

import enum
import heapq
from locale import locale_alias
from pickle import TRUE
import random
import math
from collections import deque, defaultdict, Counter
from abc import ABC, abstractmethod
from typing import Any
from enum import Enum
from functools import partial

from numpy import True_


class IOType(Enum):
    READ = 0
    WRITE = 1

class Waitable:
    def __init__(self):
        self.refcnt = 0

    def add_waiter(self):
        self.refcnt += 1

    def remove_waiter(self):
        self.refcnt -= 1

    def all_waited(self):
        return self.refcnt == 0


            

class Conversation:
    def __init__(self, conv_id, conversation_length):
        self.conv_id = conv_id
        self.conversation_length = conversation_length 
        self.kvs = []


class ConversationManager:
    def __init__(self, system, sleep_time_between_steps, first_conv_id, steps):
        self.sleep_time_between_steps = sleep_time_between_steps
        self.system = system
        self.conv_id = first_conv_id
        self.steps = steps

    def get_unique_id(self):
        self.conv_id += 1
        return self.conv_id
    
    def create_conversation(self):
        conv = Conversation(self.get_unique_id(), self.steps)
        return conv

class HandleWrapper:
    def __init__(self, system, conv):
        self.system = system
        self.conv = conv
        self._gen = self._run()

    def start(self):
        next(self._gen)

    def resume(self):
        try:
            next(self._gen)
            return False
        except StopIteration:
            return True

    def _run(self):
        waitable = Waitable()
        conv = self.conv
        system = self.system
        for step in range(conv.conversation_length):
            missing = [True] * len(conv.kvs)
            should_wait = system.storage.try_read(self, conv.kvs, missing, waitable)
            n_miss = sum(missing)
            n_hit = len(conv.kvs) - n_miss
            system.hits += n_hit
            system.misses += n_miss
            if should_wait:
                yield
            if n_miss > 0:
                system.gpus.async_enter(self, waitable, 17, works=n_miss, is_real_work=False) 
                yield
            system.gpus.async_enter(self, waitable, 17, works=1, is_real_work=True) 
            yield
            conv.kvs.append(random.getrandbits(64) + 1) 
            system.storage.write([conv.kvs[i] for i in range(len(missing)) if missing[i]] + [conv.kvs[-1]])
            system.agent_outside_service.async_enter(self, waitable, 17, works=1) 
            yield

class async_server(ABC):
    def __init__(self, system, num_servers, serve_time):
        self.system = system
        self.num_servers = num_servers
        self.serve_time = serve_time
        self.real_works = 0
        self.works = 0
        self.total_queued = 0
        self.waiters_queue = deque()  # FIFO: append on block, popleft on wake
    

    @abstractmethod
    def _async_enter(self, handler, waitable, uid, works, is_real_work):
        pass

    def get_busy_reallyBusy_totalQueued(self):
        return self.works, self.real_works, self.total_queued

    def async_enter(self,handler,  waitable, uid, works=1, is_real_work=True):
        return self._async_enter(handler, waitable, uid, works, is_real_work)

class MMC(async_server):
    def __init__(self, system, num_servers, serve_time, father = None):
        super().__init__(system, num_servers, serve_time)
        self.free_server_slots = num_servers
        self.father = father
    
    def _async_enter(self, handler, waitable, uid, works, is_real_work):
        serve_time = works * (random.expovariate(1.0 / self.serve_time))
        self.system.add_work_to_system(handler, self, serve_time, is_real_work, waitable)
    
    def enter_queue(self, handler, serve_time, is_real_work, waitable):
        self.total_queued += 1
        self.waiters_queue.append((handler, serve_time, is_real_work, waitable))
    
    def exit_queue(self):
        self.total_queued -= 1
        return self.waiters_queue.popleft()
    
    def enter_service(self, is_real_work):
        base_queue = self if not self.father else self.father
        self.free_server_slots -= 1
        if is_real_work:
            base_queue.real_works += 1
        base_queue.works += 1
    
    def exit_service(self, is_real_work):
        base_queue = self if not self.father else self.father
        self.free_server_slots += 1
        if is_real_work:
            base_queue.real_works -= 1
        base_queue.works -= 1 


class C_MM1(async_server):    
    def __init__(self, system, num_servers, serve_time, scheduling_strategy):
        super().__init__(system, num_servers, serve_time)
        self.scheduling_strategy = scheduling_strategy
        self.servers = [MMC(self.system, 1, self.serve_time, self) for _ in range(self.num_servers)]
    
    def get_least_busy_server(self):
        def load(s):
            busy = s.num_servers - s.free_server_slots
            return busy + len(s.waiters_queue)
        server = min(self.servers, key=lambda s: (load(s), self.servers.index(s)))
        return server, self.servers.index(server)

    def _async_enter(self, handler, waitable, uid, works, is_real_work):
        if self.scheduling_strategy == "local_storage_sticky":
            server_idx = random.randrange(len(self.servers))
            server = self.servers[server_idx]
        elif self.scheduling_strategy == "local_storage_least_busy":
            server, server_idx = self.get_least_busy_server()
        else:
            raise ValueError(f"ERROR: unexpected scheduling strategy: {self.scheduling_strategy}")
        server._async_enter(handler, waitable, uid, works, is_real_work) 

class Tier(C_MM1):
    def __init__(self, system, num_queues, block_serve_time, num_blocks):
        super().__init__(system, num_queues, block_serve_time, "local_storage_sticky")
        self.num_blocks = num_blocks

    @abstractmethod
    def write(self, keys):
        pass

    @abstractmethod
    def remove(self, key):
        pass
    
    @abstractmethod
    def has_key(self, key):
        pass

    def read(self, handler, waitable, num_blocks):
        if self.num_servers == 0:
            return False
        else:
            self.async_enter(handler, waitable, 17, num_blocks)
            return True

    def try_read(self, handler, kvs, missing, waitable):
        found_count = 0
        for i in range(len(kvs)):
            if not missing[i]:
                continue
            if self.has_key(kvs[i]):
                missing[i] = False
                found_count += 1
        return self.read(handler, waitable, found_count) if found_count else False

class DOCA_MEMOS(Tier):
    def __init__(self, system, num_queues, block_serve_time, num_blocks, num_ranges):
        super().__init__(system, num_queues, block_serve_time, num_blocks)
        self.range_len = num_blocks // num_ranges
        self.num_ranges = num_ranges
        print("System: constructing (this may take a while for large disk_size_in_blocks)...")
        self.disk = [-1] * num_blocks
    
    def write(self, keys):
        for key in keys:
            range_idx = random.randrange(self.num_ranges)
            offset_in_range = key % self.range_len
            block_offset = range_idx * self.range_len + offset_in_range
            self.disk[block_offset] = key
        return []
    
    def remove(self, key):
        pass

    def has_key(self, key):
        offset_in_range = key % self.range_len
        for range_idx in range(self.num_ranges):
            if self.disk[range_idx * self.range_len + offset_in_range] == key:
                return True
        return False


class LRU(Tier):
    def __init__(self, system, num_queues, block_serve_time, num_blocks):
        super().__init__(system, num_queues, block_serve_time, num_blocks) 
        self.keys = set()
        self.lru = deque()

    def write(self, keys_to_write):
        evicted = []
        for key in set(keys_to_write) - self.keys:
            self.keys.add(key)
            self.lru.append(key)
            if len(self.keys) > self.num_blocks:
                old = self.lru.popleft()
                self.keys.discard(old)
                evicted.append(old)
        return evicted

    def remove(self, key):
        pass

    def has_key(self, key):
        return key in self.keys

def build_storage_manager(system, params):
    tiers = []
    for cfg in params['storage_tiers']:
        t = cfg['type']
        num_blocks = cfg['num_blocks']
        num_queues = cfg['num_queues']
        block_serve_time = cfg['block_serve_time']
        if t == 'lru':
            tiers.append(LRU(system, num_queues, block_serve_time, num_blocks))
        elif t == 'memos':
            num_ranges = cfg['num_ranges']
            if num_blocks % num_ranges != 0:
                raise ValueError(f"memos num_blocks ({num_blocks}) must be divisible by num_ranges ({num_ranges})")
            tiers.append(DOCA_MEMOS(system, num_queues, block_serve_time, num_blocks, num_ranges))
        else:
            raise ValueError(f"unknown storage tier type: {t!r}")
    return StorageManager(tiers)

class StorageManager:
    def __init__(self, tiers_list):
        self.tiers_list = tiers_list
    
    def write(self, keys):
        keys_to_write = keys
        tier_indx = 0
        while tier_indx < len(self.tiers_list) and keys_to_write:
            keys_to_write = self.tiers_list[tier_indx].write(keys_to_write)
            tier_indx += 1

    def try_read(self, handler, kvs, missing, waitable):
        return any(tier.try_read(handler, kvs, missing, waitable) for tier in self.tiers_list)

  

class System:
    def calculate_theoretical_bw(self):
        """Calculate theoretical bandwidth metrics"""
        gpu_max_possible_agents_per_second = self.params['total_gpus'] / (
            self.params['steps'] * self.params['step_time_in_gpu']
        )
        minimal_inflight_agents_for_max_possible_agents_per_second = self.params['total_gpus'] * (
            1 + self.params['time_between_steps'] / self.params['step_time_in_gpu']
        )
        return gpu_max_possible_agents_per_second, minimal_inflight_agents_for_max_possible_agents_per_second

    def __init__(self, params):
        self.params = params
        self.gpu_max_possible_agents_per_second, self.minimal_inflight_agents_for_max_possible_agents_per_second = self.calculate_theoretical_bw()
        self.num_inflight_agents = self.params['num_inflight_agents'] if not self.params['is_use_theoretical_agents'] else int(self.minimal_inflight_agents_for_max_possible_agents_per_second)

        self.events = []  # Min heap for events
        self.T = 0
        self.event_counter = 0  # Tie-breaker for heap events


        self.agent_outside_service = MMC(self, self.num_inflight_agents, self.params['time_between_steps'])
        if self.params['scheduling_strategy'] == "shared_storage_least_busy":
            self.gpus = MMC(self, self.params['total_gpus'], self.params['step_time_in_gpu'])
        else:
            self.gpus = C_MM1(self, self.params['total_gpus'], self.params['step_time_in_gpu'], self.params['scheduling_strategy'])

        self.storage = build_storage_manager(self, self.params)

        self.conversation_manager = ConversationManager(self, self.params['time_between_steps'], 0, self.params['steps'])

        self.completed_conversations = 0
        self.inflight_conversation_count = 0
        self.misses = 0
        self.hits = 0
        self.total_busy_servers = 0
        self.total_really_busy_servers = 0
        self.total_queue_len = 0
        self.total_sleepers = 0
        self.sample_count = 0


    def push_event(self, time, event_dict):
        """Helper function to push events to heap with tie-breaker counter."""
        self.event_counter += 1
        heapq.heappush(self.events, (time, self.event_counter, event_dict))

    def add_work_to_system(self, handler, mmc, serve_time, is_real_work, waitable):
        waitable.add_waiter()
        if (mmc.free_server_slots <= 0) or (len(mmc.waiters_queue) > 0):
            mmc.enter_queue(handler, serve_time, is_real_work, waitable)
        else:
            mmc.enter_service(is_real_work)
            self.push_event(self.T + serve_time, {'type': 'handler', 'handler': handler, 'mmc': mmc, 'is_real_work': is_real_work, 'waitable': waitable})


    def process_completion_event(self, event_dict):
        mmc = event_dict['mmc']
        mmc.exit_service(event_dict['is_real_work'])
        if len(mmc.waiters_queue) > 0:
            (handler, serve_time, is_real_work, waitable) = mmc.exit_queue()
            mmc.enter_service(is_real_work)
            self.push_event(self.T + serve_time, {'type': 'handler', 'handler': handler, 'mmc': mmc, 'is_real_work': is_real_work, 'waitable': waitable}) 
        waitable_of_completed  = event_dict['waitable']
        waitable_of_completed.remove_waiter()
        if waitable_of_completed.all_waited():
            if event_dict['handler'].resume():
                self.inflight_conversation_count -= 1
                self.completed_conversations += 1

    def kick_off_new_conversation(self):
        conv = self.conversation_manager.create_conversation()
        self.inflight_conversation_count += 1
        HandleWrapper(self, conv).start()


    def simulate(self):
        if self.params['monitor_interval_virtual_time'] > 0:
            self.push_event(self.params['monitor_interval_virtual_time'], {'type':'stat'})
        while self.completed_conversations < self.params['iterations']:
            while self.inflight_conversation_count < self.num_inflight_agents:
                self.kick_off_new_conversation()
            t, counter, event_dict = heapq.heappop(self.events) 
            self.T = t
            if event_dict['type'] == 'stat':
                self.monitor_gpus()
            elif event_dict['type'] == 'handler':
                self.process_completion_event(event_dict)
            else:
                raise ValueError(f"UNKNOWN EVENT TYPE: {event_dict['type']!r}")

    def monitor_gpus(self):
        busy, really_busy, total_queued = self.gpus.get_busy_reallyBusy_totalQueued()
        self.total_busy_servers += busy
        self.total_really_busy_servers += really_busy
        self.total_queue_len += total_queued
        sleepers = self.agent_outside_service.num_servers - self.agent_outside_service.free_server_slots
        self.total_sleepers += sleepers
        self.sample_count += 1 
        
        curr_busy_ratio = busy / self.gpus.num_servers
        curr_really_busy_ratio = really_busy / self.gpus.num_servers
        avg_busy_ratio = self.total_busy_servers / self.sample_count / self.gpus.num_servers 
        avg_really_busy_ratio = self.total_really_busy_servers / self.sample_count / self.gpus.num_servers
        avg_queue_len = self.total_queue_len / self.sample_count 
        avg_sleepers = self.total_sleepers / self.sample_count
        print(f"hits={self.hits} misses={self.misses}")
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        print(f"T={self.T:.2f} - Curr: Busy={curr_busy_ratio:.4f}, ReallyBusy={curr_really_busy_ratio:.4f}, Queue={total_queued:.0f}, Sleepers={sleepers:.0f}")
        print(f"T={self.T:.2f} - Avg:  Busy={avg_busy_ratio:.4f}, ReallyBusy={avg_really_busy_ratio:.4f}, Queue={avg_queue_len:.2f}, Sleepers={avg_sleepers:.2f}, HitRate={hit_rate:.8f}%, Completed={self.completed_conversations}")
        print("-" * 80)
        if self.params['monitor_interval_virtual_time'] > 0:
            self.push_event(self.T + self.params['monitor_interval_virtual_time'], {'type':'stat'})