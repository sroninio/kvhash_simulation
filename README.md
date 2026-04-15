# KV hash prefill simulation
יThe simulation simulates agentic behaviour in cluster.

Single Agents life time (as simulated) is the following:
CONTEXT = NULL
for i in num steps:
    Agent asks new request R = CONTEXT:Q_i
    the request is routed to some node
    R is sent to the storage to see what KV exists in the storage
    everything that is missing from R goes to GPU for calculation 
    GPU produces new block A_i and returns it to the Agent
    CONTEXT := CONTEXT:Q_i:A_i
    Agent "goes to sleep for a period of time"


The simulation maintains num_inflight_agents concurent agents through a long period of time (once an agent completes new agent arrives)
The sumulation measures the following:

RESULTS:
total_agents_comleted_per_second: self explainable
gpu_utilization: the pct of time in which GPU worked (not idle)
real_gpu_utilization: the pct of time in which GPU worked on mining new tokens

STORAGE:
The sumulation simulates the exact alogirhtm of DOCA KV. (range based hash and eviction)

ROUTING:
The simulation has 3 routing modes:
shared_storage_routing: simulates shared storage(all GPUS see shared storage), the requests are routed to the least busy GPU (smallest queue)
local_storage_sticky: simulates local storage only (G3). The requests are routed to the GPU that holds this request
local_storage_least_busy: simulates local storage only (G3). The requests are routed to the least busy GPU (smallest queue)
    Here we show the  effect of shared storage. (per K inflights shared storage will give better real_gpu_utilization then local storage)




```bash
python3 prefill_simulation.py -f config.json
```

---

## Configuration (`config.json`)


### `disk_size_in_blocks`

Each step new block of KVs is produced.  disk_size_in_blocks is disk size in blocks )

### `allow_holes_recalculation`

follows vllm semantics: in case it is FALSE only the prefix until the first hole will be considered as found and all the rest recalculated 

### `random_placement_on_miss`

DEBUG param, default 0, dont change

### `evict_on_miss`

DEBUG param, default 1, dont change

### `num_inflight_agents`

concurrency: The simulation maintains num_inflight_agents concurent agents through a long period of time (once an agent completes new agent arrives)

### `steps`
How many steps are there in agents life time

### `ranges`

How many ranges are in the DOCA KV algorithm 

### `iterations`

how many conversations to complete in the simulation

### `time_between_steps`

for how long "agent goes to sleep" between 2 consecutive requests to the cluster 

### `total_gpus`

how many GPUs are in the cluster

### `step_time_in_gpu`

how much time it takes to calculate single block in GPU 
note that i put it as constant number currently, bcs Deepseek claim that it is ~constant, though in classic LLM it is some close to linear function of the context. it can obviously be easily chenged in the simulator. 

### `context_window_size`
what is the maximal context size for the request, for example if num_steps is 200 and context_window_size is 350 so after 200 steps the context will work in sliding window and consider only last 200 blocks.
if set to 0 then it will be assigned to steps.

### `force_hit_ratio`

For the cases in which we test peformance  of different routing algorithms it makes sense to fix the hit_ratio, so all the DOCA KV algorithm will be skipped. (also note that it will run much faster)

### `scheduling_strategy`
The simulation has 3 routing modes:
shared_storage_routing: simulates shared storage(all GPUS see shared storage), the requests are routed to the least busy GPU (smallest queue)
local_storage_sticky: simulates local storage only (G3). The requests are routed to the GPU that holds this request
local_storage_least_busy: simulates local storage only (G3). The requests are routed to the least busy GPU (smallest queue)
    Here we show the  effect of shared storage. (per K inflights shared storage will give better real_gpu_utilization then local storage)


### `is_use_theoretical_agents`
in case it is 1, num_inflight_agents is overriden by the theoretical minimal required number of agents that achieve maximal possible rate.


### `storage_blocks_per_second`

the BW of storage (inifinite in case set to 0)

### `output_file`

where ti write the results

### `monitor_interval_virtual_time`

How often (in virtual time) the simulator samples GPU busy/queue stats. If you set this to **`0` in `config.json`**, `prefill_simulation.py` replaces it with **`(step_time_in_gpu + time_between_steps) * 100`** before the run (same units as the other time parameters).

---

## Outputs

