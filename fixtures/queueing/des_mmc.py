"""Trusted fixture candidate: M/M/c discrete-event simulation.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with a Monte Carlo estimate
of the requested queueing quantity over multiple replications plus a
normal-approximation CI.  Self-reported references are never trusted by the
host.
"""

import heapq
import json
import os

import numpy as np

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

lambda_ = float(problem["lambda"])
mu = float(problem["mu"])
c = int(problem["c"])
quantity = problem["quantity"]
threshold = problem.get("threshold", 0.0)


def simulate_once(rng, n_customers=20000):
    time = 0.0
    next_arrival = rng.exponential(1.0 / lambda_)
    departures = []
    queue = []  # FIFO of arrival times
    in_service = 0
    served = 0
    total_wait = 0.0
    wait_gt = 0
    queue_area = 0.0
    total_service = 0.0
    prev_time = 0.0
    while served < n_customers:
        next_departure = departures[0] if departures else float("inf")
        if next_arrival <= next_departure:
            queue_area += len(queue) * (next_arrival - prev_time)
            prev_time = next_arrival
            time = next_arrival
            if in_service < c:
                service = rng.exponential(1.0 / mu)
                total_service += service
                in_service += 1
                heapq.heappush(departures, time + service)
            else:
                queue.append(time)
            next_arrival = time + rng.exponential(1.0 / lambda_)
        else:
            queue_area += len(queue) * (next_departure - prev_time)
            prev_time = next_departure
            time = next_departure
            heapq.heappop(departures)
            in_service -= 1
            served += 1
            if queue:
                arrival_time = queue.pop(0)
                wait = time - arrival_time
                total_wait += wait
                if wait > threshold:
                    wait_gt += 1
                service = rng.exponential(1.0 / mu)
                total_service += service
                in_service += 1
                heapq.heappush(departures, time + service)
    mean_queue = queue_area / time if time > 0.0 else 0.0
    utilization = total_service / (c * time) if time > 0.0 else 0.0
    return {
        "mean_wait": total_wait / served if served else 0.0,
        "mean_queue": mean_queue,
        "mean_utilization": utilization,
        "prob_wait_gt": wait_gt / served if served else 0.0,
    }


rng = np.random.default_rng(42)
replications = 8
metrics = [simulate_once(rng)[quantity] for _ in range(replications)]
estimate = float(np.mean(metrics))
std = float(np.std(metrics, ddof=1))
half_width = 1.96 * std / np.sqrt(replications)
confidence_interval = [estimate - half_width, estimate + half_width]

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump(
        {"estimate": estimate, "confidence_interval": confidence_interval},
        fh,
    )
