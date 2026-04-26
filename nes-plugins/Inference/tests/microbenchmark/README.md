# Prediction Cache Data Generators

This repository contains several synthetic workload generators designed to evaluate **cache replacement strategies** under controlled access patterns.

The generators are designed to isolate different properties that influence cache performance:

-   popularity skew

-   temporal locality

-   burstiness

-   workload drift


Each generator produces a sequence of records (byte buffers) whose **key identity determines cache behavior**.

The goal is to study how different cache policies behave under different workload structures.

---

# Overview of Generators

| Generator | Workload Property | Description |
| --- | --- | --- |
| Zipf | Popularity skew | Models workloads where some keys are much more popular than others |
| Temporal Locality | Recency locality | Models sliding-window workloads with repeated accesses |
| Burstiness | Bursty access phases | Models workloads with temporary hot subsets of keys |
| Deterministic | Baseline | Used only for throughput benchmarking |

The deterministic generator is used only to measure raw cache throughput and does not model realistic access patterns.

---

# Drift

All generators support optional **workload drift**. 
It simulates environments where **access patterns change over time**.

Two parameters control drift behavior.

---

## `driftInterval`

Number of emitted records between drift events.

Example:

```
driftInterval = 100000
```

This means the active workload structure is modified every 100k records.

Smaller values cause more frequent drift.

---

## `driftFraction`

Fraction of the active set replaced during a drift event.

Example values:

```
0.0  → no drift  
0.1  → slow drift  
0.5  → moderate drift  
1.0  → full replacement
```

Drift modifies only the **active subset of keys** used by the generator.

---

# Generator Descriptions

---

# Zipf Generator

## Purpose

Models workloads with **skewed key popularity**, i.e., a small subset of keys receives most accesses.

---

## Parameters

| Parameter | Description |
| --- | --- |
| `numKeys` | size of the key universe |
| `totalRecords` | total number of generated accesses |
| `zipfExponent` | skew strength |
| `driftInterval` | number of accesses between drift events |
| `driftFraction` | fraction of hot keys replaced during drift |

---

## Zipf Exponent

The exponent controls skew strength.

Typical values:

```
0.0  → uniform distribution  
0.6  → mild skew  
1.0  → strong skew  
1.2  → very strong skew
```

---

## Effect of Drift

Without drift, **same hot keys dominate** for the entire trace.
With drift, **the set of hot keys changes** over time.

Drift primarily impacts **frequency-based policies (LFU)**.

---

# Temporal Locality Generator

## Purpose

Models workloads with **recency-based locality**.

Common examples:

-   sliding window analytics

-   video frame processing

-   audio processing pipelines

-   sensor streams


---

## Parameters

| Parameter | Description |
| --- | --- |
| `universeSize` | total key universe |
| `seriesLength` | length of the active series |
| `windowSize` | sliding window size |
| `overlapRatio` | overlap between consecutive windows |
| `totalRecords` | total number of accesses |
| `driftInterval` | records between drift events |
| `driftFraction` | fraction of series shifted during drift |

---

## Overlap Ratio

Defines how much consecutive windows overlap.

```
0.0  → no overlap  
0.5  → moderate overlap  
0.8  → strong locality  
0.95 → extremely strong locality
```

---

## Effect of Drift

Without drift, **the same series region repeats** indefinitely.
With drift, **the sliding window gradually moves** through the key universe.

This simulates:

-   moving time windows

-   evolving sensor readings

-   shifting data streams


Drift causes previously hot keys to disappear from the workload.

---

# Burstiness Generator

## Purpose

Models workloads with **bursty access patterns**.

In bursty workloads:

-   requests arrive in phases

-   a temporary subset of keys becomes hot

-   bursts are followed by cooldown periods


Examples include:

-   event-driven workloads

-   user interaction spikes

-   streaming bursts

-   sensor activity triggers


---

## Parameters

| Parameter | Description |
| --- | --- |
| `numKeys` | size of the key universe |
| `dutyCycle` | burst intensity |
| `onPeriod` | length of each burst |
| `totalRecords` | total accesses |
| `driftInterval` | records between drift events |
| `driftFraction` | fraction of burst hotset replaced |

---

## Duty Cycle

Controls burst strength.

```
1.0  → uniform workload  
0.2  → moderate bursts  
0.05 → strong bursts  
0.01 → extreme bursts
```

Lower values create **tighter burst hotsets**.

---

## On Period

Number of records emitted during a burst.

Example:

```
onPeriod = 1000
```

During this phase, accesses concentrate on the active burst hotset.

---

## Effect of Drift

Without drift, burst **hotsets repeat cyclically**.
With drift, the burst **hotset gradually changes** over time.

This models environments such as:

-   shifting user sessions

-   rotating working sets

-   changing workload phases


---

# Recommended Experimental Parameters

The following configuration provides stable benchmark results.

## Zipf

```
numKeys = 1000  
totalRecords = 1,000,000  
zipfExponent = 1.0  
driftInterval = 100000  
driftFraction ∈ {0.0, 0.1, 0.5, 1.0}
```

---

## Temporal Locality

```
universeSize = 5000  
seriesLength = 1000  
windowSize = 100  
overlapRatio = 0.8  
totalRecords = 1,000,000  
driftInterval = 100000  
driftFraction ∈ {0.0, 0.1, 0.5, 1.0}
```

---

## Burstiness

```
numKeys = 1000  
dutyCycle = 0.2  
onPeriod = 1000  
totalRecords = 1,000,000  
driftInterval = 100000  
driftFraction ∈ {0.0, 0.1, 0.5, 1.0}
```

---

# Expected Policy Behavior

| Pattern | Typical Best Policies |
| --- | --- |
| Zipf (no drift) | LFU |
| Zipf (with drift) | LRU / Second Chance |
| Temporal locality | LRU / Second Chance |
| Burstiness | LRU / Second Chance |
| High drift | all policies degrade |

Drift primarily penalizes policies relying on **long-term frequency statistics**.

---

# Summary

The generators provide a controlled way to study how cache replacement policies behave under different workload structures:

-   **Zipf** → popularity skew

-   **Temporal locality** → recency reuse

-   **Burstiness** → short-lived hotsets

-   **Drift** → evolving workloads
