# NeuroFlow-FL: Deep Reinforcement Learning for Adaptive Traffic Signal Control

## What It Is

NeuroFlow-FL is a simulation-based intelligent traffic management system that uses **Deep Reinforcement Learning (RL)** to replace traditional fixed-time traffic signals with adaptive, learning-based controllers. The project is implemented entirely in open-source software (SUMO, Python, Stable-Baselines3) and explores how AI agents can learn to reduce congestion, waiting times, and emissions at urban intersections.

---

## The Problem Addressed

Traditional traffic signals operate on **fixed-time cycles** programmed months or years in advance. They cannot adapt to:

- Time-of-day demand variations
- Weather events or accidents
- Real-time queue buildups
- Emergency vehicle priority needs

This inflexibility causes excessive waiting, fuel waste, emissions, and delayed emergency responses.

---

## Core Approach

| Component | Description |
|-----------|-------------|
| **Simulation Engine** | SUMO (Simulation of Urban MObility) — microscopic traffic simulator |
| **RL Algorithms** | DQN (Deep Q-Network) and PPO (Proximal Policy Optimisation) |
| **State Space** | 5 features: queue lengths from 4 approach directions + current signal phase |
| **Action Space** | Binary decision: keep current phase or switch to next phase |
| **Reward Function** | Weighted penalty on vehicle waiting time, queue length, and emergency vehicle delay |
| **Training Framework** | Stable-Baselines3 with custom Gymnasium environment wrapper |

---

## Two-Phase Experimental Design

### Phase 1 — Single Intersection
- One 4-way signalised intersection with 2 lanes per approach
- Trained both DQN and PPO for 200,000 timesteps
- **Key result**: Both algorithms achieved identical performance — a **73.7% reduction in average vehicle waiting time** (from 760.86s to 199.73s) and **9.5% improvement in reward** over the fixed-time baseline
- Finding: with a simple binary action space, both algorithms hit the same performance ceiling

### Phase 2 — Multi-Intersection Network
- Extended to a 4-intersection linear arterial network
- Independent agents trained at each junction
- **Key result**: PPO outperformed DQN by **15.7% in overall network reward**, winning at 3 of 4 intersections
- DQN only won at the terminal junction (J4), the simplest sub-problem
- Finding: algorithm choice becomes consequential as complexity increases — PPO handles non-stationarity better in multi-agent settings

---

## Key Innovations

**1. Emergency Vehicle Preemption Module**
- Detects emergency vehicles within 100m of any approach
- Automatically overrides RL policy to grant green phase
- Seamlessly hands control back to the RL agent after the vehicle clears
- Demonstrates that safety-critical requirements can coexist with learned policies

**2. Federated Learning Architecture**
- Designed using the **Flower (flwr)** framework for privacy-preserving multi-intersection training
- Intersections share **model weights only** (not raw traffic data) via FedAvg aggregation
- Addresses data sovereignty and GDPR concerns for smart city deployment
- Enables collaborative improvement without centralising sensitive vehicle movement data

---

## Technology Stack

| Layer | Tools |
|-------|-------|
| Simulation | SUMO 1.19+, TraCI API |
| RL Framework | Stable-Baselines3, Gymnasium |
| Deep Learning | PyTorch 2.x |
| Federated Learning | Flower (flwr) |
| Containerisation | Docker (in progress) |
| Visualisation | TensorBoard, Matplotlib, Seaborn |
| Data Processing | NumPy, Pandas |

All open-source, reproducible on standard consumer hardware (CPU-only training).

---

## Key Results Summary

| Metric | Fixed-Time | RL (Phase 1) | Improvement |
|--------|-----------|--------------|-------------|
| Avg. waiting time | 760.86s | 199.73s | **-73.7%** |
| Episode reward | -71,198 | -64,441 | **+9.5%** |
| Queue length | 46.44 | 32.24 | **-30.6%** |

**Phase 2 network-wide**: PPO averaged **-700.77** reward vs DQN's **-832.30** — a meaningful 15.7% advantage under complex, non-stationary conditions.

---

## Real-World Implications

- **Urban traffic management**: potential for tens of millions of vehicle-minutes saved daily across city networks
- **Emergency response**: compatible with life-critical preemption requirements
- **Smart city privacy**: federated design enables GDPR-compliant deployment
- **Accessibility**: entirely open-source, low hardware requirements, no proprietary licenses needed

---

## Current Limitations

- Results are simulation-only (no real-world deployment yet)
- Single training seed per algorithm (no statistical confidence intervals)
- Deterministic traffic demand (zero variance across evaluation episodes)
- Binary action space limits policy richness
- Phase 2 used sequential training rather than true concurrent multi-agent learning
- Real-time sensing (e.g. CCTV-based queue estimation) is a planned deployment consideration, not yet implemented — current state comes from SUMO/TraCI ground truth

---

## Roadmap

Active development priorities, roughly in order:

1. **Containerise the project with Docker** — consistent dependency versions (SUMO, PyTorch, SB3, Flower) across Linux and Windows devices for the team, enabling parallel multi-seed training runs
2. **Remap simulation network to a real intersection** — import real road geometry (via OpenStreetMap / `netconvert`) from an intersection in our city, replacing the generic synthetic 4-way network for more concrete, presentable results
3. **RL agent refactor** — restructure the training pipeline (currently a single monolithic script) into clean, separated modules: entry point, training loop, agent interfaces, and results/logging
4. **Multi-seed validation** — retrain with ~5 seeds per algorithm per scenario to report results with confidence intervals rather than single-run numbers
5. **Variable traffic demand** — randomised vehicle arrival rates across episodes, replacing fixed deterministic demand
6. **Grid topology scenario** — extend beyond the linear arterial network to a 2D grid, testing coordination under real spillback patterns (likely requires moving from sequential to concurrent multi-agent training)
7. **System design document** — formalise the architecture covering environment, agents, preemption, and federated learning end-to-end
8. **Emergency preemption refinement** — explore smoother, ETA-based preemption in place of the current hard 100m-radius override
9. **CCTV sensing demo (exploratory)** — small proof-of-concept using sample/public traffic footage to estimate queue length via object detection, feeding the same state format the trained models expect; a supplementary validation, not a full real-world deployment

---

## Bottom Line

NeuroFlow-FL demonstrates that **deep RL can learn highly effective traffic signal policies from simulation alone**, with PPO proving superior for complex multi-intersection networks. The integration of emergency preemption and federated learning positions the work as a practical foundation for next-generation, privacy-preserving smart traffic systems.