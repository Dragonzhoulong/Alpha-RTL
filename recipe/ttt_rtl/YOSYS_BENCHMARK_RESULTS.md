# Yosys Synthesis Time Benchmark Results

## Summary

Benchmarked Yosys synthesis on 11 RTLLM reference designs (1 failed due to missing module).

**Key Findings:**
- **Min time:** 0.12s (edge_detect, 5 cells)
- **Max time:** 0.59s (alu, 2651 cells)
- **Mean time:** 0.22s
- **Median time:** 0.16s

## Recommended `timeout_s` Settings

Based on the benchmark results:

| Setting | Value | Use Case |
|---------|-------|----------|
| **Conservative** | **5.0s** | Production (99th percentile × 10) |
| **Balanced** | **3.0s** | Training (95th percentile × 5) |
| **Aggressive** | **1.5s** | Quick experiments (mean × 7) |

### Rationale

1. **Reference designs are optimized** - LLM-generated code will likely be less efficient and take longer to synthesize
2. **Safety margin** - Account for:
   - Suboptimal code structure
   - Redundant logic
   - Larger designs (LLM may generate verbose code)
   - System load variations

3. **Actual synthesis times:**
   - Simple designs (5-50 cells): 0.12-0.16s
   - Medium designs (50-300 cells): 0.16-0.32s
   - Complex designs (300-3000 cells): 0.32-0.59s

## Detailed Results

| Problem | Time (s) | Area (cells) | Wires | Category |
|---------|----------|--------------|-------|----------|
| edge_detect | 0.12 | 5 | 8 | Miscellaneous |
| adder_8bit | 0.12 | 40 | 70 | Arithmetic |
| signal_generator | 0.13 | 45 | 38 | Miscellaneous |
| counter_12 | 0.14 | 14 | 14 | Control |
| fsm | 0.14 | 40 | 37 | Control |
| asyn_fifo | 0.16 | 296 | 177 | Memory |
| freq_div | 0.16 | 54 | 38 | Miscellaneous |
| adder_16bit | 0.20 | 240 | 378 | Arithmetic |
| adder_32bit | 0.32 | 596 | 860 | Arithmetic |
| multi_16bit | 0.32 | 543 | 451 | Arithmetic |
| **alu** | **0.59** | **2651** | **2629** | **RISC-V (largest)** |

## Recommended Configuration

### For PPA Optimization (Your Goal)

```yaml
# recipe/ttt_rtl/config/ttt_rtl_trainer.yaml
custom_reward_function:
  reward_kwargs:
    timeout_s: 5.0           # ✓ Conservative for LLM-generated code
    w_syntax: 0.1
    w_func: 0.8
    w_ppa: 0.5               # Enable PPA optimization
    failure_reward: 0.0
```

### Timeout Scaling by Problem Complexity

If you want to optimize timeout per problem:

```python
# In rtl_reward.py, adjust timeout based on problem
timeout_map = {
    "edge_detect": 1.0,
    "counter_12": 1.0,
    "adder_8bit": 1.5,
    "fsm": 1.5,
    "adder_16bit": 2.0,
    "asyn_fifo": 2.5,
    "adder_32bit": 3.0,
    "multi_16bit": 3.0,
    "alu": 5.0,
}
timeout_s = timeout_map.get(problem_id, 3.0)  # Default 3.0s
```

## Area Distribution

- **Tiny** (< 50 cells): 4 problems (edge_detect, counter_12, adder_8bit, signal_generator)
- **Small** (50-300 cells): 3 problems (fsm, freq_div, adder_16bit, asyn_fifo)
- **Medium** (300-1000 cells): 2 problems (multi_16bit, adder_32bit)
- **Large** (> 1000 cells): 1 problem (alu)

## PPA Reward Curve Analysis

With current formula: `r_ppa = 1.0 / (1.0 + area / 100.0)`

| Design | Area | PPA Reward | Quality |
|--------|------|------------|---------|
| edge_detect | 5 | 0.95 | Excellent |
| counter_12 | 14 | 0.88 | Excellent |
| adder_8bit | 40 | 0.71 | Good |
| fsm | 40 | 0.71 | Good |
| adder_16bit | 240 | 0.29 | Average |
| asyn_fifo | 296 | 0.25 | Average |
| adder_32bit | 596 | 0.14 | Poor |
| alu | 2651 | 0.04 | Very Poor |

**Observation:** The reward curve heavily penalizes large designs. This is good for encouraging compact implementations.

## Conclusion

**Recommended setting for your PPA optimization goal:**

```yaml
custom_reward_function:
  reward_kwargs:
    timeout_s: 5.0  # Safe for all problems, including complex ones
```

This gives:
- ✓ Enough time for LLM-generated (potentially suboptimal) code
- ✓ Safety margin for system load
- ✓ Covers even the largest problem (alu: 0.59s × 8.5 = 5.0s)
- ✓ Won't timeout on verbose/redundant code

**For faster iteration during development:**
- Use `timeout_s: 2.0` for simple problems (adder, counter, fsm)
- Use `timeout_s: 5.0` for complex problems (alu, multi, asyn_fifo)
