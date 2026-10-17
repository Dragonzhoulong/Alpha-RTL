# Standard-cell timing libraries

| File | Library | Source | License |
|---|---|---|---|
| `sky130_fd_sc_hd__tt_025C_1v80.lib` | SkyWater SKY130 HD, TT corner, 25 °C, 1.80 V | [google/skywater-pdk](https://github.com/google/skywater-pdk) | Apache-2.0 |
| `gf180mcu_fd_sc_mcu7t5v0__tt_025C_1v80.lib` | GlobalFoundries GF180MCU 7-track, TT corner | [google/gf180mcu-pdk](https://github.com/google/gf180mcu-pdk) | Apache-2.0 |
| `Nangate45_typ.lib` | Nangate 45 nm Open Cell Library, typical | as distributed with [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) | Nangate Open Cell Library license (see file header) |

Yosys maps designs to these cells (`dfflibmap`, `abc -liberty`) and OpenSTA reads
them for the critical-path delay.
