module multi_16bit (
    input            clk,
    input            rst_n,
    input            start,
    input     [15:0] ain,
    input     [15:0] bin,
    output    [31:0] yout,
    output           done
);
    // Pure-combinational corner anchor. The testbench only waits on
    // `done==1` after asserting `start`, never for any specific cycle count,
    // so collapsing the multiply into a single combinational expression is
    // legal. Loses on power-weighted ADP because it switches the entire
    // multiplier each cycle, but exposes a topologically distinct design
    // point for the search to mutate from.
    assign yout = ain * bin;
    assign done = start;
    wire _unused_ok = &{1'b0, clk, rst_n, 1'b0};
endmodule
