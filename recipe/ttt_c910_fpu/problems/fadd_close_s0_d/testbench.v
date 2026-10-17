/*
 * Verification testbench for ct_fadd_close_s0_d.
 *
 * Instantiates both the REFERENCE (golden model) and CANDIDATE (DUT), drives
 * identical inputs, and compares all outputs.
 *
 * ff1_pred / ff1_pred_onehot are compared only when close_eq == 0 (unequal
 * inputs). When both operands are equal, all F bits are zero and the reference
 * produces x for those outputs (don't-care by design).
 *
 * Usage (iverilog):
 *   iverilog -o simv -g2005-sv \
 *     ct_fadd_close_s0_d.v       \   <-- candidate
 *     reference_renamed.v        \   <-- golden (module name: ct_fadd_close_s0_d_ref)
 *     testbench.v
 *   vvp simv
 */

`timescale 1ns / 1ps

module testbench;

  // ---- Signals ----
  reg  [52:0] close_adder0;
  reg  [52:0] close_adder1;

  // Reference outputs
  wire        ref_close_eq;
  wire        ref_close_op_chg;
  wire [52:0] ref_close_sum_a_b;
  wire [52:0] ref_close_sum_b_a;
  wire [5 :0] ref_ff1_pred;
  wire [52:0] ref_ff1_pred_onehot;

  // DUT outputs
  wire        dut_close_eq;
  wire        dut_close_op_chg;
  wire [52:0] dut_close_sum_a_b;
  wire [52:0] dut_close_sum_b_a;
  wire [5 :0] dut_ff1_pred;
  wire [52:0] dut_ff1_pred_onehot;

  // Counters
  integer total_tests;
  integer pass_count;
  integer fail_count;
  integer i;
  integer seed;
  reg [63:0] rnd;

  // ---- Reference (golden model) ----
  ct_fadd_close_s0_d_ref u_ref (
    .close_adder0     (close_adder0),
    .close_adder1     (close_adder1),
    .close_eq         (ref_close_eq),
    .close_op_chg     (ref_close_op_chg),
    .close_sum_a_b    (ref_close_sum_a_b),
    .close_sum_b_a    (ref_close_sum_b_a),
    .ff1_pred         (ref_ff1_pred),
    .ff1_pred_onehot  (ref_ff1_pred_onehot)
  );

  // ---- DUT (candidate) ----
  ct_fadd_close_s0_d u_dut (
    .close_adder0     (close_adder0),
    .close_adder1     (close_adder1),
    .close_eq         (dut_close_eq),
    .close_op_chg     (dut_close_op_chg),
    .close_sum_a_b    (dut_close_sum_a_b),
    .close_sum_b_a    (dut_close_sum_b_a),
    .ff1_pred         (dut_ff1_pred),
    .ff1_pred_onehot  (dut_ff1_pred_onehot)
  );

  // ---- Comparison task ----
  task check_output;
    begin
      #1; // Allow combinational logic to settle
      total_tests = total_tests + 1;

      // Stage 1: always verify arithmetic outputs
      if (dut_close_sum_a_b !== ref_close_sum_a_b ||
          dut_close_sum_b_a !== ref_close_sum_b_a ||
          dut_close_op_chg  !== ref_close_op_chg  ||
          dut_close_eq      !== ref_close_eq) begin
        fail_count = fail_count + 1;
        if (fail_count <= 10) begin
          $display("FAIL test %0d [arithmetic]: a0=%h a1=%h", total_tests, close_adder0, close_adder1);
          $display("  REF: eq=%b opchg=%b sum_ab=%h sum_ba=%h",
                   ref_close_eq, ref_close_op_chg, ref_close_sum_a_b, ref_close_sum_b_a);
          $display("  DUT: eq=%b opchg=%b sum_ab=%h sum_ba=%h",
                   dut_close_eq, dut_close_op_chg, dut_close_sum_a_b, dut_close_sum_b_a);
        end
      end
      // Stage 2: when operands differ, verify ff1 prediction
      // (ff1_pred is don't-care when close_eq = 1, because all F bits are zero)
      else if (ref_close_eq == 1'b0 && ref_ff1_pred !== {6{1'bx}}) begin
        if (dut_ff1_pred       !== ref_ff1_pred       ||
            dut_ff1_pred_onehot !== ref_ff1_pred_onehot) begin
          fail_count = fail_count + 1;
          if (fail_count <= 10) begin
            $display("FAIL test %0d [ff1_pred]: a0=%h a1=%h", total_tests, close_adder0, close_adder1);
            $display("  REF: ff1_pred=%0d onehot=%h", ref_ff1_pred, ref_ff1_pred_onehot);
            $display("  DUT: ff1_pred=%0d onehot=%h", dut_ff1_pred, dut_ff1_pred_onehot);
          end
        end else begin
          pass_count = pass_count + 1;
        end
      end else begin
        pass_count = pass_count + 1;
      end
    end
  endtask

  // ---- Main test sequence ----
  initial begin
    total_tests = 0;
    pass_count  = 0;
    fail_count  = 0;
    seed        = 42;

    // ================================================================
    // Boundary values
    // ================================================================
    close_adder0 = 53'h0;                close_adder1 = 53'h0;                check_output; // A=B=0
    close_adder0 = 53'h1fffffffffffff;   close_adder1 = 53'h1fffffffffffff;   check_output; // A=B=all-ones
    close_adder0 = 53'h0;                close_adder1 = 53'h1fffffffffffff;   check_output; // A=0, B=max
    close_adder0 = 53'h1fffffffffffff;   close_adder1 = 53'h0;                check_output; // A=max, B=0
    close_adder0 = 53'h1;                close_adder1 = 53'h0;                check_output; // A=1, B=0
    close_adder0 = 53'h0;                close_adder1 = 53'h1;                check_output; // A=0, B=1
    close_adder0 = 53'h1fffffffffffff;   close_adder1 = 53'h1ffffffffffffe;   check_output; // A=max, B=max-1
    close_adder0 = 53'h10000000000000;   close_adder1 = 53'h0;                check_output; // MSB only
    close_adder0 = 53'h0;                close_adder1 = 53'h10000000000000;   check_output;
    // Alternating patterns
    close_adder0 = 53'h15555555555555;   close_adder1 = 53'h0aaaaaaaaaaaaa;   check_output;
    close_adder0 = 53'h0aaaaaaaaaaaaa;   close_adder1 = 53'h15555555555555;   check_output;
    // Equal alternating
    close_adder0 = 53'h15555555555555;   close_adder1 = 53'h15555555555555;   check_output;
    close_adder0 = 53'h0aaaaaaaaaaaaa;   close_adder1 = 53'h0aaaaaaaaaaaaa;   check_output;

    // ================================================================
    // Walking 1: close_adder0 = 1<<i, close_adder1 = 0
    // Exercises ff1_pred at each bit position (F[i] set, all higher F = 0)
    // ================================================================
    close_adder1 = 53'h0;
    for (i = 0; i <= 52; i = i + 1) begin
      close_adder0 = (53'd1 << i);
      check_output;
    end

    // ================================================================
    // Walking 1: close_adder0 = 0, close_adder1 = 1<<i
    // ================================================================
    close_adder0 = 53'h0;
    for (i = 0; i <= 52; i = i + 1) begin
      close_adder1 = (53'd1 << i);
      check_output;
    end

    // ================================================================
    // Equal inputs (close_eq=1): verify arithmetic only, ff1 is don't-care
    // ================================================================
    for (i = 0; i <= 52; i = i + 1) begin
      close_adder0 = (53'd1 << i);
      close_adder1 = (53'd1 << i);
      check_output;
    end

    // ================================================================
    // Near-equal pairs: differ by exactly one bit
    // ================================================================
    close_adder0 = 53'h0aaaaaaaaaaaa5; // arbitrary base
    for (i = 0; i <= 52; i = i + 1) begin
      close_adder1 = close_adder0 ^ (53'd1 << i);
      check_output;
    end

    close_adder0 = 53'h15555555555555;
    for (i = 0; i <= 52; i = i + 1) begin
      close_adder1 = close_adder0 ^ (53'd1 << i);
      check_output;
    end

    // ================================================================
    // Consecutive values sweep: A = X, B = X+1
    // ================================================================
    for (i = 0; i <= 52; i = i + 1) begin
      close_adder0 = (53'd1 << i) - 53'd1;  // 0b0...011...1
      close_adder1 = (53'd1 << i);
      check_output;
      close_adder0 = (53'd1 << i);
      close_adder1 = (53'd1 << i) - 53'd1;
      check_output;
    end

    // ================================================================
    // Power-of-2 pairs: A = 2^i, B = 2^j
    // ================================================================
    for (i = 0; i <= 52; i = i + 4) begin
      close_adder0 = (53'd1 << i);
      close_adder1 = (53'd1 << ((i + 2) % 53));
      check_output;
    end

    // ================================================================
    // Random tests (main coverage)
    // ================================================================
    for (i = 0; i < 2000; i = i + 1) begin
      rnd = {$random(seed), $random(seed)};
      close_adder0 = rnd[52:0];
      rnd = {$random(seed), $random(seed)};
      close_adder1 = rnd[52:0];
      check_output;
    end

    // ================================================================
    // Near-equal random: flip one bit of a random value
    // ================================================================
    for (i = 0; i < 500; i = i + 1) begin
      rnd = {$random(seed), $random(seed)};
      close_adder0 = rnd[52:0];
      close_adder1 = close_adder0 ^ (53'd1 << ($random(seed) % 53));
      check_output;
    end

    // ================================================================
    // Sparse patterns: 2-3 bits set at random positions
    // ================================================================
    for (i = 0; i < 200; i = i + 1) begin
      close_adder0 = (53'd1 << ($random(seed) % 53)) |
                     (53'd1 << ($random(seed) % 53));
      close_adder1 = (53'd1 << ($random(seed) % 53));
      check_output;
    end

    // ---- Report ----
    $display("========================================");
    $display("Test completed: %0d / %0d passed", pass_count, total_tests);
    if (fail_count == 0) begin
      $display("Your Design Passed");
    end else begin
      $display("Test completed with %0d / %0d failures", fail_count, total_tests);
    end
    $display("========================================");
    $finish;
  end

endmodule
