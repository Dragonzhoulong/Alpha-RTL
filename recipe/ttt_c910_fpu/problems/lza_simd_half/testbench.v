/*
 * Verification testbench for ct_vfmau_lza_simd_half.
 *
 * Instantiates both the REFERENCE (golden) and the CANDIDATE (DUT) module,
 * drives identical random + edge-case inputs, and compares lza_result and
 * lza_result_zero outputs combinationally.
 *
 * Usage (iverilog):
 *   iverilog -o simv -g2005-sv \
 *     ct_vfmau_lza_simd_half.v  \   <-- candidate
 *     reference_renamed.v        \   <-- reference (module renamed to ct_vfmau_lza_simd_half_ref)
 *     testbench.v
 *   vvp simv
 */

`timescale 1ns / 1ps

module testbench;

  // ---- Parameters ----
  localparam NUM_RANDOM_TESTS = 800;

  // ---- Signals ----
  reg  [23:0] summand;
  reg  [23:0] addend;
  reg         sub_vld;

  // Reference outputs
  wire [4:0] ref_lza_result;
  wire       ref_lza_result_zero;

  // DUT outputs
  wire [4:0] dut_lza_result;
  wire       dut_lza_result_zero;

  // Counters
  integer total_tests;
  integer pass_count;
  integer fail_count;
  integer i;
  integer seed;

  // ---- Reference (golden model) ----
  ct_vfmau_lza_simd_half_ref u_ref (
    .addend          (addend),
    .lza_result      (ref_lza_result),
    .lza_result_zero (ref_lza_result_zero),
    .sub_vld         (sub_vld),
    .summand         (summand)
  );

  // ---- DUT (candidate) ----
  ct_vfmau_lza_simd_half u_dut (
    .addend          (addend),
    .lza_result      (dut_lza_result),
    .lza_result_zero (dut_lza_result_zero),
    .sub_vld         (sub_vld),
    .summand         (summand)
  );

  // ---- Helper task ----
  task check_output;
    begin
      #1; // Allow combinational logic to settle
      total_tests = total_tests + 1;

      if (dut_lza_result !== ref_lza_result ||
          dut_lza_result_zero !== ref_lza_result_zero) begin
        fail_count = fail_count + 1;
        if (fail_count <= 10) begin
          $display("FAIL test %0d: mismatch", total_tests);
          $display("  summand=%h addend=%h sub_vld=%b", summand, addend, sub_vld);
          $display("  REF lza_result=%0d zero=%b  DUT lza_result=%0d zero=%b",
                   ref_lza_result, ref_lza_result_zero,
                   dut_lza_result, dut_lza_result_zero);
        end
      end else begin
        pass_count = pass_count + 1;
      end
    end
  endtask

  // ---- Main test sequence ----
  initial begin
    // Initialize
    total_tests = 0;
    pass_count  = 0;
    fail_count  = 0;
    seed        = 42;
    summand     = 24'b0;
    addend      = 24'b0;
    sub_vld     = 1'b0;

    // ================================================================
    // Edge cases: sub_vld = 0 (addition mode)
    // ================================================================
    sub_vld = 1'b0;

    // Both zero
    summand = 24'h0; addend = 24'h0;
    check_output;

    // Both all-ones
    summand = 24'hFFFFFF; addend = 24'hFFFFFF;
    check_output;

    // MSB only
    summand = 24'h0; addend = 24'h0;
    summand[23] = 1'b1;
    check_output;

    summand = 24'h0; addend = 24'h0;
    addend[23] = 1'b1;
    check_output;

    // LSB only
    summand = 24'h0; addend = 24'h0;
    summand[0] = 1'b1;
    check_output;

    summand = 24'h0; addend = 24'h0;
    addend[0] = 1'b1;
    check_output;

    // One operand zero, other all-ones
    summand = 24'hFFFFFF; addend = 24'h0;
    check_output;

    summand = 24'h0; addend = 24'hFFFFFF;
    check_output;

    // Alternating patterns
    summand = 24'hAAAAAA; addend = 24'h555555;
    check_output;

    summand = 24'h555555; addend = 24'hAAAAAA;
    check_output;

    // Single bit walking through all positions
    for (i = 23; i >= 0; i = i - 1) begin
      summand = 24'h0; addend = 24'h0;
      summand[i] = 1'b1;
      check_output;

      summand = 24'h0; addend = 24'h0;
      addend[i] = 1'b1;
      check_output;
    end

    // Adjacent bits (carry propagate patterns)
    for (i = 23; i >= 1; i = i - 2) begin
      summand = 24'h0; addend = 24'h0;
      summand[i] = 1'b1; addend[i-1] = 1'b1;
      check_output;
    end

    // Carry propagate chain: P = summand ^ addend = all 1s
    summand = 24'hAAAAAA; addend = 24'h555555;
    check_output;

    // Carry generate chain: G = summand & addend = all 1s
    summand = 24'hFFFFFF; addend = 24'hFFFFFF;
    check_output;

    // Carry delete chain: D = ~(summand | addend) = all 1s (both zero)
    summand = 24'h0; addend = 24'h0;
    check_output;

    // ================================================================
    // Edge cases: sub_vld = 1 (subtraction mode)
    // ================================================================
    sub_vld = 1'b1;

    // Both zero
    summand = 24'h0; addend = 24'h0;
    check_output;

    // Both all-ones
    summand = 24'hFFFFFF; addend = 24'hFFFFFF;
    check_output;

    // One all-ones, other zero
    summand = 24'hFFFFFF; addend = 24'h0;
    check_output;

    summand = 24'h0; addend = 24'hFFFFFF;
    check_output;

    // MSB patterns
    summand = 24'h0; addend = 24'h0;
    summand[23] = 1'b1;
    check_output;

    summand = 24'h0; addend = 24'h0;
    addend[23] = 1'b1;
    check_output;

    // Walking bit in subtraction mode
    for (i = 23; i >= 0; i = i - 1) begin
      summand = 24'h0; addend = 24'h0;
      summand[i] = 1'b1;
      check_output;
    end

    // Alternating in sub mode
    summand = 24'hAAAAAA; addend = 24'h555555;
    check_output;

    summand = 24'h555555; addend = 24'hAAAAAA;
    check_output;

    // ================================================================
    // Random tests: sub_vld = 0
    // ================================================================
    sub_vld = 1'b0;
    for (i = 0; i < NUM_RANDOM_TESTS / 2; i = i + 1) begin
      summand = $random(seed) & 24'hFFFFFF;
      addend  = $random(seed) & 24'hFFFFFF;
      check_output;
    end

    // ================================================================
    // Random tests: sub_vld = 1
    // ================================================================
    sub_vld = 1'b1;
    for (i = 0; i < NUM_RANDOM_TESTS / 2; i = i + 1) begin
      summand = $random(seed) & 24'hFFFFFF;
      addend  = $random(seed) & 24'hFFFFFF;
      check_output;
    end

    // ================================================================
    // Random tests: mixed sub_vld with sparse patterns
    // ================================================================
    for (i = 0; i < 100; i = i + 1) begin
      sub_vld = $random(seed) & 1;
      summand = 24'h0;
      addend  = 24'h0;
      summand[{$random(seed)} % 24] = 1'b1;
      summand[{$random(seed)} % 24] = 1'b1;
      addend[{$random(seed)} % 24]  = 1'b1;
      check_output;
    end

    // ================================================================
    // Random tests: nearly-equal operands (close-path scenario)
    // ================================================================
    sub_vld = 1'b1;
    for (i = 0; i < 100; i = i + 1) begin
      summand = $random(seed) & 24'hFFFFFF;
      addend  = summand;
      addend[{$random(seed)} % 24] = ~addend[{$random(seed)} % 24];
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
