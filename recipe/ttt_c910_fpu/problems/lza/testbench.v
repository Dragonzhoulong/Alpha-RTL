/*
 * Verification testbench for ct_vfmau_lza.
 *
 * Instantiates both the REFERENCE (golden) and the CANDIDATE (DUT) module,
 * drives identical random + edge-case inputs, and compares lza_result
 * outputs combinationally.
 *
 * Usage (iverilog):
 *   iverilog -o simv -g2005-sv \
 *     ct_vfmau_lza.v           \   <-- candidate
 *     reference_renamed.v       \   <-- reference (module renamed to ct_vfmau_lza_ref)
 *     deps/ct_vfmau_lza_42.v deps/ct_vfmau_lza_32.v \
 *     testbench.v
 *   vvp simv
 */

`timescale 1ns / 1ps

module testbench;

  // ---- Parameters ----
  localparam NUM_RANDOM_TESTS = 800;

  // ---- Signals ----
  reg  [107:0] summand;
  reg  [107:0] addend;
  reg          sub_vld;

  // Reference outputs
  wire [6:0] ref_lza_result;

  // DUT outputs
  wire [6:0] dut_lza_result;

  // Counters
  integer total_tests;
  integer pass_count;
  integer fail_count;
  integer i;
  integer seed;

  // ---- Reference (golden model) ----
  ct_vfmau_lza_ref u_ref (
    .addend     (addend),
    .lza_result (ref_lza_result),
    .sub_vld    (sub_vld),
    .summand    (summand)
  );

  // ---- DUT (candidate) ----
  ct_vfmau_lza u_dut (
    .addend     (addend),
    .lza_result (dut_lza_result),
    .sub_vld    (sub_vld),
    .summand    (summand)
  );

  // ---- Helper task ----
  task check_output;
    begin
      #1; // Allow combinational logic to settle
      total_tests = total_tests + 1;

      if (dut_lza_result !== ref_lza_result) begin
        fail_count = fail_count + 1;
        if (fail_count <= 10) begin
          $display("FAIL test %0d: lza_result mismatch", total_tests);
          $display("  summand=%h addend=%h sub_vld=%b", summand, addend, sub_vld);
          $display("  REF lza_result=%0d  DUT lza_result=%0d", ref_lza_result, dut_lza_result);
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
    summand     = 108'b0;
    addend      = 108'b0;
    sub_vld     = 1'b0;

    // ================================================================
    // Edge cases: sub_vld = 0 (addition mode)
    // ================================================================
    sub_vld = 1'b0;

    // Both zero
    summand = 108'h0; addend = 108'h0;
    check_output;

    // Both all-ones
    summand = {108{1'b1}}; addend = {108{1'b1}};
    check_output;

    // MSB only
    summand = 108'h0; addend = 108'h0;
    summand[107] = 1'b1; addend[107] = 1'b0;
    check_output;

    summand = 108'h0; addend = 108'h0;
    summand[107] = 1'b0; addend[107] = 1'b1;
    check_output;

    // LSB only
    summand = 108'h0; addend = 108'h0;
    summand[0] = 1'b1; addend[0] = 1'b0;
    check_output;

    // One operand zero, other all-ones
    summand = {108{1'b1}}; addend = 108'h0;
    check_output;

    summand = 108'h0; addend = {108{1'b1}};
    check_output;

    // Alternating patterns
    summand = 108'hAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAA;
    addend  = 108'h555_5555_5555_5555_5555_5555_555;
    check_output;

    summand = 108'h555_5555_5555_5555_5555_5555_555;
    addend  = 108'hAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAA;
    check_output;

    // Single bit walking through high bits [107:44]
    for (i = 107; i >= 44; i = i - 4) begin
      summand = 108'h0;
      addend  = 108'h0;
      summand[i] = 1'b1;
      addend[i]  = 1'b0;
      check_output;

      summand = 108'h0;
      addend  = 108'h0;
      summand[i] = 1'b0;
      addend[i]  = 1'b1;
      check_output;
    end

    // Single bit walking through low bits [43:0]
    for (i = 43; i >= 0; i = i - 4) begin
      summand = 108'h0;
      addend  = 108'h0;
      summand[i] = 1'b1;
      check_output;

      summand = 108'h0;
      addend  = 108'h0;
      addend[i] = 1'b1;
      check_output;
    end

    // Boundary between high/low (bits around 44)
    summand = 108'h0; addend = 108'h0;
    summand[44] = 1'b1; addend[43] = 1'b1;
    check_output;

    summand = 108'h0; addend = 108'h0;
    summand[45] = 1'b1; summand[44] = 1'b1; addend[43] = 1'b1;
    check_output;

    // Carry propagate chain: P = summand ^ addend = all 1s
    summand = 108'hAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAA;
    addend  = 108'h555_5555_5555_5555_5555_5555_555;
    check_output;

    // Carry generate chain: G = summand & addend = all 1s
    summand = {108{1'b1}}; addend = {108{1'b1}};
    check_output;

    // Carry delete chain: D = ~(summand | addend) = all 1s (both zero)
    summand = 108'h0; addend = 108'h0;
    check_output;

    // ================================================================
    // Edge cases: sub_vld = 1 (subtraction mode)
    // ================================================================
    sub_vld = 1'b1;

    // Both zero
    summand = 108'h0; addend = 108'h0;
    check_output;

    // Both all-ones
    summand = {108{1'b1}}; addend = {108{1'b1}};
    check_output;

    // One all-ones, other zero
    summand = {108{1'b1}}; addend = 108'h0;
    check_output;

    summand = 108'h0; addend = {108{1'b1}};
    check_output;

    // MSB patterns
    summand = 108'h0; addend = 108'h0;
    summand[107] = 1'b1;
    check_output;

    summand = 108'h0; addend = 108'h0;
    addend[107] = 1'b1;
    check_output;

    // Walking bit in subtraction mode
    for (i = 107; i >= 0; i = i - 8) begin
      summand = 108'h0;
      addend  = 108'h0;
      summand[i] = 1'b1;
      check_output;
    end

    // Alternating in sub mode
    summand = 108'hAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAA;
    addend  = 108'h555_5555_5555_5555_5555_5555_555;
    check_output;

    // ================================================================
    // Random tests: sub_vld = 0
    // ================================================================
    sub_vld = 1'b0;
    for (i = 0; i < NUM_RANDOM_TESTS / 2; i = i + 1) begin
      summand = {$random(seed), $random(seed), $random(seed), $random(seed)} & {108{1'b1}};
      addend  = {$random(seed), $random(seed), $random(seed), $random(seed)} & {108{1'b1}};
      check_output;
    end

    // ================================================================
    // Random tests: sub_vld = 1
    // ================================================================
    sub_vld = 1'b1;
    for (i = 0; i < NUM_RANDOM_TESTS / 2; i = i + 1) begin
      summand = {$random(seed), $random(seed), $random(seed), $random(seed)} & {108{1'b1}};
      addend  = {$random(seed), $random(seed), $random(seed), $random(seed)} & {108{1'b1}};
      check_output;
    end

    // ================================================================
    // Random tests: mixed sub_vld with sparse patterns
    // ================================================================
    for (i = 0; i < 100; i = i + 1) begin
      sub_vld = $random(seed) & 1;
      // Sparse patterns: few bits set
      summand = 108'h0;
      addend  = 108'h0;
      summand[$random(seed) % 108] = 1'b1;
      summand[$random(seed) % 108] = 1'b1;
      addend[$random(seed) % 108]  = 1'b1;
      check_output;
    end

    // ================================================================
    // Random tests: nearly-equal operands (close-path scenario)
    // ================================================================
    sub_vld = 1'b1;
    for (i = 0; i < 100; i = i + 1) begin
      summand = {$random(seed), $random(seed), $random(seed), $random(seed)} & {108{1'b1}};
      // addend = summand with a few bits flipped
      addend = summand;
      addend[$random(seed) % 108] = ~addend[$random(seed) % 108];
      addend[$random(seed) % 108] = ~addend[$random(seed) % 108];
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
