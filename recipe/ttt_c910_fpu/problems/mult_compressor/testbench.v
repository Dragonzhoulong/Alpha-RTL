/*
 * Verification testbench for ct_vfmau_mult_compressor.
 *
 * Instantiates both the REFERENCE (golden) and the CANDIDATE (DUT) module,
 * drives identical random + edge-case inputs, and compares sum/carry/simd
 * outputs cycle-by-cycle.
 *
 * Usage (iverilog):
 *   iverilog -o simv -g2005-sv \
 *     ct_vfmau_mult_compressor.v  \   <-- candidate
 *     reference_renamed.v          \   <-- reference (module renamed to ct_vfmau_mult_compressor_ref)
 *     deps/compressor_32.v deps/compressor_42.v deps/booth_code_v1.v deps/gated_clk_cell.v \
 *     testbench.v
 *   vvp simv
 */

`timescale 1ns / 1ps

module testbench;

  // ---- Parameters ----
  localparam CLK_PERIOD = 10;
  localparam NUM_RANDOM_TESTS = 500;
  localparam NUM_EDGE_TESTS   = 50;

  // ---- Signals ----
  reg         clk;
  reg         cp0_vfpu_icg_en;
  reg         cp0_yy_clk_en;
  reg         dp_xx_ex1_double;
  reg         dp_xx_ex1_half;
  reg         dp_xx_ex1_single;
  reg         mult1_ex1_op0_half0_hidden_bit;
  reg         mult1_ex1_op0_hidden_bit;
  reg         mult1_ex1_op1_half0_hidden_bit;
  reg         mult1_ex1_op1_hidden_bit;
  reg  [51:0] op0_frac;
  reg  [51:0] op1_frac;
  reg         pad_yy_icg_scan_en;
  reg         pipe_down;

  // Reference outputs
  wire [105:0] ref_carry;
  wire [21:0]  ref_simd_half0_product;
  wire [105:0] ref_sum;

  // DUT outputs
  wire [105:0] dut_carry;
  wire [21:0]  dut_simd_half0_product;
  wire [105:0] dut_sum;

  // Sum-equivalence: compare mathematical product, not individual sum/carry
  wire [106:0] ref_product = {1'b0, ref_sum} + {1'b0, ref_carry};
  wire [106:0] dut_product = {1'b0, dut_sum} + {1'b0, dut_carry};

  // Counters
  integer total_tests;
  integer pass_count;
  integer fail_count;
  integer i;
  integer seed;

  // ---- Clock generation ----
  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  // ---- Reference (golden model) ----
  ct_vfmau_mult_compressor_ref u_ref (
    .carry                          (ref_carry),
    .cp0_vfpu_icg_en                (cp0_vfpu_icg_en),
    .cp0_yy_clk_en                  (cp0_yy_clk_en),
    .dp_xx_ex1_double               (dp_xx_ex1_double),
    .dp_xx_ex1_half                 (dp_xx_ex1_half),
    .dp_xx_ex1_single               (dp_xx_ex1_single),
    .forever_cpuclk                 (clk),
    .mult1_ex1_op0_half0_hidden_bit (mult1_ex1_op0_half0_hidden_bit),
    .mult1_ex1_op0_hidden_bit       (mult1_ex1_op0_hidden_bit),
    .mult1_ex1_op1_half0_hidden_bit (mult1_ex1_op1_half0_hidden_bit),
    .mult1_ex1_op1_hidden_bit       (mult1_ex1_op1_hidden_bit),
    .op0_frac                       (op0_frac),
    .op1_frac                       (op1_frac),
    .pad_yy_icg_scan_en             (pad_yy_icg_scan_en),
    .pipe_down                      (pipe_down),
    .simd_half0_product             (ref_simd_half0_product),
    .sum                            (ref_sum)
  );

  // ---- DUT (candidate) ----
  ct_vfmau_mult_compressor u_dut (
    .carry                          (dut_carry),
    .cp0_vfpu_icg_en                (cp0_vfpu_icg_en),
    .cp0_yy_clk_en                  (cp0_yy_clk_en),
    .dp_xx_ex1_double               (dp_xx_ex1_double),
    .dp_xx_ex1_half                 (dp_xx_ex1_half),
    .dp_xx_ex1_single               (dp_xx_ex1_single),
    .forever_cpuclk                 (clk),
    .mult1_ex1_op0_half0_hidden_bit (mult1_ex1_op0_half0_hidden_bit),
    .mult1_ex1_op0_hidden_bit       (mult1_ex1_op0_hidden_bit),
    .mult1_ex1_op1_half0_hidden_bit (mult1_ex1_op1_half0_hidden_bit),
    .mult1_ex1_op1_hidden_bit       (mult1_ex1_op1_hidden_bit),
    .op0_frac                       (op0_frac),
    .op1_frac                       (op1_frac),
    .pad_yy_icg_scan_en             (pad_yy_icg_scan_en),
    .pipe_down                      (pipe_down),
    .simd_half0_product             (dut_simd_half0_product),
    .sum                            (dut_sum)
  );

  // ---- Helper tasks ----

  task apply_inputs_and_check;
    reg product_ok;
    reg simd_ok;
    begin
      // Drive inputs on negedge to avoid setup races
      @(negedge clk);
      pipe_down = 1'b1;

      // Wait for EX1 pipeline register to latch
      @(posedge clk);
      // Wait for EX2 combinational to settle
      @(negedge clk);

      // Now drive pipe_down=0 so registers hold, then check EX2 outputs
      // Actually: the outputs sum/carry are combinational from the registered
      // values. After one pipe_down=1 clock edge, EX2 outputs are valid on
      // the same cycle's combinational path. Check after the next negedge.

      total_tests = total_tests + 1;
      product_ok = (dut_product === ref_product);
      simd_ok = (!dp_xx_ex1_half) || (dut_simd_half0_product === ref_simd_half0_product);

      // Compare mathematical equivalence: sum + carry must match
      if (!product_ok) begin
        fail_count = fail_count + 1;
      end

      if (!product_ok) begin
        if (fail_count <= 10) begin
          $display("FAIL test %0d: product mismatch (sum+carry)", total_tests);
          $display("  op0_frac=%h op1_frac=%h mode=D%0b/S%0b/H%0b",
                   op0_frac, op1_frac, dp_xx_ex1_double, dp_xx_ex1_single, dp_xx_ex1_half);
          $display("  REF product=%h (sum=%h carry=%h)", ref_product, ref_sum, ref_carry);
          $display("  DUT product=%h (sum=%h carry=%h)", dut_product, dut_sum, dut_carry);
        end
      end

      // Also check simd_half0_product (combinational in EX1, no pipeline delay)
      if (!simd_ok) begin
        fail_count = fail_count + 1;
        if (fail_count <= 10) begin
          $display("FAIL test %0d: simd_half0_product mismatch", total_tests);
          $display("  REF=%h DUT=%h", ref_simd_half0_product, dut_simd_half0_product);
        end
      end

      if (product_ok && simd_ok) begin
        pass_count = pass_count + 1;
      end
    end
  endtask

  task set_double_mode;
    begin
      dp_xx_ex1_double = 1'b1;
      dp_xx_ex1_single = 1'b0;
      dp_xx_ex1_half   = 1'b0;
      mult1_ex1_op0_hidden_bit = 1'b1;
      mult1_ex1_op1_hidden_bit = 1'b1;
      mult1_ex1_op0_half0_hidden_bit = 1'b0;
      mult1_ex1_op1_half0_hidden_bit = 1'b0;
    end
  endtask

  task set_single_mode;
    begin
      dp_xx_ex1_double = 1'b0;
      dp_xx_ex1_single = 1'b1;
      dp_xx_ex1_half   = 1'b0;
      mult1_ex1_op0_hidden_bit = 1'b1;
      mult1_ex1_op1_hidden_bit = 1'b1;
      mult1_ex1_op0_half0_hidden_bit = 1'b0;
      mult1_ex1_op1_half0_hidden_bit = 1'b0;
    end
  endtask

  task set_half_mode;
    begin
      dp_xx_ex1_double = 1'b0;
      dp_xx_ex1_single = 1'b0;
      dp_xx_ex1_half   = 1'b1;
      mult1_ex1_op0_hidden_bit = 1'b0;
      mult1_ex1_op1_hidden_bit = 1'b0;
      mult1_ex1_op0_half0_hidden_bit = 1'b1;
      mult1_ex1_op1_half0_hidden_bit = 1'b1;
    end
  endtask

  // ---- Main test sequence ----
  initial begin
    // Initialize
    total_tests = 0;
    pass_count  = 0;
    fail_count  = 0;
    seed        = 42;

    cp0_vfpu_icg_en    = 1'b1;
    cp0_yy_clk_en      = 1'b1;
    pad_yy_icg_scan_en = 1'b0;
    pipe_down          = 1'b0;
    op0_frac           = 52'b0;
    op1_frac           = 52'b0;

    set_double_mode;

    // Reset cycle
    @(posedge clk);
    @(posedge clk);

    // ---- Edge cases: double precision ----
    set_double_mode;

    // Zero * Zero
    op0_frac = 52'h0; op1_frac = 52'h0;
    apply_inputs_and_check;

    // Max * Max
    op0_frac = 52'hF_FFFF_FFFF_FFFF; op1_frac = 52'hF_FFFF_FFFF_FFFF;
    apply_inputs_and_check;

    // One * Max
    op0_frac = 52'h0_0000_0000_0001; op1_frac = 52'hF_FFFF_FFFF_FFFF;
    apply_inputs_and_check;

    // Max * One
    op0_frac = 52'hF_FFFF_FFFF_FFFF; op1_frac = 52'h0_0000_0000_0001;
    apply_inputs_and_check;

    // Alternating bits
    op0_frac = 52'hA_AAAA_AAAA_AAAA; op1_frac = 52'h5_5555_5555_5555;
    apply_inputs_and_check;

    // Hidden bit = 0 (denormal)
    mult1_ex1_op0_hidden_bit = 1'b0;
    op0_frac = 52'h8_0000_0000_0000; op1_frac = 52'hF_FFFF_FFFF_FFFF;
    apply_inputs_and_check;
    mult1_ex1_op0_hidden_bit = 1'b1;

    mult1_ex1_op1_hidden_bit = 1'b0;
    op0_frac = 52'hF_FFFF_FFFF_FFFF; op1_frac = 52'h8_0000_0000_0000;
    apply_inputs_and_check;
    mult1_ex1_op1_hidden_bit = 1'b1;

    // Both denormal
    mult1_ex1_op0_hidden_bit = 1'b0;
    mult1_ex1_op1_hidden_bit = 1'b0;
    op0_frac = 52'hA_AAAA_AAAA_AAAA; op1_frac = 52'h5_5555_5555_5555;
    apply_inputs_and_check;
    mult1_ex1_op0_hidden_bit = 1'b1;
    mult1_ex1_op1_hidden_bit = 1'b1;

    // ---- Edge cases: single precision ----
    set_single_mode;

    op0_frac = 52'h0; op1_frac = 52'h0;
    apply_inputs_and_check;

    op0_frac = {29'b0, 23'h7FFFFF}; op1_frac = {29'b0, 23'h7FFFFF};
    apply_inputs_and_check;

    op0_frac = {29'b0, 23'h555555}; op1_frac = {29'b0, 23'h2AAAAA};
    apply_inputs_and_check;

    // ---- Edge cases: half precision ----
    set_half_mode;

    op0_frac = 52'h0; op1_frac = 52'h0;
    apply_inputs_and_check;

    op0_frac = {42'b0, 10'h3FF}; op1_frac = {42'b0, 10'h3FF};
    apply_inputs_and_check;

    op0_frac = {42'b0, 10'h2AA}; op1_frac = {42'b0, 10'h155};
    apply_inputs_and_check;

    // ---- Random tests: double precision ----
    set_double_mode;
    for (i = 0; i < NUM_RANDOM_TESTS; i = i + 1) begin
      op0_frac = {$random(seed), $random(seed)} & {52{1'b1}};
      op1_frac = {$random(seed), $random(seed)} & {52{1'b1}};
      // Randomly toggle hidden bits
      if (i % 20 == 0) begin
        mult1_ex1_op0_hidden_bit = $random(seed) & 1;
        mult1_ex1_op1_hidden_bit = $random(seed) & 1;
      end
      apply_inputs_and_check;
    end
    // Restore hidden bits
    mult1_ex1_op0_hidden_bit = 1'b1;
    mult1_ex1_op1_hidden_bit = 1'b1;

    // ---- Random tests: single precision ----
    set_single_mode;
    for (i = 0; i < NUM_RANDOM_TESTS / 2; i = i + 1) begin
      op0_frac = {29'b0, $random(seed)} & {52{1'b1}};
      op1_frac = {29'b0, $random(seed)} & {52{1'b1}};
      apply_inputs_and_check;
    end

    // ---- Random tests: half precision ----
    set_half_mode;
    for (i = 0; i < NUM_RANDOM_TESTS / 2; i = i + 1) begin
      op0_frac = {42'b0, $random(seed)} & {52{1'b1}};
      op1_frac = {42'b0, $random(seed)} & {52{1'b1}};
      if (i % 10 == 0) begin
        mult1_ex1_op0_half0_hidden_bit = $random(seed) & 1;
        mult1_ex1_op1_half0_hidden_bit = $random(seed) & 1;
      end
      apply_inputs_and_check;
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
