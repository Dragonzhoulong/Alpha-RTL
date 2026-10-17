module multi_16bit (
    input            clk,
    input            rst_n,
    input            start,
    input     [15:0] ain,
    input     [15:0] bin,
    output    [31:0] yout,
    output           done
);
    // Shift-register variant of the reference: instead of a 32-bit variable
    // shifter `breg << (i-1)`, pre-shift bin by 1 each cycle. Same #cycles
    // as the reference but the per-cycle critical path drops to a 32-bit
    // add + a 16/32-bit AND-mask.
    reg [31:0] b_shifted;
    reg [15:0] a_r;
    reg [31:0] acc;
    reg [ 4:0] i;
    reg        done_r;

    wire [31:0] addend = a_r[0] ? b_shifted : 32'd0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            b_shifted <= 32'd0;
            a_r       <= 16'd0;
            acc       <= 32'd0;
            i         <= 5'd0;
            done_r    <= 1'b0;
        end else if (start) begin
            if (i == 5'd0) begin
                a_r       <= ain;
                b_shifted <= {16'b0, bin};
                acc       <= 32'd0;
                done_r    <= 1'b0;
                i         <= 5'd1;
            end else if (i < 5'd17) begin
                acc       <= acc + addend;
                a_r       <= {1'b0, a_r[15:1]};
                b_shifted <= b_shifted << 1;
                i         <= i + 1'b1;
                if (i == 5'd16) done_r <= 1'b1;
            end
        end else begin
            i      <= 5'd0;
            done_r <= 1'b0;
        end
    end

    assign yout = acc;
    assign done = done_r;
endmodule
