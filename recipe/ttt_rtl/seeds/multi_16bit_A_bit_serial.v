module multi_16bit (
    input            clk,
    input            rst_n,
    input            start,
    input     [15:0] ain,
    input     [15:0] bin,
    output    [31:0] yout,
    output           done
);
    // LSB-first bit-serial multiplier.
    // The 32-bit acc holds the partial product in [31:16] and the multiplier
    // bits being consumed in [15:0]. Each cycle: optionally add multiplicand
    // to the upper half, then right-shift the whole 32-bit register by 1.
    // Per-cycle work = a 16-bit AND + a 16-bit add + a register shift, much
    // smaller than the reference's 32-bit variable shifter.
    reg [31:0] acc;
    reg [15:0] mcand;
    reg [ 4:0] cnt;
    reg        done_r;

    wire [15:0] addend = mcand & {16{acc[0]}};
    wire [16:0] sum_hi = {1'b0, acc[31:16]} + {1'b0, addend};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc    <= 32'd0;
            mcand  <= 16'd0;
            cnt    <= 5'd0;
            done_r <= 1'b0;
        end else if (start) begin
            if (cnt == 5'd0) begin
                acc    <= {16'b0, bin};
                mcand  <= ain;
                cnt    <= 5'd1;
                done_r <= 1'b0;
            end else if (cnt < 5'd17) begin
                acc <= {sum_hi, acc[15:1]};
                cnt <= cnt + 1'b1;
                if (cnt == 5'd16) done_r <= 1'b1;
            end
        end else begin
            cnt    <= 5'd0;
            done_r <= 1'b0;
        end
    end

    assign yout = acc;
    assign done = done_r;
endmodule
