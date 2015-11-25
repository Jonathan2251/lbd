#TRACE=-D TRACE
all:
	iverilog ${TRACE} -o cpu0Is cpu0.v 
	iverilog ${TRACE} -D CPU0II -o cpu0IIs cpu0.v 
	iverilog ${TRACE} -D DLINKER -o cpu0Id cpu0.v 
	iverilog ${TRACE} -D DLINKER -D CPU0II -o cpu0IId cpu0.v 
	iverilog ${TRACE} -D SIMULATE_DELAY_SLOT -o cpu0Isp cpu0.v 
	iverilog ${TRACE} -D SIMULATE_DELAY_SLOT -D CPU0II -o cpu0IIsp cpu0.v 
	iverilog ${TRACE} -D SIMULATE_DELAY_SLOT -D DLINKER -o cpu0Idp cpu0.v 
	iverilog ${TRACE} -D SIMULATE_DELAY_SLOT -D DLINKER -D CPU0II -o cpu0IIdp cpu0.v

.PHONY: clean
clean:
	rm -rf dlconfig cpu0.hex cpu0Is cpu0IIs cpu0Id cpu0IId cpu0Isp cpu0IIsp 
	rm -f cpu0Idp cpu0IIdp *~ cpu0.config

