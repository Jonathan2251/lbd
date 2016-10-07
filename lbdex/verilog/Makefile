#TRACE=-D TRACE
all:
	iverilog ${TRACE} -o cpu0Is cpu0.v 
	iverilog ${TRACE} -D CPU0II -o cpu0IIs cpu0.v 
	iverilog ${TRACE} -D SIMULATE_DELAY_SLOT -o cpu0Isp cpu0.v 
	iverilog ${TRACE} -D SIMULATE_DELAY_SLOT -D CPU0II -o cpu0IIsp cpu0.v 

.PHONY: clean
clean:
	rm -rf cpu0.hex cpu0Is cpu0IIs cpu0Isp cpu0IIsp 
	rm -f *~ cpu0.config

