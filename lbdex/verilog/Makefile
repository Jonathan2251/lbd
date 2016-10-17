#TRACE=-D TRACE
all:
	iverilog ${TRACE} -o cpu0Is cpu0.v 
	iverilog ${TRACE} -D CPU0II -o cpu0IIs cpu0.v 

.PHONY: clean
clean:
	rm -rf cpu0.hex cpu0Is cpu0IIs 
	rm -f *~ cpu0.config

