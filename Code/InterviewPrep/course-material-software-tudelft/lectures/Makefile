LECTURES := $(shell find $(EDIR) -maxdepth 1 -type d -name 'lecture*')

all:
	$(foreach c,$(LECTURES),$(MAKE) -C $(c) && ) true
