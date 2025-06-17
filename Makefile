# Tool-chain
PYTHON = python
PROTOC = $(PYTHON) -m grpc_tools.protoc

# Directories
PROTO_BASE = ./rpc/proto
PROTO_OUT = ./

# Files
PROTO_TARGET = ./rpc/proto/base_nn.proto

protoc:
	$(PROTOC) -I. --python_out=$(PROTO_OUT) --pyi_out=$(PROTO_OUT) --grpc_python_out=$(PROTO_OUT) $(PROTO_TARGET)