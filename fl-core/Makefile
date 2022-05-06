UNAME_S := $(shell uname -s)
generate-rpc:
	- python -m grpc_tools.protoc -I ./protos --python_out=./protos --grpc_python_out=./protos ./protos/peace_maker_core.proto


