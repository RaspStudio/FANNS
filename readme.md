# Federated ANN Search 

A Federated ANN Search system based on [faiss](https://github.com/facebookresearch/faiss), [pytorch](https://github.com/pytorch/pytorch), and [gRPC](https://github.com/grpc/grpc).


## Run Example

To start the FANNS server and run a query test, you need to clone the repository, install the dependencies.
Then, generate the gRPC sources, configure the server, and finally start the server and run the query test.

```bash
# 1. Generate GRPC sources
make

# 2. Configure the server in `config.py`
vi ./config.py

# 3. Start the server
python ./test/run_server.py

# 4. Start query test
python ./test/run_query.py
```

