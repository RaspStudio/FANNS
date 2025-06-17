import grpc
import functools
import concurrent.futures as futures

from typing import Callable, TypeVar, Generic


class FedSource:

    def __init__(self, id: int, target: str, method_map: dict[str, tuple[type[object], str, Callable, Callable]] = {}):
        self.id_ = id
        self.target_ = target
        self.stubs_: dict[type[object], object] = {}
        self.comm = 0

        for method_name, (stub_class, method_name_in_stub, request_wrapper, response_unwrapper) in method_map.items():
            if stub_class not in self.stubs_:
                channel = grpc.insecure_channel(target, options=[(
                    'grpc.max_send_message_length', 1000 * 1024 * 1024), ('grpc.max_receive_message_length', 1000 * 1024 * 1024)])
                self.stubs_[stub_class] = stub_class(channel)
            stub = self.stubs_[stub_class]

            method_in_stub = getattr(stub, method_name_in_stub)
            method_wrapped = functools.partial(
                self._call_stub_method, method_in_stub, request_wrapper, response_unwrapper)
            setattr(self, method_name, method_wrapped)

    def _call_stub_method(self, method: Callable, request_wrapper: Callable, response_unwrapper: Callable, *args, **kwargs):
        request = request_wrapper(*args, **kwargs)
        response = method(request)
        self.comm += request.ByteSize() + response.ByteSize()
        return response_unwrapper(response)


class FederationManager:

    def __init__(self, n_sources: int, targets: list[str], id: int = -1, method_map: dict[str, tuple[type[object], str, Callable, Callable]] = {}):
        self.n_sources_ = n_sources
        self.sources_: list[FedSource] = [
            FedSource(id, target, method_map) if id != fid else None
            for fid, target in enumerate(targets)
        ]

        for method_name in method_map.keys():
            method_wrapped = functools.partial(
                self._call_source_method, method_name)
            setattr(self, method_name, method_wrapped)

    def _call_source_method(self, method_name: str, id: int, *args, **kwargs):
        source = self.sources_[id]
        if source is None:
            raise ValueError(f"Source {id} is not available")
        return getattr(source, method_name)(*args, **kwargs)

    def get_total_comm(self):
        total_com = 0
        for src in self.sources_:
            if src is not None:
                total_com += src.comm
        return total_com


S = TypeVar('S', bound=object)


class AbstractServicer(Generic[S]):

    def __init__(self, method_map: dict[str, tuple[Callable, Callable, Callable]], *args, **kwargs):
        self.args_ = args
        self.kwargs_ = kwargs

        for method_name, (handler, request_unwrapper, response_wrapper) in method_map.items():
            method_wrapped = functools.partial(
                self._call_handler, handler, request_unwrapper, response_wrapper)
            setattr(self, method_name, method_wrapped)

    def _call_handler(self, handler: Callable, request_unwrapper: Callable, response_wrapper: Callable, request, context):
        request = request_unwrapper(request)
        if isinstance(request, tuple):
            response = handler(*request, *self.args_, **self.kwargs_)
        else:
            response = handler(request, *self.args_, **self.kwargs_)
        if isinstance(response, tuple):
            return response_wrapper(*response)
        else:
            return response_wrapper(response)


class ServiceManager:
    def __init__(self, port: str, services: list[tuple[AbstractServicer, Callable]], *args, **kwargs):
        self.server_ = grpc.server(futures.ThreadPoolExecutor(max_workers=3), options=[(
            'grpc.max_send_message_length', 1000 * 1024 * 1024), ('grpc.max_receive_message_length', 1000 * 1024 * 1024)])
        self.server_.add_insecure_port("[::]:" + port)
        self.port_ = port

        for servicer, add_to_server in services:
            add_to_server(servicer, self.server_)

    def serve(self):
        self.server_.start()
        print("Server started, listening on " + self.port_, flush=True)
        self.server_.wait_for_termination()
        print("Server on " + self.port_ + " terminated")
