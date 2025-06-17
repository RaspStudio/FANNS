from typing import Callable


class FederationManager:
    n_sources_: int

    def __init__(self, n_sources: int, targets: list[str], id: int = -1, method_map: dict[str, tuple[type[object], str, Callable, Callable]] = {}):
        ...

    # basenn.proto
    def base_nn(self, fid: int, id: str, data: bytes, phase: int, args: list[int], fargs: list[float]) -> list[bytes]:
        ...

    def get_total_comm(self) -> int:
        ...


class AbstractServicer:

    def __init__(self, method_map: dict[str, tuple[Callable, Callable]], *args, **kwargs):
        ...


class ServiceManager:
    def __init__(self, port: str, services: list[tuple[AbstractServicer, Callable]], *args, **kwargs):
        ...

    def serve(self):
        ...
