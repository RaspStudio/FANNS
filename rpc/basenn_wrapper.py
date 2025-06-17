from rpc.proto.base_nn_pb2_grpc import BaseNNServiceServicer, add_BaseNNServiceServicer_to_server
from rpc.federation import AbstractServicer
from typing import Callable
from rpc.proto.base_nn_pb2_grpc import BaseNNServiceStub
from rpc.proto.base_nn_pb2 import BaseNNRequest, BaseNNResponse


def pack_basenn_request(id: str, data: bytes, phase: int, args: list[int], fargs: list[float]) -> BaseNNRequest:
    return BaseNNRequest(id=id, data=data, phase=phase, args=args, fargs=fargs)


def unpack_basenn_request(request: BaseNNRequest) -> tuple[str, bytes, int, list[int], list[float]]:
    return request.id, request.data, request.phase, request.args, request.fargs


def pack_basenn_response(datas: list[bytes]) -> BaseNNResponse:
    return BaseNNResponse(datas=datas)


def unpack_basenn_response(response: BaseNNResponse) -> list[bytes]:
    return [data for data in response.datas]


# ---------------------# Client Implementations #---------------------#

basenn_method_map = {
    "base_nn": (BaseNNServiceStub, "base_nn", pack_basenn_request, unpack_basenn_response)
}


class BaseNNServicer(AbstractServicer[BaseNNServiceServicer]):
    ...


def get_servicer(handler: Callable, *args, **kwargs) -> tuple[AbstractServicer, Callable]:
    return BaseNNServicer({
        "base_nn": (handler, unpack_basenn_request, pack_basenn_response)
    }, *args, **kwargs), add_BaseNNServiceServicer_to_server
