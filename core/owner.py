from core.interface import AbstractData, AbstractVectorStorage


class OwnerContext[D: AbstractData]:

    def __init__(self, vstore: AbstractVectorStorage[D]) -> None:
        self.vstore_ = vstore
        self.qstore_: dict[str, dict] = {}

    def qstore(self, qid: str) -> dict:
        if qid not in self.qstore_:
            self.qstore_[qid] = {}
        return self.qstore_[qid]

    def vstore(self) -> AbstractVectorStorage[D]:
        return self.vstore_
