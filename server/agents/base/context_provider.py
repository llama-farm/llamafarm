from abc import ABC, abstractmethod


class LFAgentContextProvider(ABC):
    title: str

    def __init__(self, title: str):
        self.title = title

    @abstractmethod
    def get_info(self) -> str:
        pass
