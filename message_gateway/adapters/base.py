from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    def __init__(self, name: str):
        self.name = name
        self.gateway = None
    
    @abstractmethod
    async def start(self):
        """Start the adapter (e.g. start polling or listening)"""
        pass
        
    @abstractmethod
    async def send_message(self, user_id: str, text: str):
        """Send a message to a user on this platform"""
        pass
        
    async def handle_incoming(self, user_id: str, text: str) -> str:
        """Call this when a message arrives from the platform"""
        if self.gateway:
            return await self.gateway.handle_message(self.name, user_id, text)
        return "Gateway not connected."
