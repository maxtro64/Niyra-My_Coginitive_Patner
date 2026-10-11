from message_gateway.adapters.base import BaseAdapter

class WhatsAppAdapter(BaseAdapter):
    def __init__(self, api_key: str):
        super().__init__("whatsapp")
        self.api_key = api_key

    async def start(self):
        """Start the WhatsApp webhook or long-polling client."""
        print("[WhatsAppAdapter] Initialized (stub). Waiting for API credentials...")
        # Implement Twilio or WhatsApp Business API integration here.

    async def send_message(self, user_id: str, text: str):
        """Send a WhatsApp message."""
        print(f"[WhatsAppAdapter] (Stub) Sending to {user_id}: {text}")
