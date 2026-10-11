from message_gateway.adapters.base import BaseAdapter

class EmailAdapter(BaseAdapter):
    def __init__(self, smtp_server: str, imap_server: str, credentials: dict):
        super().__init__("email")
        self.smtp_server = smtp_server
        self.imap_server = imap_server
        self.credentials = credentials

    async def start(self):
        """Start the email IMAP listener."""
        print("[EmailAdapter] Initialized (stub). Waiting for IMAP/SMTP credentials...")
        # Implement IMAP IDLE or polling here to receive emails.

    async def send_message(self, user_id: str, text: str):
        """Send an email."""
        print(f"[EmailAdapter] (Stub) Sending email to {user_id}: {text}")
