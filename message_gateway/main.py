import asyncio
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from message_gateway.core import MessageGateway
from message_gateway.adapters.telegram_adapter import TelegramAdapter
from message_gateway.adapters.whatsapp_adapter import WhatsAppAdapter
from message_gateway.adapters.email_adapter import EmailAdapter

async def main():
    gateway = MessageGateway()

    # 1. Register Telegram Adapter
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    allowed_users = [int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()]
    if telegram_token:
        telegram_adapter = TelegramAdapter(token=telegram_token, allowed_users=allowed_users)
        gateway.register_adapter(telegram_adapter)
    else:
        print("Warning: TELEGRAM_BOT_TOKEN not found. Telegram adapter will not be loaded.")

    # 2. Register WhatsApp Adapter (Stub)
    whatsapp_api_key = os.environ.get("WHATSAPP_API_KEY")
    if whatsapp_api_key:
        whatsapp_adapter = WhatsAppAdapter(api_key=whatsapp_api_key)
        gateway.register_adapter(whatsapp_adapter)

    # 3. Register Email Adapter (Stub)
    email_smtp = os.environ.get("EMAIL_SMTP_SERVER")
    if email_smtp:
        email_adapter = EmailAdapter(smtp_server=email_smtp, imap_server="...", credentials={})
        gateway.register_adapter(email_adapter)

    # Start the gateway (starts all registered adapters and the cron scheduler)
    await gateway.start_all()
    
    print("\n[Message Gateway] Running... Press Ctrl+C to stop.")
    
    # Keep the main thread alive since adapters run as async tasks
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Message Gateway] Stopped.")
