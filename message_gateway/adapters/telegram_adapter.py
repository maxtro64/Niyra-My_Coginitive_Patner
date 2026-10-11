import os
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from message_gateway.adapters.base import BaseAdapter

class TelegramAdapter(BaseAdapter):
    def __init__(self, token: str, allowed_users: list[int] = None):
        super().__init__("telegram")
        self.token = token
        self.allowed = allowed_users or []
        self.app = None

    def _allowed(self, update: Update) -> bool:
        if not self.allowed:
            return True 
        return bool(update.effective_user) and update.effective_user.id in self.allowed

    async def _on_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._allowed(update):
            return
        await ctx.bot.send_chat_action(update.effective_chat.id, "typing")
        user_id = str(update.effective_chat.id)
        text = update.message.text
        
        response = await self.handle_incoming(user_id, text)
        await update.message.reply_text(response)

    async def start(self):
        """Start the telegram bot polling."""
        print("[TelegramAdapter] Starting bot...")
        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(MessageHandler(filters.TEXT, self._on_message))
        
        # Initialize and start the application asynchronously
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        # Note: We do not call run_polling() because it blocks. This allows concurrent adapters.

    async def send_message(self, user_id: str, text: str):
        """Send a message to a specific user id (chat id)."""
        if self.app and self.app.bot:
            await self.app.bot.send_message(chat_id=int(user_id), text=text)
