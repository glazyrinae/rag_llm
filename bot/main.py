import os
import aiohttp
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

# Получаем токен бота из переменных окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Инициализируем бота и диспетчер
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

async def call_api(chat_id: str, user_message: str) -> str:
    """Функция для отправки запроса к API"""
    try:
        # Отправляем POST запрос к API
        async with aiohttp.ClientSession() as session:
            data = user_message
            url_response = 'https://api.telegram.org/bot{BOT_TOKEN}/getUpdates'
            async with session.post(
                f"http://api:8000/api/ask?url_response={url_response}&question={user_message}&chat_id={chat_id}",
                json=data,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result.get("result", "Ответ от API пустой")
                else:
                    return f"Ошибка API: статус {response.status}"
                    
    except asyncio.TimeoutError:
        return "Таймаут при обращении к API"
    except Exception as e:
        return f"Ошибка соединения с API: {str(e)}"

@dp.message(Command("start"))
async def start_command(message: Message):
    await message.answer("👋 Привет! Я простой бот. Отправь мне любое сообщение, и я перешлю его в API!")

@dp.message(Command("help"))
async def help_command(message: Message):
    await message.answer("Просто напиши любое сообщение, и я отправлю его на обработку в API!")

@dp.message()
async def handle_all_messages(message: Message):
    # Показываем что бот печатает
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    # Отправляем сообщение в API и получаем ответ
    api_response = await call_api(message.chat.id, message.text)
    
    # Отправляем ответ пользователю
    await message.answer(f"📨 Ваше сообщение: {message.text}\n\n🔔 Ответ от API: {api_response}")

async def main():
    print("Бот запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())