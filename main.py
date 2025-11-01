import asyncio
from conversation import handle_conversation, wake_word_detection

async def main():
    while True: 
        await wake_word_detection()
        await handle_conversation()

if __name__ == "__main__":
    asyncio.run(main())