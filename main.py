import asyncio
from conversation import handle_conversation

async def main():
    while True: 
        await handle_conversation()

if __name__ == "__main__":
    asyncio.run(main())