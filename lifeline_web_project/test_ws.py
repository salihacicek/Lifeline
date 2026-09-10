import asyncio
import websockets

async def test_ws():
    uri = "ws://localhost:8000/ws/ecg/2025-07-21_Rec002.csv"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected")
            msg = await websocket.recv()
            print(f"Received: {msg}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
