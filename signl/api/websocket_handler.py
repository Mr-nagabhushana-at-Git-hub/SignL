# src/majorSignL/api/websocket_handler.py
import asyncio
import logging
import numpy as np
import cv2
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🔌 Connection +1 | Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"🔌 Connection -1 | Total: {len(self.active_connections)}")

    async def broadcast_json(self, data: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

async def websocket_endpoint(websocket: WebSocket, app_state):
    await app_state.websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                app_state.latest_frame = frame
                
    except WebSocketDisconnect:
        pass
    finally:
        app_state.websocket_manager.disconnect(websocket)
