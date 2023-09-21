from typing import Callable, Any
from pynput.keyboard import Key, KeyCode, Listener
import anyio
from threading import Event


async def start(callback: Callable[[], Any | None]):
    hotkey_event = Event()
    
    def on_release(key: Key):
        # this function is called in a separate thread, so we need to use a thread-safe way to communicate with the main thread
        nonlocal hotkey_event
        if key == KeyCode(63): # macos globe button
            hotkey_event.set()

    listener = Listener(on_release = on_release)
    listener.start() # start listening in a separate thread
    
    while True: # this loop runs in the main thread
        await anyio.sleep(0.1) # release the thread for other tasks
        
        if not hotkey_event.is_set(): 
            continue
        
        callback()
        hotkey_event.clear()
