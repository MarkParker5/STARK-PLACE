from typing import Callable, Any
import os
from pynput.keyboard import Key, KeyCode, Listener, HotKey
import anyio
from asyncer import asyncify
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
        
        if hotkey_event.is_set():
            '''
            optional: play a sound to indicate that the hotkey was pressed
            
            play all sounds in macos using shell: sh`for s in /System/Library/Sounds/*; do echo "$s" && afplay "$s"; done`
            for linux: check the `/usr/share/sounds/` directory and use `aplay` instead of `afplay`
            
            as an alternative, you can use the system-sounds pypi library to list and play sounds
            pypi.org/project/system-sounds
            or github.com/MarkParker5/system-sounds
            
            or use the a SpeechSynthesizer to say something like "Yes, sir?"
            '''
            os.system('afplay /System/Library/Sounds/Blow.aiff &') # play the sound in the background (macos)
            callback()
            hotkey_event.clear()
