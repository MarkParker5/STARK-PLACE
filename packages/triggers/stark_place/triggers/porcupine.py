from typing import Any, Callable
import anyio
import pvporcupine
from pvrecorder import PvRecorder


async def start(
    access_key: str,
    keyword_paths: list[str],
    model_path: str,
    callback: Callable[[], Any | None]
):
    # register at https://console.picovoice.ai and copy the access key
    # train and download a model for the your platform at https://console.picovoice.ai/ppn
    # don't rename either the model file or the keyword files
    porcupine = pvporcupine.create(
        access_key = access_key,
        keyword_paths = keyword_paths,
        model_path = model_path
    )

    recorder = PvRecorder(
        frame_length = porcupine.frame_length,
        device_index = 0
    )
    recorder.start()

    try:
        while True: # this loop runs in the main thread
            await anyio.sleep(0.01) # release the thread for other tasks
            
            pcm = recorder.read()
            keyword_index = porcupine.process(pcm)
            
            if keyword_index == -1: 
                continue
            
            callback()
            break # stop listening after the first keyword is detected to release the microphone device
    finally:
        # stop the recorder, delete the porcupine instances, and release the microphone device
        porcupine.delete()
        recorder.delete()
