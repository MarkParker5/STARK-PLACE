'''
Requirements except stark:
    - pvporcupine
'''

import asyncer
from stark import CommandsContext, CommandsManager, Response
from stark.general.blockage_detector import BlockageDetector
from stark.interfaces.protocols import SpeechRecognizer, SpeechSynthesizer
from stark.interfaces.vosk import VoskSpeechRecognizer
from stark.interfaces.silero import SileroSpeechSynthesizer
from stark.voice_assistant import VoiceAssistant, Mode

from stark_place.triggers import porcupine # import the trigger
from stark_place.notifications import sound


vosk_model_ur = 'YOUR_CHOSEN_VOSK_MODEL_URL'
silero_model_ur = 'YOUR_CHOSEN_SILERO_MODEL_URL'

# register at https://console.picovoice.ai and copy the access key
# train and download a model for the your platform at https://console.picovoice.ai/ppn
# don't rename either the model file or the keyword files
access_key = 'YOUR_ACCESS_KEY'
keyword_paths = ['YOUR_KEYWORD_PATH.ppn',]
model_path = 'YOUR_MODEL_PATH.pv'

recognizer = VoskSpeechRecognizer(model_url = vosk_model_ur)
synthesizer = SileroSpeechSynthesizer(model_url = silero_model_ur)

manager = CommandsManager()

@manager.new('hello')
async def hello_command() -> Response:
    text = voice = 'Hello, world!'
    return Response(text=text, voice=voice)

async def run(
    manager: CommandsManager,
    speech_recognizer: SpeechRecognizer,
    speech_synthesizer: SpeechSynthesizer
):
    async with asyncer.create_task_group() as main_task_group:
        context = CommandsContext(
            task_group = main_task_group, 
            commands_manager = manager
        )
        voice_assistant = VoiceAssistant(
            speech_recognizer = speech_recognizer,
            speech_synthesizer = speech_synthesizer,
            commands_context = context
        )
        speech_recognizer.delegate = voice_assistant
        context.delegate = voice_assistant

        voice_assistant.mode = Mode.external() # stop listening after first response
        
        # trigger-listener setup
        
        def add_porcupine_listener():
            # soonify returns immediately, but the wrapped function is added to the task group (main loop)
            # porcuping and speech recognizer use the same microphone device, so they can't run at the same time
            # porcupine.start runs until the first wake-word is detected and needs to be restarted after the speech recognizer is stopped
            def on_wake_word():
                sound.play() # optional: play a sound when the wake-word is detected, check the realisation in stark_place/notifications/sound.py
                main_task_group.soonify(start_speech_recognizer)()
            
            main_task_group.soonify(porcupine.start)(
                access_key = access_key,
                keyword_paths = keyword_paths,
                model_path = model_path,
                callback = on_wake_word
            )
        
        async def start_speech_recognizer():
            await speech_recognizer.start_listening() # awaits until the speech recognizer is stopped
            add_porcupine_listener() # start listening for the wake-word after the speech recognizer is stopped
        
        add_porcupine_listener() # start listening for the wake-word
        
        # other tasks
        
        main_task_group.soonify(context.handle_responses)()

        detector = BlockageDetector()
        main_task_group.soonify(detector.monitor)()

async def main():
    await run(manager, recognizer, synthesizer)

if __name__ == '__main__':
    asyncer.runnify(main)() # or anyio.run(main), same thing
