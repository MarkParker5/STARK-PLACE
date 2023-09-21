'''
Requirements except stark:
    - pynput
'''

import asyncer
from stark import CommandsContext, CommandsManager, Response
from stark.general.blockage_detector import BlockageDetector
from stark.interfaces.protocols import SpeechRecognizer, SpeechSynthesizer
from stark.interfaces.vosk import VoskSpeechRecognizer
from stark.interfaces.silero import SileroSpeechSynthesizer
from stark.voice_assistant import VoiceAssistant, Mode

from stark_place.triggers import keyboard_key # import the trigger
from stark_place.notifications import sound


vosk_model_ur = 'YOUR_CHOSEN_VOSK_MODEL_URL'
silero_model_ur = 'YOUR_CHOSEN_SILERO_MODEL_URL'

recognizer = VoskSpeechRecognizer(model_url=vosk_model_ur)
synthesizer = SileroSpeechSynthesizer(model_url=silero_model_ur)

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
        
        def on_hotkey():
            sound.play() # optional: play a sound when the hotkey is pressed, check the realisation in stark_place/notifications/sound.py
            main_task_group.soonify(speech_recognizer.start_listening)
        
        main_task_group.soonify(keyboard_key.start)(on_hotkey) # add trigger to the main loop
        # main_task_group.soonify(speech_recognizer.start_listening)() # don't start listening until the hotkey is pressed
        main_task_group.soonify(context.handle_responses)()

        detector = BlockageDetector()
        main_task_group.soonify(detector.monitor)()

async def main():
    await run(manager, recognizer, synthesizer)

if __name__ == '__main__':
    asyncer.runnify(main)() # or anyio.run(main), same thing
