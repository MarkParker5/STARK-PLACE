import asyncer
from stark import CommandsContext, CommandsManager, Response
from stark.interfaces.protocols import SpeechRecognizer, SpeechSynthesizer
from stark.interfaces.vosk import VoskSpeechRecognizer
from stark.interfaces.silero import SileroSpeechSynthesizer
from stark.voice_assistant import VoiceAssistant
from stark.general.blockage_detector import BlockageDetector


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
        # context.fallback_command = fallback_command
        
        main_task_group.soonify(speech_recognizer.start_listening)()
        main_task_group.soonify(context.handle_responses)()

        detector = BlockageDetector()
        main_task_group.soonify(detector.monitor)()

async def main():
    await run(manager, recognizer, synthesizer)

if __name__ == '__main__':
    asyncer.runnify(main)() # or anyio.run(main), same thing
