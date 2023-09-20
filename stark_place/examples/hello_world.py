import anyio
from stark import run, CommandsManager, Response
from stark.interfaces.vosk import VoskSpeechRecognizer
from stark.interfaces.silero import SileroSpeechSynthesizer


VOSK_MODEL_URL = "YOUR_CHOSEN_VOSK_MODEL_URL"
SILERO_MODEL_URL = "YOUR_CHOSEN_SILERO_MODEL_URL"

recognizer = VoskSpeechRecognizer(model_url=VOSK_MODEL_URL)
synthesizer = SileroSpeechSynthesizer(model_url=SILERO_MODEL_URL)

manager = CommandsManager()

@manager.new('hello')
async def hello_command() -> Response:
    text = voice = 'Hello, world!'
    return Response(text=text, voice=voice)

async def main():
    await run(manager, recognizer, synthesizer)

if __name__ == '__main__':
    anyio.run(main)
