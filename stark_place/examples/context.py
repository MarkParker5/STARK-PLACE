import anyio
from stark import run, CommandsManager, Response
from stark.core import ResponseHandler
from stark.core.types import Word
from stark.interfaces.vosk import VoskSpeechRecognizer
from stark.interfaces.silero import SileroSpeechSynthesizer


VOSK_MODEL_URL = "YOUR_CHOSEN_VOSK_MODEL_URL"
SILERO_MODEL_URL = "YOUR_CHOSEN_SILERO_MODEL_URL"

recognizer = VoskSpeechRecognizer(model_url=VOSK_MODEL_URL)
synthesizer = SileroSpeechSynthesizer(model_url=SILERO_MODEL_URL)

manager = CommandsManager()

@manager.new('hello', hidden=True) 
def hello_context(**params):
    voice = text = f'Hi, {params["name"]}!'
    return Response(text=text, voice=voice)

@manager.new('bye', hidden=True)
def bye_context(name: Word, handler: ResponseHandler):
    handler.pop_context()
    return Response(text=f'Bye, {name}!')

@manager.new('hello $name:Word')
def hello(name: Word):
    text = voice = f'Hello, {name}!'
    return Response(
        text=text,
        voice=voice,
        commands=[hello_context, bye_context],
        parameters={'name': name}
    )

async def main():
    await run(manager, recognizer, synthesizer)

if __name__ == '__main__':
    anyio.run(main)
