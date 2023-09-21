import os


def play():
    '''
    play all sounds in macos using shell: sh`for s in /System/Library/Sounds/*; do echo "$s" && afplay "$s"; done`
    for linux: check the `/usr/share/sounds/` directory and use `aplay` instead of `afplay`
    
    as an alternative, you can use the cross-platform system-sounds pypi library to list and play sounds
    pypi.org/project/system-sounds
    or github.com/MarkParker5/system-sounds
    
    or use the a SpeechSynthesizer to say something like "Yes, sir?"
    '''
    os.system('afplay /System/Library/Sounds/Blow.aiff &') # play the sound in the background (non-blocking, immediate return) (macos)
