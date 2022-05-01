import sys
import importlib


commands = {'sample': ['main', 'sample_from_args']}

help_msg = ('Add one of the following commands and its arguments '
            '(`<command> -h` for help): {}' % list(commands))

if __name__ == '__main__':
    try:
        command_or_input = sys.argv[1].lower()
    except IndexError:  # no command
        print(help_msg)
        exit()
    else:
        module, func = commands.get(command_or_input, (None, None))

        if module is not None:
            sys.argv.pop(1)
            assert func is not None
            getattr(importlib.import_module('cosmofit.' + module), func)()
        else:
            if command_or_input in ['-h', '--help']:
                print(help_msg)
                exit()
            else:
                # no command --> assume run with input file as 1st arg (don't pop!)
                module, func = commands['sample']
                getattr(importlib.import_module('cosmofit.' + module), func)()
