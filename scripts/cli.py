import sys

from absl import app

AVAILABLE_SCRIPTS = [
    'preprocess', 
    'info'
]


def help():
    print(f"""usage: acids-dataset [ {' | '.join(AVAILABLE_SCRIPTS)} ]

positional arguments:
  command     Command to launch with rave. valid commands : {' | '.join(AVAILABLE_SCRIPTS)} 
""")
    exit()


def main():
    if len(sys.argv) == 1:
        help()
    elif sys.argv[1] not in AVAILABLE_SCRIPTS:
        help()

    command = sys.argv[1]

    if command == 'preprocess':
        from scripts import preprocess
        sys.argv[0] = preprocess.__name__
        app.run(preprocess.main)
    elif command == 'info':
        from scripts import info 
        sys.argv[0] = info.__name__
        app.run(info.main)