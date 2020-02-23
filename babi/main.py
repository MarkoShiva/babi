import argparse
import curses
import os.path
from typing import Optional
from typing import Sequence

from babi.file import File
from babi.highlight import load_grammars
from babi.highlight import Theme
from babi.screen import EditResult
from babi.screen import make_stdscr
from babi.screen import Screen
from babi.user_data import xdg_config


def _edit(screen: Screen) -> EditResult:
    screen.file.ensure_loaded(screen.status)

    while True:
        screen.status.tick(screen.margin)
        screen.draw()
        screen.file.move_cursor(screen.stdscr, screen.margin)

        key = screen.get_char()
        if key.keyname in File.DISPATCH:
            File.DISPATCH[key.keyname](screen.file, screen.margin)
        elif key.keyname in Screen.DISPATCH:
            ret = Screen.DISPATCH[key.keyname](screen)
            if isinstance(ret, EditResult):
                return ret
        elif isinstance(key.wch, str) and key.wch.isprintable():
            screen.file.c(key.wch, screen.margin)
        else:
            screen.status.update(f'unknown key: {key}')


def c_main(
        stdscr: 'curses._CursesWindow',
        theme: Theme,
        args: argparse.Namespace,
) -> None:
    grammars = load_grammars()
    screen = Screen(
        stdscr,
        [File(f, grammars) for f in args.filenames or [None]],
        theme,
    )
    with screen.perf.log(args.perf_log), screen.history.save():
        while screen.files:
            screen.i = screen.i % len(screen.files)
            res = _edit(screen)
            if res == EditResult.EXIT:
                del screen.files[screen.i]
                screen.status.clear()
            elif res == EditResult.NEXT:
                screen.i += 1
                screen.status.clear()
            elif res == EditResult.PREV:
                screen.i -= 1
                screen.status.clear()
            else:
                raise AssertionError(f'unreachable {res}')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', metavar='filename', nargs='*')
    parser.add_argument('--perf-log')
    args = parser.parse_args(argv)

    theme_filename = xdg_config('theme.json')
    if os.path.exists(theme_filename):
        theme = Theme.parse(theme_filename)
    else:
        theme = Theme.blank()

    with make_stdscr(theme) as stdscr:
        c_main(stdscr, theme, args)
    return 0


if __name__ == '__main__':
    exit(main())
