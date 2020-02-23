import curses
import functools
import itertools
import json
import os.path
import plistlib
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import Generator
from typing import Generic
from typing import List
from typing import Match
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar

import onigurumacffi

from babi.user_data import xdg_data

if TYPE_CHECKING:
    from typing import Protocol  # python3.8+
else:
    Protocol = object

A_ITALIC = getattr(curses, 'A_ITALIC', 0x80000000)  # new in py37

# yes I know this is wrong, but it's good enough for now
UN_COMMENT = re.compile(r'^\s*//.*$', re.MULTILINE)

compile_regset = functools.lru_cache()(onigurumacffi.compile_regset)

T = TypeVar('T')
TKey = TypeVar('TKey')
TValue = TypeVar('TValue')
Scope = Tuple[str, ...]


class Color(NamedTuple):
    r: int
    g: int
    b: int

    @classmethod
    def parse(cls, s: str) -> 'Color':
        return cls(r=int(s[1:3], 16), g=int(s[3:5], 16), b=int(s[5:7], 16))

    def as_curses(self) -> Tuple[int, int, int]:
        return (
            int(1000 * self.r / 255),
            int(1000 * self.g / 255),
            int(1000 * self.b / 255),
        )


class Style(NamedTuple):
    fg: Color
    bg: Color
    b: bool
    i: bool
    u: bool


class Selector(NamedTuple):
    # TODO: parts: Tuple[str, ...]
    s: str

    @classmethod
    def parse(cls, s: str) -> 'Selector':
        return cls(s)

    def matches(self, scope: Scope) -> Tuple[bool, int]:
        s = scope[-1]
        if self.s == s or s.startswith(f'{self.s}.'):
            return (True, self.s.count('.'))
        else:
            return (False, -1)


DEFAULT_SELECTOR = Selector.parse('')


def _select(
        scope: Scope,
        rules: Tuple[Tuple[Selector, T], ...],
        default: T,
) -> T:
    for scope_len in range(len(scope), 0, -1):
        sub_scope = scope[:scope_len]
        matches = []
        for selector, t in rules:
            is_matched, priority = selector.matches(sub_scope)
            if is_matched:
                matches.append((priority, t))
        if matches:  # TODO: and len(matches) == 1
            _, ret = max(matches)
            return ret

    return default


class Theme(NamedTuple):
    default: Style
    fg_rules: Tuple[Tuple[Selector, Color], ...]
    bg_rules: Tuple[Tuple[Selector, Color], ...]
    b_rules: Tuple[Tuple[Selector, bool], ...]
    i_rules: Tuple[Tuple[Selector, bool], ...]
    u_rules: Tuple[Tuple[Selector, bool], ...]

    @classmethod
    def parse(cls, filename: str) -> 'Theme':
        with open(filename) as f:
            contents = UN_COMMENT.sub('', f.read())
            data = json.loads(contents)

        fg_d = {DEFAULT_SELECTOR: Color(0xff, 0xff, 0xff)}
        bg_d = {DEFAULT_SELECTOR: Color(0x00, 0x00, 0x00)}
        b_d = {DEFAULT_SELECTOR: False}
        i_d = {DEFAULT_SELECTOR: False}
        u_d = {DEFAULT_SELECTOR: False}

        for k in ('foreground', 'editor.foreground'):
            if k in data['colors']:
                fg_d[DEFAULT_SELECTOR] = Color.parse(data['colors'][k])
                break

        for k in ('background', 'editor.background'):
            if k in data['colors']:
                bg_d[DEFAULT_SELECTOR] = Color.parse(data['colors'][k])
                break

        for theme_item in data['tokenColors']:
            if 'scope' not in theme_item:
                scopes = ['']  # some sort of default scope?
            elif isinstance(theme_item['scope'], str):
                scopes = [
                    s.strip() for s in theme_item['scope'].split(',')
                    # some themes have a trailing comma -- do they
                    # intentionally mean to match that? is it a bug? should I
                    # send a patch?
                    if s.strip()
                ]
            else:
                scopes = theme_item['scope']

            for scope in scopes:
                selector = Selector.parse(scope)
                if 'foreground' in theme_item['settings']:
                    color = Color.parse(theme_item['settings']['foreground'])
                    fg_d[selector] = color
                if 'background' in theme_item['settings']:
                    color = Color.parse(theme_item['settings']['background'])
                    bg_d[selector] = color
                if theme_item['settings'].get('fontStyle') == 'bold':
                    b_d[selector] = True
                elif theme_item['settings'].get('fontStyle') == 'italic':
                    i_d[selector] = True
                elif theme_item['settings'].get('fontStyle') == 'underline':
                    u_d[selector] = True

        return cls(
            default=Style(
                fg=fg_d.pop(DEFAULT_SELECTOR),
                bg=bg_d.pop(DEFAULT_SELECTOR),
                b=b_d.pop(DEFAULT_SELECTOR),
                i=i_d.pop(DEFAULT_SELECTOR),
                u=u_d.pop(DEFAULT_SELECTOR),
            ),
            fg_rules=tuple(fg_d.items()),
            bg_rules=tuple(bg_d.items()),
            b_rules=tuple(b_d.items()),
            i_rules=tuple(i_d.items()),
            u_rules=tuple(u_d.items()),
        )

    @classmethod
    def blank(cls) -> 'Theme':
        return cls(
            default=Style(
                fg=Color(0xff, 0xff, 0xff), bg=Color(0x2d, 0x09, 0x22),
                b=False, i=False, u=False,
            ),
            fg_rules=(), bg_rules=(), b_rules=(), i_rules=(), u_rules=(),
        )

    @functools.lru_cache(maxsize=None)
    def select(self, scope: Scope) -> Style:
        return Style(
            fg=_select(scope, self.fg_rules, self.default.fg),
            bg=_select(scope, self.bg_rules, self.default.bg),
            b=_select(scope, self.b_rules, self.default.b),
            i=_select(scope, self.i_rules, self.default.i),
            u=_select(scope, self.u_rules, self.default.u),
        )

    @functools.lru_cache(maxsize=None)
    def color_mappings(
        self,
    ) -> Tuple[Dict[Color, int], Dict[Tuple[Color, Color], int]]:
        assert curses.can_change_color()

        all_bgs = {self.default.bg}.union(dict(self.bg_rules).values())
        all_fgs = {self.default.fg}.union(dict(self.fg_rules).values())

        colors = {self.default.bg: 0, self.default.fg: 7}

        def _color_id() -> Generator[int, None, None]:
            """need to skip already assigned colors"""
            skip = frozenset(colors.values())
            i = 0
            while True:
                i += 1
                if i not in skip:
                    yield i

        colors.update({
            color: i
            for i, color in zip(_color_id(), all_bgs | all_fgs)
            if color not in colors
        })

        ret = {(self.default.bg, self.default.fg): 0}
        all_combinations = set(itertools.product(all_bgs, all_fgs))
        all_combinations.discard((self.default.bg, self.default.fg))
        ret.update({
            (bg, fg): i for i, (bg, fg) in enumerate(all_combinations, 1)
        })

        return colors, ret

    def attr(self, style: Style) -> int:
        _, pairs = self.color_mappings()
        return (
            curses.color_pair(pairs[(style.bg, style.fg)]) |
            curses.A_BOLD * style.b |
            A_ITALIC * style.i |
            curses.A_UNDERLINE * style.u
        )


Captures = Tuple[Tuple[int, '_Rule'], ...]


class _Rule(Protocol):
    """hax for recursive types python/mypy#731"""
    @property
    def name(self) -> Optional[str]: ...
    @property
    def match(self) -> Optional[str]: ...
    @property
    def begin(self) -> Optional[str]: ...
    @property
    def end(self) -> Optional[str]: ...
    @property
    def content_name(self) -> Optional[str]: ...
    @property
    def captures(self) -> Captures: ...
    @property
    def begin_captures(self) -> Captures: ...
    @property
    def end_captures(self) -> Captures: ...
    @property
    def include(self) -> Optional[str]: ...
    @property
    def patterns(self) -> 'Tuple[_Rule, ...]': ...


class Rule(NamedTuple):
    name: Optional[str]
    match: Optional[str]
    begin: Optional[str]
    end: Optional[str]
    content_name: Optional[str]
    captures: Captures
    begin_captures: Captures
    end_captures: Captures
    include: Optional[str]
    patterns: Tuple[_Rule, ...]

    @classmethod
    def from_dct(cls, dct: Dict[str, Any]) -> _Rule:
        name = dct.get('name')
        match = dct.get('match')
        begin = dct.get('begin')
        end = dct.get('end')
        content_name = dct.get('contentName')

        if 'captures' in dct:
            captures = tuple(
                (int(k), Rule.from_dct(v))
                for k, v in dct['captures'].items()
            )
        else:
            captures = ()

        if 'beginCaptures' in dct:
            begin_captures = tuple(
                (int(k), Rule.from_dct(v))
                for k, v in dct['beginCaptures'].items()
            )
        else:
            begin_captures = ()

        if 'endCaptures' in dct:
            end_captures = tuple(
                (int(k), Rule.from_dct(v))
                for k, v in dct['endCaptures'].items()
            )
        else:
            end_captures = ()

        # Using the captures key for a begin/end rule is short-hand for
        # giving both beginCaptures and endCaptures with same values
        if begin and captures:
            end_captures = begin_captures = captures
            captures = ()

        include = dct.get('include')

        if 'patterns' in dct:
            patterns = tuple(Rule.from_dct(d) for d in dct['patterns'])
        else:
            patterns = ()

        return cls(
            name=name,
            match=match,
            begin=begin,
            end=end,
            content_name=content_name,
            captures=captures,
            begin_captures=begin_captures,
            end_captures=end_captures,
            include=include,
            patterns=patterns,
        )


class FDict(Generic[TKey, TValue]):
    def __init__(self, dct: Dict[TKey, TValue]) -> None:
        self._hash = hash(tuple(sorted(dct.items())))
        self._dct = dct

    def __hash__(self) -> int:
        return self._hash

    def __getitem__(self, k: TKey) -> TValue:
        return self._dct[k]


class Grammar(NamedTuple):
    scope_name: str
    first_line_match: Optional[onigurumacffi._Pattern]
    file_types: FrozenSet[str]
    patterns: Tuple[_Rule, ...]
    repository: FDict[str, _Rule]

    @classmethod
    def parse(cls, filename: str) -> 'Grammar':
        with open(filename, 'rb') as f:
            # https://github.com/python/typeshed/pull/3738
            data = plistlib.load(f)  # type: ignore

        scope_name = data['scopeName']
        if 'firstLineMatch' in data:
            first_line_match = onigurumacffi.compile(data['firstLineMatch'])
        else:
            first_line_match = None
        if 'fileTypes' in data:
            file_types = frozenset(data['fileTypes'])
        else:
            file_types = frozenset()
        patterns = tuple(Rule.from_dct(dct) for dct in data['patterns'])
        if 'repository' in data:
            repository = FDict({
                k: Rule.from_dct(dct) for k, dct in data['repository'].items()
            })
        else:
            repository = FDict({})
        return cls(
            scope_name=scope_name,
            first_line_match=first_line_match,
            file_types=file_types,
            patterns=patterns,
            repository=repository,
        )

    @classmethod
    def blank(cls) -> 'Grammar':
        return cls(
            scope_name='source.unknown',
            first_line_match=None,
            file_types=frozenset(),
            patterns=(),
            repository=FDict({}),
        )

    def matches_file(self, filename: str, first_line: str) -> bool:
        _, ext = os.path.splitext(filename)
        if ext.lstrip('.') in self.file_types:
            return True
        elif self.first_line_match is not None and os.path.exists(filename):
            with open(filename) as f:
                first_line = f.readline()
            return bool(self.first_line_match.match(first_line))
        else:
            return False


class Region(NamedTuple):
    start: int
    end: int
    scope: Scope


class CursesRegion(NamedTuple):
    x: int
    n: int
    color: int

    @property
    def end(self) -> int:
        return self.x + self.n


Regions = Tuple[Region, ...]
CursesRegions = Tuple[CursesRegion, ...]
State = Tuple['_Entry', ...]
StyleCB = Callable[[Match[str], State], Tuple[State, Regions]]


class _Entry(Protocol):
    """hax for recursive types python/mypy#731"""
    @property
    def grammar(self) -> Grammar: ...
    @property
    def regset(self) -> onigurumacffi._RegSet: ...
    @property
    def callbacks(self) -> Tuple[StyleCB, ...]: ...
    @property
    def scope(self) -> Scope: ...


class Entry(NamedTuple):
    grammar: Grammar
    regset: onigurumacffi._RegSet
    callbacks: Tuple[StyleCB, ...]
    scope: Scope


@functools.lru_cache(maxsize=None)
def highlight_line(state: State, line: str) -> Tuple[State, Regions]:
    ret = []
    pos = 0
    while pos < len(line):
        entry = state[-1]

        idx, match = entry.regset.search(line, pos)
        if match is not None:
            if match.start() > pos:
                ret.append(Region(pos, match.start(), entry.scope))

            state, regions = entry.callbacks[idx](match, state)
            ret.extend(regions)

            pos = match.end()
        else:
            ret.append(Region(pos, len(line), entry.scope))
            pos = len(line)

    return state, tuple(ret)


def _expand_captures(
        scope: Scope,
        match: Match[str],
        captures: Captures,
) -> Regions:
    ret: List[Region] = []
    pos, pos_end = match.span()
    for i, rule in captures:
        try:
            group_s = match[i]
        except IndexError:  # some grammars are malformed here?
            continue
        if not group_s:
            continue

        start, end = match.span(i)
        if start < pos:
            # TODO: could maybe bisect but this is probably fast enough
            j = len(ret) - 1
            while j > 0 and start < ret[j - 1].end:
                j -= 1

            oldtok = ret[j]
            newtok = []
            if start > oldtok.start:
                newtok.append(oldtok._replace(end=start))

            # TODO: this is duplicated below
            if not rule.match and not rule.begin and not rule.include:
                assert rule.name is not None
                newtok.append(Region(start, end, (*oldtok.scope, rule.name)))
            else:
                raise NotImplementedError('complex capture rule')

            if end < oldtok.end:
                newtok.append(oldtok._replace(start=end))
            ret[j:j + 1] = newtok
        else:
            if start > pos:
                ret.append(Region(pos, start, scope))

            if not rule.match and not rule.begin and not rule.include:
                assert rule.name is not None
                ret.append(Region(start, end, (*scope, rule.name)))
            else:
                raise NotImplementedError('complex capture rule')

            pos = end

    if pos < pos_end:
        ret.append(Region(pos, pos_end, scope))
    return tuple(ret)


def _end_cb(
        match: Match[str],
        state: State,
        *,
        end_captures: Captures,
) -> Tuple[State, Regions]:
    return state[:-1], _expand_captures(state[-1].scope, match, end_captures)


def _match_cb(
        match: Match[str],
        state: State,
        *,
        rule: _Rule,
) -> Tuple[State, Regions]:
    if rule.name is not None:
        scope = (*state[-1].scope, rule.name)
    else:
        scope = state[-1].scope
    return state, _expand_captures(scope, match, rule.captures)


def _begin_cb(
        match: Match[str],
        state: State,
        *,
        rule: _Rule,
) -> Tuple[State, Regions]:
    assert rule.end is not None
    prev_entry = state[-1]

    if rule.name is not None:
        scope = (*prev_entry.scope, rule.name)
    else:
        scope = prev_entry.scope
    if rule.content_name is not None:
        next_scopes = (*scope, rule.content_name)
    else:
        next_scopes = scope

    end = match.expand(rule.end)
    entry = make_entry(
        prev_entry.grammar, rule.patterns, end, rule.end_captures, next_scopes,
    )

    return (*state, entry), _expand_captures(scope, match, rule.begin_captures)


@functools.lru_cache(maxsize=None)
def _regs_cbs(
        grammar: Grammar,
        rules: Tuple[_Rule, ...],
) -> Tuple[Tuple[str, ...], Tuple[StyleCB, ...]]:
    regs = []
    cbs: List[StyleCB] = []

    rules_stack = list(reversed(rules))
    while rules_stack:
        rule = rules_stack.pop()

        # XXX: can a rule have an include also?
        if rule.include is not None:
            assert rule.match is None
            assert rule.begin is None
            if rule.include == '$self':
                rules_stack.extend(reversed(grammar.patterns))
                continue
            else:
                rule = grammar.repository[rule.include[1:]]

        if rule.match is None and rule.begin is None and rule.patterns:
            rules_stack.extend(reversed(rule.patterns))
        elif rule.match is not None:
            regs.append(rule.match)
            cbs.append(functools.partial(_match_cb, rule=rule))
        elif rule.begin is not None:
            regs.append(rule.begin)
            cbs.append(functools.partial(_begin_cb, rule=rule))
        else:
            raise AssertionError(f'unreachable {rule}')

    return tuple(regs), tuple(cbs)


def make_entry(
        grammar: Grammar,
        patterns: Tuple[_Rule, ...],
        end: str,
        end_captures: Captures,
        scope: Scope,
) -> _Entry:
    regs, cbs = _regs_cbs(grammar, patterns)
    end_cb = functools.partial(_end_cb, end_captures=end_captures)
    return Entry(grammar, compile_regset(end, *regs), (end_cb, *cbs), scope)


def load_grammars() -> Dict[str, Grammar]:
    ret = {'source.unknown': Grammar.blank()}

    syntax_dir = xdg_data('textmate_syntax')
    if os.path.exists(syntax_dir):
        for filename in os.listdir(syntax_dir):
            grammar = Grammar.parse(os.path.join(syntax_dir, filename))
            ret[grammar.scope_name] = grammar

    return ret
