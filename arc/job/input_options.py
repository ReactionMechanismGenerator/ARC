"""
This module reads the options an ESS input deck was written with, in the program's own syntax, and records them per
job in the scheduler's output dictionary.
"""

import re

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from arc.job.adapter import JobAdapter


ORCA_COORDINATE_INPUT_NAMES = ('xyz', 'int', 'gzmt')
ORCA_RESOURCE_BLOCK_NAMES = ('maxcore', 'pal')
ORCA_RESOURCE_KEYWORD_REGEX = re.compile(r'(?i)(?:(?<=!)|\s+)pal\d+(?=\s|$)')
QCHEM_RESOURCE_REM_VARIABLES = ('mem_total', 'mem_static', 'cc_memory', 'threads', 'nthreads')
MOLPRO_GEOMETRY_PREFIX = 'geometry='
MOLPRO_RESOURCE_CARD_PREFIXES = ('memory,',)
CFOUR_KEYWORD_LINE_PREFIXES = ('*CFOUR(', '*ACES2(', '*CRAPS(')
CFOUR_RESOURCE_KEYWORD = r'(?:MEMORY_SIZE|MEM_UNIT)[ \t]*=[ \t]*[^,)\s]*'
CFOUR_RESOURCE_KEYWORD_AFTER_A_COMMA_REGEX = re.compile(r'(?i)[ \t]*,\s*' + CFOUR_RESOURCE_KEYWORD)
CFOUR_RESOURCE_KEYWORD_BEFORE_A_COMMA_REGEX = re.compile(r'(?i)(?<![^(\s])[ \t]*' + CFOUR_RESOURCE_KEYWORD
                                                         + r'[ \t]*,?[ \t]*')
GAUSSIAN_GEOM_OPTION_REGEX = re.compile(r'(?i)(?<![\w-])geom(?:etry)?\s*(?:=\s*(\([^)]*\)|[^\s,()/]+)|(\([^)]*\)))')
GAUSSIAN_GEOM_ALLCHECK_OPTIONS = ('allcheck', 'allchk')
TERACHEM_FILE_AND_RESOURCE_DIRECTIVES = ('chkfile', 'jobname', 'coordinates', 'gpumem', 'gpus', 'scrdir')
XTB_GEOMETRY_FILE_EXTENSIONS = ('.sdf', '.xyz', '.mol', '.coord')
XTB_RESOURCE_FLAGS = ('--parallel', '-P')


def gaussian_route_sections(lines: Iterable[str], first_only: bool = False) -> list[str]:
    """
    Get the Gaussian route sections of an input deck or a log, each as a single line, in order.

    A route starts at a line whose first non-blank character is ``#`` and ends at the following blank line
    (an input deck) or rule of dashes (a log's echoed route). The first route is the first such line. Every later
    route must directly follow a rule of dashes, which is how a log opens each ``Link1`` step, so a ``#`` line
    wrapped out of an archive block is not read as a route. With ``first_only`` the lines are read only as far as
    the first route.

    A log wraps the route mid-token at a fixed column and indents every echoed line by one space; its lines are
    joined after dropping that single leading space and without inserting one, so ``integral=(grid=ultr`` +
    ``afine, ...`` rejoins as written. A deck's route lines are not wrapped, so they are stripped and joined by a
    single space.

    Args:
        lines (Iterable[str]): The lines of a deck or a log, e.g., an open file.
        first_only (bool, optional): Whether to stop at the first route.

    Returns:
        list[str]: The routes, empty when there is none.
    """
    routes: list[str] = list()
    collected: list[str] = list()
    in_route = False
    previous_was_rule = False

    def finish(ended_on_dash_rule: bool) -> None:
        if ended_on_dash_rule:
            routes.append(''.join(part[1:] if part.startswith(' ') else part for part in collected))
        else:
            routes.append(' '.join(part.strip() for part in collected))

    for line in lines:
        stripped = line.strip()
        is_rule = bool(stripped) and set(stripped) == {'-'}
        if not in_route:
            if stripped.startswith('#') and (not routes or previous_was_rule):
                in_route = True
                collected = [line.rstrip('\n')]
            previous_was_rule = is_rule
            continue
        if not stripped or is_rule:
            finish(is_rule)
            in_route = False
            previous_was_rule = is_rule
            if first_only:
                return routes
            continue
        collected.append(line.rstrip('\n'))
    if in_route:
        finish(False)
    return routes


def orca_input_sections(lines: Iterable[str]) -> list[tuple[str, list[str]]]:
    """
    Split the stripped lines of an ORCA input into its keyword lines and ``%`` blocks, in input order.

    Each keyword line (``!``) is a section named ``'!'`` and each ``$`` line (e.g., ``$new_job``) a section named
    ``'$'``. A ``%`` block is named by its first word, lowercased (``'scf'``, ``'maxcore'``), and holds every line
    from its ``%`` line up to the next line that begins with ``%``, ``!``, ``*`` or ``$``, so nested sub-blocks and
    their ``end`` lines are kept; trailing blank lines are dropped. The coordinate input (a ``* xyz``, ``* int`` or
    ``* gzmt`` block, up to and including its closing ``*``) is left out, as is any other ``*`` line (e.g.,
    ``* xyzfile``). A ``%coords`` block is kept under the name ``'coords'``. Lines outside every section are
    left out.

    Args:
        lines (Iterable[str]): The input lines, each stripped of surrounding whitespace.

    Returns:
        list[tuple[str, list[str]]]: The ``(name, lines)`` sections.
    """
    sections: list[tuple[str, list[str]]] = list()
    current: list[str] | None = None
    in_coordinates = False
    for line in lines:
        if in_coordinates:
            in_coordinates = not line.startswith('*')
        elif line.startswith('%'):
            sections.append((line[1:].split(None, 1)[0].lower() if line[1:].strip() else '', [line]))
            current = sections[-1][1]
        elif line.startswith(('!', '*', '$')):
            current = None
            if line.startswith(('!', '$')):
                sections.append((line[0], [line]))
            in_coordinates = line.startswith('*') \
                and line[1:].lower().split()[:1] in [[name] for name in ORCA_COORDINATE_INPUT_NAMES]
        elif current is not None:
            current.append(line)
    for _, section in sections:
        while section and not section[-1]:
            section.pop()
    return sections


def _gaussian_input_options(text: str) -> str | None:
    """
    The route section of a Gaussian input deck as one line (see ``gaussian_route_sections``), followed by every
    input section after the molecule specification (e.g., ModRedundant lines, a ``gen`` or ``genecp`` basis set and
    its ECP, ``scrf=read`` parameters), each preceded by an empty line as sections are separated in the deck.
    Link 0 (``%``) lines, the title section and the molecule specification (charge, multiplicity and coordinates)
    are left out; with ``geom=allcheck`` (or ``geom=allchk``, also inside ``geom=(...)``, read from the route without
    its ``!`` comments) there is no title or molecule section to leave out. Only the first job of a ``--Link1--``
    deck is read.
    """
    sections, current, route_index = list(), list(), None
    for line in (line.strip() for line in text.splitlines()):
        if line.lower() == '--link1--':
            break
        if line and not (route_index is None and not current and line.startswith('%')):
            if route_index is None and line.startswith('#'):
                route_index = len(sections)
            current.append(line)
        elif not line and current:
            sections.append(current)
            current = list()
    if current:
        sections.append(current)
    routes = gaussian_route_sections(text.splitlines(), first_only=True)
    if route_index is None or not routes:
        return None
    skipped = 0 if _gaussian_route_reads_all_from_checkpoint(sections[route_index]) else 2
    trailing = sections[route_index + 1 + skipped:]
    return '\n\n'.join([routes[0]] + ['\n'.join(section) for section in trailing])


def _gaussian_route_reads_all_from_checkpoint(route_lines: list[str]) -> bool:
    """
    Whether the route's ``geom`` keyword carries ``allcheck`` or ``allchk``, given as ``geom=allcheck``,
    ``geom=(allcheck, ...)`` or ``geom(allcheck)``. Each route line is read up to its ``!`` comment.
    """
    route = ' '.join(line.split('!', 1)[0] for line in route_lines)
    for match in GAUSSIAN_GEOM_OPTION_REGEX.finditer(route):
        value = match.group(1) or match.group(2) or ''
        options = re.split(r'[,\s]+', value.strip('()').lower())
        if any(option in GAUSSIAN_GEOM_ALLCHECK_OPTIONS for option in options):
            return True
    return False


def _orca_input_options(text: str) -> str | None:
    """
    The keyword lines, ``$`` lines and ``%`` blocks of an ORCA input deck, in input order, one line per line.
    The coordinate input (including ``%coords``), the resource blocks ``%maxcore`` and ``%pal`` and the ``PALn``
    keywords are left out; a keyword line left holding nothing else is dropped.
    """
    sections = orca_input_sections(line.strip() for line in text.splitlines())
    lines = list()
    for name, section in sections:
        if name in ('coords',) + ORCA_RESOURCE_BLOCK_NAMES:
            continue
        for line in section:
            if name == '!':
                line = ORCA_RESOURCE_KEYWORD_REGEX.sub('', line).rstrip()
            if line and line != '!':
                lines.append(line)
    return '\n'.join(lines) or None


def _qchem_input_options(text: str) -> str | None:
    """
    Every non-blank line of a Q-Chem input deck except its ``$molecule`` section and the memory and thread variables
    of ``$rem`` (``MEM_TOTAL``, ``MEM_STATIC``, ``CC_MEMORY``, ``THREADS``, ``NTHREADS``).
    """
    kept, in_molecule = list(), False
    for line in (line.strip() for line in text.splitlines()):
        if in_molecule:
            in_molecule = line.lower() != '$end'
        elif line.lower() == '$molecule':
            in_molecule = True
        elif line and re.split(r'[\s=]+', line.lower(), maxsplit=1)[0] not in QCHEM_RESOURCE_REM_VARIABLES:
            kept.append(line)
    return '\n'.join(kept) or None


def _molpro_input_options(text: str) -> str | None:
    """
    Every non-blank line of a Molpro input deck except its ``***`` title line, its ``memory`` card, its ``geometry``
    block and its ``---`` end line.
    """
    kept, in_geometry = list(), False
    for line in (line.strip() for line in text.splitlines()):
        if in_geometry:
            in_geometry = '}' not in line
        elif line.lower().startswith(MOLPRO_GEOMETRY_PREFIX):
            in_geometry = '}' not in line
        elif line and not line.startswith(('***', '---')) \
                and not line.lower().startswith(MOLPRO_RESOURCE_CARD_PREFIXES):
            kept.append(line)
    return '\n'.join(kept) or None


def _cfour_input_options(text: str) -> str | None:
    """
    Every non-blank line of a CFour ``ZMAT`` from its ``*CFOUR(`` (or ``*ACES2(``, ``*CRAPS(``) keyword line on,
    without the memory keywords ``MEMORY_SIZE`` and ``MEM_UNIT``. A memory keyword is removed with the comma that
    precedes it, so the separator that followed it (a comma, a line break or the closing parenthesis) now follows
    the keyword before it; a memory keyword with no comma before it (the first of the list or of a line) is removed
    with the comma that follows it.
    """
    lines = [line.strip() for line in text.splitlines()]
    start = next((i for i, line in enumerate(lines) if line.upper().startswith(CFOUR_KEYWORD_LINE_PREFIXES)), None)
    if start is None:
        return None
    options = CFOUR_RESOURCE_KEYWORD_AFTER_A_COMMA_REGEX.sub('', '\n'.join(lines[start:]))
    options = CFOUR_RESOURCE_KEYWORD_BEFORE_A_COMMA_REGEX.sub('', options)
    return '\n'.join(line.strip() for line in options.splitlines() if line.strip()) or None


def _terachem_input_options(text: str) -> str | None:
    """
    Every non-blank line of a TeraChem input deck except comments, the lone ``end`` line, and the file and resource
    directives ``chkfile``, ``jobname``, ``coordinates``, ``gpumem``, ``gpus`` and ``scrdir``.
    """
    kept = list()
    for line in (line.strip() for line in text.splitlines()):
        if not line or line.startswith('#') or line.lower() == 'end' \
                or line.split()[0].lower() in TERACHEM_FILE_AND_RESOURCE_DIRECTIVES:
            continue
        kept.append(line)
    return '\n'.join(kept) or None


def _xtb_input_options(text: str) -> str | None:
    """
    The command-line arguments of an xTB command that follow its geometry file, up to the output redirection,
    without the parallelism flag (``--parallel`` or ``-P``, with its thread count when one follows), followed by the
    lines of the detailed input (the ``$`` blocks) when the text carries them after the command.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    tokens = lines[0].split('>', 1)[0].split()
    geometry = next((i for i, token in enumerate(tokens) if token.lower().endswith(XTB_GEOMETRY_FILE_EXTENSIONS)),
                    None)
    kept, index = list(), geometry + 1 if geometry is not None else len(tokens)
    while index < len(tokens):
        if tokens[index] in XTB_RESOURCE_FLAGS:
            index += 2 if index + 1 < len(tokens) and tokens[index + 1].isdigit() else 1
            continue
        kept.append(tokens[index])
        index += 1
    arguments = ' '.join(kept)
    return '\n'.join([arguments] + lines[1:] if arguments else lines[1:]) or None


INPUT_OPTIONS_READERS = {'cfour': _cfour_input_options,
                         'gaussian': _gaussian_input_options,
                         'molpro': _molpro_input_options,
                         'orca': _orca_input_options,
                         'qchem': _qchem_input_options,
                         'terachem': _terachem_input_options,
                         'xtb': _xtb_input_options,
                         }


def get_input_options(software: str | None, input_text: str | None) -> str | None:
    """
    Get the options of an ESS input as it was written, in the program's own syntax.

    Gaussian: the route section on one line, then every section after the molecule specification. ORCA: the ``!``
    keyword lines, ``$`` lines and ``%`` blocks, one line per line, without the coordinate input. Q-Chem: every
    section but ``$molecule``. Molpro: every command but the title, ``geometry`` block and end line. CFour: the
    ``*CFOUR(...)`` keyword list and what follows it. TeraChem: every directive but comments, ``end`` and the file
    directives. xTB: the command-line arguments after the geometry file, then the detailed input. Memory and
    parallelism settings are left out for every program (Gaussian's Link 0 lines, ORCA's ``%maxcore``, ``%pal`` and
    ``PALn``, Q-Chem's memory and thread ``$rem`` variables, Molpro's ``memory`` card, CFour's ``MEMORY_SIZE`` and
    ``MEM_UNIT``, TeraChem's ``gpumem`` and ``gpus``, xTB's ``--parallel`` and ``-P``). Lines are stripped of
    surrounding whitespace and blank lines are dropped, except the empty line that separates Gaussian sections;
    nothing else is changed.

    Args:
        software (str, optional): The program's ARC name, e.g., ``'gaussian'``.
        input_text (str, optional): The input as written.

    Returns:
        str | None: The options, or ``None`` for a program without a reader, an empty input, or an input with none.
    """
    reader = INPUT_OPTIONS_READERS.get(str(software).lower()) if software else None
    if reader is None or not input_text:
        return None
    return reader(input_text)


def capture_input_options(obj: 'JobAdapter', input_text: str) -> None:
    """
    Set ``obj.input_options`` from the input deck the job just wrote: ``{label: options}`` for its first species,
    see ``get_input_options``, or ``None`` when the job has no species.

    Args:
        obj (JobAdapter): The job adapter.
        input_text (str): The input deck as written.
    """
    obj.input_options = {obj.species[0].label: get_input_options(obj.job_adapter, input_text)} \
        if obj.species else None


def job_input_options(job: 'JobAdapter | None', label: str) -> str | None:
    """
    Get the options a job wrote into the input of a species, see ``capture_input_options``.

    Args:
        job (JobAdapter, optional): The job.
        label (str): The species label.

    Returns:
        str | None: The options, or ``None`` when the job recorded none for the species.
    """
    options = getattr(job, 'input_options', None)
    value = options.get(label) if isinstance(options, dict) else None
    return value if isinstance(value, str) and value else None


def set_recorded_input_options(entry: dict, job_key: str, input_options: str | None) -> None:
    """
    Record the input options of the job whose log is exported under ``job_key`` in a species' output entry
    (``Scheduler.output[label]['input_options'][job_key]``). ``None`` or an empty string removes the record.

    Args:
        entry (dict): The species' output entry, modified in place.
        job_key (str): 'opt', 'freq', 'sp', or 'composite'.
        input_options (str, optional): The options of the job.
    """
    recorded = entry.setdefault('input_options', dict())
    if isinstance(input_options, str) and input_options:
        recorded[job_key] = input_options
    else:
        recorded.pop(job_key, None)

