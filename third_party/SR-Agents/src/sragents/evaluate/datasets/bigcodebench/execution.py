"""BigCodeBench execution in local subprocesses with resource and time limits.

Ported from:
  - https://github.com/bigcode-project/bigcodebench
  (``bigcodebench/eval/__init__.py``: unsafe_execute, untrusted_check)
  - https://github.com/bigcode-project/bigcodebench
  (``bigcodebench/eval/utils.py``: safe_environment, reliability_guard, swallow_io, ...)
"""

import contextlib
import ast
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from multiprocessing import Value
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMEOUT_LIMIT = 240.0

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3
_ENVIRONMENT_ERROR = 4

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


# ---------------------------------------------------------------------------
# Safety utilities (ported from eval/utils.py)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def swallow_subprocess_output():
    """Context manager to swallow stdout and stderr for subprocesses."""
    original_popen = subprocess.Popen
    original_run = subprocess.run

    def _popen_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_popen(*args, **kwargs)

    def _run_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_run(*args, **kwargs)

    subprocess.Popen = _popen_patch
    subprocess.run = _run_patch
    try:
        yield
    finally:
        subprocess.Popen = original_popen
        subprocess.run = original_run


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                with swallow_subprocess_output():
                    yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def safe_environment():
    # Save original functions
    original_kill = os.kill
    original_killpg = os.killpg
    original_system = os.system
    original_subprocess_call = subprocess.call
    original_subprocess_check_output = subprocess.check_output
    original_subprocess_run = subprocess.run
    original_subprocess_popen = subprocess.Popen
    original_os_popen = os.popen
    original_os_execv = os.execv
    original_os_execvp = os.execvp
    original_os_execvpe = os.execvpe

    current_pid = os.getpid()
    current_pgid = os.getpgid(current_pid)
    child_pids = []

    def safe_kill(pid, sig):
        try:
            pgid = os.getpgid(pid)
            if pid == current_pid or pid in child_pids:
                original_kill(pid, sig)
            else:
                print(f"Prevented attempt to kill PID {pid} with signal {sig}")
        except ProcessLookupError:
            pass

    def safe_killpg(pgid, sig):
        if pgid == current_pgid or pgid in {os.getpgid(pid) for pid in child_pids}:
            original_killpg(pgid, sig)
        else:
            print(f"Prevented attempt to kill PGID {pgid} with signal {sig}")

    def safe_system(command):
        print(f"Intercepted system command: {command}")
        if 'kill' in command or 'killall' in command:
            return 0
        return original_system(command)

    def safe_subprocess_call(command, *args, **kwargs):
        print(f"Intercepted subprocess call: {command}")
        if 'kill' in command or 'killall' in command:
            return 0
        return original_subprocess_call(command, *args, **kwargs)

    def safe_subprocess_check_output(command, *args, **kwargs):
        print(f"Intercepted command: {command}")
        if 'ps' in command:
            return b""
        return original_subprocess_check_output(command, *args, **kwargs)

    def safe_subprocess_run(*args, **kwargs):
        print(f"Intercepted subprocess run command: {args}")
        if 'kill' in args[0] or 'killall' in args[0]:
            return subprocess.CompletedProcess(args, 0, b'', b'')
        return original_subprocess_run(*args, **kwargs)

    class SafePopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            print(f"Intercepted Popen command: {args}")
            kwargs['preexec_fn'] = os.setsid
            super().__init__(*args, **kwargs)
            child_pids.append(self.pid)

        def communicate(self, *args, **kwargs):
            try:
                return super().communicate(*args, **kwargs)
            except subprocess.TimeoutExpired:
                print("Timeout expired, intercepted and returning None")
                return None, None

        def kill(self):
            print(f"Intercepted kill call for PID {self.pid}")
            safe_kill(self.pid, signal.SIGTERM)

        def terminate(self):
            print(f"Intercepted terminate call for PID {self.pid}")
            safe_kill(self.pid, signal.SIGTERM)

    def safe_os_popen(command):
        print(f"Intercepted os.popen command: {command}")
        if 'kill' in command or 'killall' in command:
            return os.popen('echo Intercepted')
        return original_os_popen(command)

    def safe_exec(*args, **kwargs):
        print(f"Intercepted exec command: {args}")

    # Override the risky functions with the safe versions
    os.kill = safe_kill
    os.killpg = safe_killpg
    os.system = safe_system
    subprocess.call = safe_subprocess_call
    subprocess.check_output = safe_subprocess_check_output
    subprocess.run = safe_subprocess_run
    subprocess.Popen = SafePopen
    os.popen = safe_os_popen
    os.execv = safe_exec
    os.execvp = safe_exec
    os.execvpe = safe_exec

    try:
        yield
    finally:
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(10):
                    time.sleep(0.1)
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                else:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"Error handling process {pid}: {e}")

        os.kill = original_kill
        os.killpg = original_killpg
        os.system = original_system
        subprocess.call = original_subprocess_call
        subprocess.check_output = original_subprocess_check_output
        subprocess.run = original_subprocess_run
        subprocess.Popen = original_subprocess_popen
        os.popen = original_os_popen
        os.execv = original_os_execv
        os.execvp = original_os_execvp
        os.execvpe = original_os_execvpe


def reliability_guard(max_as_limit, max_data_limit, max_stack_limit):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    import os
    import time
    from datetime import datetime

    os.environ['TZ'] = 'UTC'
    time.tzset()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

    if max_as_limit and max_data_limit and max_stack_limit:
        import resource

        max_as_limit = max_as_limit * 1024 * 1024
        max_data_limit = max_data_limit * 1024 * 1024
        max_stack_limit = max_stack_limit * 1024 * 1024

        # A finite RLIMIT_AS makes TensorFlow's pthread_create return EAGAIN
        # (tf_data_private_threadpool) and abort the process. The reference
        # solution hits the same abort, so the whole BigCodeBench run stops.
        # RLIMIT_DATA still caps the data segment.
        resource.setrlimit(
            resource.RLIMIT_DATA, (max_data_limit, max_data_limit)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (max_stack_limit, max_stack_limit)
            )

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import matplotlib.pyplot as plt
    plt.close('all')


# ---------------------------------------------------------------------------
# Core execution (ported from eval/__init__.py)
# ---------------------------------------------------------------------------

class EvaluationEnvironmentError(RuntimeError):
    """Infrastructure failure: must never be aggregated as a candidate score."""


def _candidate_symbol_import_error(exc, code):
    """An invalid from-import in candidate code is an answer failure.

    Missing packages, imports in benchmark tests and failures inside installed
    packages still propagate as environment errors. Inspect the originating
    frame so a candidate importing a broken library is not blamed for setup.
    """
    if not isinstance(exc, ImportError) or isinstance(exc, ModuleNotFoundError):
        return False
    tb = exc.__traceback__
    if tb is None:
        return False
    while tb.tb_next is not None:
        tb = tb.tb_next
    if tb.tb_frame.f_code.co_filename != '__test__.py':
        return False
    try:
        nodes = ast.walk(ast.parse(code))
    except SyntaxError:
        return False
    for node in nodes:
        if (isinstance(node, ast.ImportFrom) and node.level == 0
                and node.lineno <= tb.tb_lineno <= node.end_lineno
                and node.module == exc.name):
            module = sys.modules.get(node.module)
            if module is not None:
                return any(alias.name != '*' and not hasattr(module, alias.name)
                           for alias in node.names)
    return False


class _TestResult(unittest.TestResult):
    def __init__(self, candidate_code=''):
        super().__init__()
        self.candidate_code = candidate_code
        self.environment_errors = []
        self.timed_out = False

    def addError(self, test, err):
        if isinstance(err[1], ImportError) and not _candidate_symbol_import_error(err[1], self.candidate_code):
            self.environment_errors.append(f'{type(err[1]).__name__}: {err[1]}')
        if isinstance(err[1], TimeoutException):
            self.timed_out = True
        super().addError(test, err)


def unsafe_execute(entry_point, code, test_code, timeout, max_as_limit,
                   max_data_limit, max_stack_limit, stat, phase):
    with safe_environment(), create_tempdir():
        import shutil
        import builtins
        rmtree, rmdir, change_dir = shutil.rmtree, os.rmdir, os.chdir
        # Setup exceptions deliberately propagate to the process supervisor.
        reliability_guard(max_as_limit, max_data_limit, max_stack_limit)
        module_name = '__test__'
        new_module = types.ModuleType(module_name)
        new_module.__dict__.update({'__builtins__': builtins, '__file__': '__test__.py',
            '__package__': None, '__doc__': None, 'sys': sys, 'os': os, 'environ': os.environ})
        try:
            phase.value = 1  # candidate execution has actually started
            with swallow_io(), time_limit(timeout):
                exec(compile(code + '\n' + test_code, '__test__.py', 'exec'), new_module.__dict__)
                sys.modules[module_name] = new_module
                suite = unittest.TestLoader().loadTestsFromTestCase(getattr(new_module, 'TestCases'))
                result = _TestResult(code)
                suite.run(result)
            if result.environment_errors:
                raise EvaluationEnvironmentError('; '.join(result.environment_errors))
            if result.timed_out:
                stat.value = _TIMEOUT
            else:
                stat.value = _FAILED if result.failures or result.errors else _SUCCESS
        except (ImportError, EvaluationEnvironmentError) as exc:
            if _candidate_symbol_import_error(exc, code):
                stat.value = _FAILED
                return
            # Keep unresolved dependency/setup failures out of accuracy.
            raise EvaluationEnvironmentError(f'{type(exc).__name__}: {exc}') from exc
        except TimeoutException:
            stat.value = _TIMEOUT
        except BaseException:
            stat.value = _FAILED
        finally:
            shutil.rmtree, os.rmdir, os.chdir = rmtree, rmdir, change_dir


def _isolated_execute(arguments, stat, phase, error):
    try:
        os.setsid()
        unsafe_execute(*arguments, stat, phase)
    except BaseException as exc:
        error.value = f'{type(exc).__name__}: {exc}'.encode('utf-8')[:4095]
        stat.value = _ENVIRONMENT_ERROR


def _process_check(code, test_code, entry_point, max_as_limit, max_data_limit,
                   max_stack_limit, min_time_limit=10, gt_time_limit=60) -> Tuple[str, Dict]:
    timeout = max(float(os.getenv('BIGCODEBENCH_TIMEOUT_PER_TASK', TIMEOUT_LIMIT)),
                  min_time_limit, gt_time_limit) + 1
    context = multiprocessing.get_context('spawn')
    stat, phase = context.Value('i', _UNKNOWN), context.Value('i', 0)
    error = context.Array('c', 4096, lock=False)
    arguments = (entry_point, code, test_code, timeout,
                 max_as_limit, max_data_limit, max_stack_limit)
    process = context.Process(target=_isolated_execute, args=(arguments, stat, phase, error))
    try:
        process.start()
        process.join(timeout=timeout + 1)
        timed_out = process.is_alive()
        if timed_out:
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        if stat.value == _ENVIRONMENT_ERROR:
            raise EvaluationEnvironmentError(error.value.decode('utf-8', errors='replace'))
        if timed_out:
            if phase.value == 0:
                raise EvaluationEnvironmentError('Evaluator initialization timed out')
            return TIMEOUT, {}
        if process.exitcode != 0 or stat.value == _UNKNOWN:
            raise EvaluationEnvironmentError(f'Evaluator exited without a result (exit={process.exitcode})')
        return _mapping[stat.value], {}
    except EvaluationEnvironmentError:
        raise
    except Exception as exc:
        raise EvaluationEnvironmentError(f'Evaluator process failure: {exc}') from exc
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            process.close()


def untrusted_check(code, test_code, entry_point, max_as_limit, max_data_limit,
                    max_stack_limit, min_time_limit=10, gt_time_limit=60):
    return _process_check(code, test_code, entry_point, max_as_limit, max_data_limit,
                          max_stack_limit, min_time_limit, gt_time_limit)
