import logging
import time

import pytest
from src.utils.timer import timer, timer_decorator


class TestTimerContextManager:
    def test_timer_basic(self, capfd):
        with timer("test_operation"):
            time.sleep(0.1)

        captured = capfd.readouterr()
        assert "[test_operation] start" in captured.out
        assert "[test_operation] done in" in captured.out
        assert " s" in captured.out

    def test_timer_with_logging(self, caplog):
        with caplog.at_level(logging.INFO):
            with timer("test_with_log", log=True):
                time.sleep(0.05)

        assert "[test_with_log] start" in caplog.text
        assert "[test_with_log] done in" in caplog.text

    def test_timer_without_logging(self, caplog):
        with caplog.at_level(logging.INFO):
            with timer("test_no_log", log=False):
                time.sleep(0.05)

        assert "[test_no_log]" not in caplog.text

    def test_timer_exception_handling(self, capfd):
        with pytest.raises(ValueError):
            with timer("test_exception"):
                raise ValueError("Test exception")

        captured = capfd.readouterr()
        assert "[test_exception] start" in captured.out
        # Context manager doesn't print "done" message on exception


class TestTimerDecorator:
    def test_decorator_with_name(self, capfd):
        @timer_decorator("custom_name")
        def test_func():
            time.sleep(0.1)
            return "result"

        result = test_func()

        captured = capfd.readouterr()
        assert "[custom_name] start" in captured.out
        assert "[custom_name] done in" in captured.out
        assert result == "result"

    def test_decorator_without_name(self, capfd):
        @timer_decorator()
        def test_func():
            time.sleep(0.05)
            return 42

        result = test_func()

        captured = capfd.readouterr()
        assert "[test_func] start" in captured.out
        assert "[test_func] done in" in captured.out
        assert result == 42

    def test_decorator_with_arguments(self, capfd):
        @timer_decorator("func_with_args")
        def add(a, b):
            return a + b

        result = add(2, 3)

        captured = capfd.readouterr()
        assert "[func_with_args] start" in captured.out
        assert "[func_with_args] done in" in captured.out
        assert result == 5

    def test_decorator_with_kwargs(self, capfd):
        @timer_decorator()
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = greet("Alice", greeting="Hi")

        captured = capfd.readouterr()
        assert "[greet] start" in captured.out
        assert "[greet] done in" in captured.out
        assert result == "Hi, Alice!"

    def test_decorator_with_logging(self, caplog):
        with caplog.at_level(logging.INFO):

            @timer_decorator("logged_func", log=True)
            def test_func():
                return "logged"

            test_func()

        assert "[logged_func] start" in caplog.text
        assert "[logged_func] done in" in caplog.text

    def test_decorator_without_logging(self, caplog):
        with caplog.at_level(logging.INFO):

            @timer_decorator("not_logged", log=False)
            def test_func():
                return "not logged"

            test_func()

        assert "[not_logged]" not in caplog.text

    def test_decorator_preserves_function_metadata(self):
        @timer_decorator()
        def documented_func():
            """This is a documented function."""
            return "doc"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a documented function."

    def test_decorator_exception_handling(self, capfd):
        @timer_decorator("exception_func")
        def raise_error():
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            raise_error()

        captured = capfd.readouterr()
        assert "[exception_func] start" in captured.out
        assert "[exception_func] done in" in captured.out
