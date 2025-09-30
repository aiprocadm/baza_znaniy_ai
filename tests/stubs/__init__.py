"""Local stub packages used exclusively for the test suite.

The tests configure ``sys.path`` to import these modules before checking for
real third-party dependencies. This ensures that running the real
application picks up the genuine packages when they are installed while the
unit tests remain hermetic.
"""
