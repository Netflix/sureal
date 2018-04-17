rem Note: On windows, run this script to run tests

set PYTHONPATH=PYTHONPATH;python/src

python -m unittest discover -s python/test -p "*_test.py"

pause
