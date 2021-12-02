#!/usr/bin/env sh

set -e

DECORATORS_DIR=$(python -c 'import decorators;print(decorators.__path__[0])')
TESTRUNNER_DIR=$(python -c 'import weblabTestRunner;print(weblabTestRunner.__path__[0])')

cd output/implementation/
for testsuite in $(find . -name testsuite.py)
do
    echo Testing ${testsuite}
    cd $(dirname ${testsuite})
    PYTHONPATH=${DECORATORS_DIR}:${TESTRUNNER_DIR} python3 testsuite.py
    cd - > /dev/null
done
