#! /usr/bin/env bash

NAME="sdot"
TAG="0.1"
RELEASE="$NAME-$TAG.zip"

cd ..
zip -r $RELEASE $NAME -y -x "$NAME/bak/*" "$NAME/dist/*" "$NAME/build/*" \
    "$NAME/releases/*" "$NAME/.cache/*" \
    "$NAME/.git/*" "$NAME/.git*" "$NAME/TODO.md" "$NAME/examples/out/*" "$NAME/examples/misc/*" \
    "$NAME/src/$NAME.egg-info/*" "$NAME/**/*.pyc" "$NAME/examples/assets_link" \
    "$NAME/$NAME.zip" "$NAME/lib/geogram/build/*" "$NAME/benchmarks/*" "$NAME/**/*.pkl" \
    "$NAME/cameraman.jpg" "$NAME/scripts/*" "$NAME/.pytest_cache/*" \
    "$NAME/**/__pycache__/" "$NAME/results/*" \
    "$NAME/*.xyz"
mv $RELEASE $NAME/releases/
cd $NAME

