# Web UI Dev Server Reloading

## Current Behavior
- The server runs with `use_reloader=false` in `webui/run.py` to avoid `SystemExit` under debuggers like `debugpy`.
- With the reloader disabled, code changes do not auto-restart the server.

## Restart Options
- Manual restart
  - Stop with `Ctrl+C`
  - Start: `pipenv run python webui/run.py`
- Flask reloader (not under debugger)
  - `pipenv run python -m flask --app webui.app run --debug --reload -p 7070`
  - Or temporarily set `app.run(..., use_reloader=True)` in `webui/run.py`
- External file watcher
  - Install: `pipenv run pip install watchdog`
  - Run: `watchmedo auto-restart -d webui -p "*.py" -- pipenv run python webui/run.py`

## Notes
- The Werkzeug reloader spawns a child process; under `debugpy` this can surface as `SystemExit`. Use the reloader only when not attaching a debugger.
