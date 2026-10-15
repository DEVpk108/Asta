"""Runtime logging configuration; no changes until explicitly configured."""
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import threading

_lock = threading.RLock()


class JSONFormatter(logging.Formatter):
    def format(self, record):
        payload = dict(time=self.formatTime(record), level=record.levelname,
                       logger=record.name, message=record.getMessage())
        if record.exc_info:
            payload['exception'] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging():
    """Configure owned handlers only; preserve libraries'/host application handlers.

    ASTA_LOG_LEVEL defaults to INFO; ASTA_LOG_FILE is opt-in because transcripts
    can contain personal information. JSON format is optional. No global print
    interception or builtins monkey patching is used.
    """
    with _lock:
        root = logging.getLogger()
        level = getattr(logging, os.getenv('ASTA_LOG_LEVEL', 'INFO').upper(), None)
        if not isinstance(level, int):
            raise ValueError('ASTA_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL')
        formatter = JSONFormatter() if os.getenv('ASTA_LOG_FORMAT', '').lower() == 'json' else logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        handlers = [logging.StreamHandler(sys.stdout)]
        try:
            filename = os.getenv('ASTA_LOG_FILE', '').strip()
            if filename:
                path = Path(filename).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(path, maxBytes=5*1024*1024,
                    backupCount=3, encoding='utf-8')
                handlers.append(file_handler)
                if os.name != 'nt':
                    path.chmod(0o600)
            for handler in handlers:
                handler.setFormatter(formatter)
                handler._asta_owned = True
        except BaseException:
            for handler in handlers:
                handler.close()
            raise
        for handler in list(root.handlers):
            if getattr(handler, '_asta_owned', False):
                root.removeHandler(handler)
                handler.close()
        root.setLevel(level)
        for handler in handlers:
            root.addHandler(handler)
        return tuple(handlers)
