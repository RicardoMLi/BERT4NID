import logging

from config import logging_config, path_config


class Logger(logging.Logger):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        sh = logging.StreamHandler()
        sh.setFormatter(logging_config.stream_formatter)
        sh.setLevel(logging_config.level)
        self.addHandler(sh)

        fh = logging.FileHandler(path_config.logs / f'{name}.log')
        fh.setFormatter(logging_config.file_formatter)
        fh.setLevel(logging_config.level)
        self.addHandler(fh)

    def turn_on(self):
        self.setLevel(logging_config.level)
        for handler in self.handlers:
            handler.setLevel(logging_config.level)

    def turn_off(self):
        self.setLevel(logging.CRITICAL + 1)
        for handler in self.handlers:
            handler.setLevel(logging.CRITICAL + 1)

