class IntegrityCollector:
    """Collect integrity issues and report them according to the selected mode."""

    def __init__(self, errors="warn"):
        if errors not in ("warn", "raise"):
            raise ValueError("errors must be 'warn' or 'raise'")

        self.errors = errors
        self.messages = []

    def report(self, message, logger):
        if self.errors == "raise":
            self.messages.append(message)
        else:
            logger.warning(message)

    def raise_if_errors(self):
        if self.errors == "raise" and self.messages:
            raise ValueError("\n".join(self.messages))