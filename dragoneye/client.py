import os
from typing import Optional

from dragoneye.classification import Classification


class Dragoneye:
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("DRAGONEYE_API_KEY")

        assert (
            api_key is not None
        ), "API key is required - set the DRAGONEYE_API_KEY environment variable or pass it to the [Dragoneye] constructor"

        self.api_key = api_key

        self.classification = Classification(self)
