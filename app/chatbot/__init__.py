
def build_chat_chain(*args, **kwargs):
	from .chain import build_chat_chain as _build

	return _build(*args, **kwargs)


def get_chat_chain(*args, **kwargs):
	from .chain import get_chat_chain as _get

	return _get(*args, **kwargs)


__all__ = ["build_chat_chain", "get_chat_chain"]
