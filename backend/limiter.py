from config import settings

try:
	from slowapi import Limiter
	from slowapi.util import get_remote_address
except ModuleNotFoundError:
	Limiter = None
	get_remote_address = None


class _NoopLimiter:
	def limit(self, *_args, **_kwargs):
		def decorator(func):
			return func
		return decorator

if Limiter is None:
	limiter = _NoopLimiter()
else:
	limiter = Limiter(
		key_func=get_remote_address,
		default_limits=["60/minute"],
		enabled=settings.RATE_LIMIT_ENABLED,
	)
