import redis
from server.core.config import settings
_pool = redis.ConnectionPool.from_url(settings.cfg["stores"]["valkey_url"])
r = redis.Redis(connection_pool=_pool)
def set_step(key: str, value: str, ttl_sec: int = 3600): r.set(key, value, ex=ttl_sec)
def get_step(key: str): 
    v = r.get(key); 
    return v.decode() if v else None
