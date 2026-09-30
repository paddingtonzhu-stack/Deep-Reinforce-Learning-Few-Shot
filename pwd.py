"""Minimal Windows compatibility shim used by Sample Factory for log metadata."""
from collections import namedtuple
import getpass

struct_passwd = namedtuple("struct_passwd", "pw_name pw_passwd pw_uid pw_gid pw_gecos pw_dir pw_shell")

def getpwuid(uid):
    return struct_passwd(getpass.getuser(), "", uid, 0, "", "", "")
