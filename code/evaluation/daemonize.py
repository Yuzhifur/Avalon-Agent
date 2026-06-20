#!/usr/bin/env python3
"""Fully detach a command into its own session (macOS has no `setsid`).

Standard double-fork daemon: the grandchild gets a new session (no controlling
tty) and is reparented to init, so it survives the launching shell/agent session
ending — the failure mode that deadlocked the first Gate-1 attempt for ~20h.

Usage: python3 daemonize.py <logfile> <pidfile> <cmd> [args...]
Prints nothing; the parent exits 0 immediately. The daemon's pid is written to
<pidfile>; its stdout+stderr go to <logfile>; stdin is /dev/null.
"""
import os
import sys

log, pidfile, cmd = sys.argv[1], sys.argv[2], sys.argv[3:]
if not cmd:
    sys.stderr.write("usage: daemonize.py <log> <pidfile> <cmd...>\n")
    raise SystemExit(2)

if os.fork() > 0:
    os._exit(0)            # original parent returns to the caller immediately
os.setsid()                # new session, detach from controlling terminal
if os.fork() > 0:
    os._exit(0)            # session leader exits; grandchild can't reacquire a tty

with open(pidfile, "w") as fh:
    fh.write(str(os.getpid()))

fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
dn = os.open(os.devnull, os.O_RDONLY)
os.dup2(dn, 0)
os.execvp(cmd[0], cmd)
