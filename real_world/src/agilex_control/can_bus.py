"""Passive multi-arm feedback and explicitly selected-arm CAN transmission."""

import collections
import select
import socket
import struct
import time
from .protocol import decode_state


class CanBus:
    def __init__(self, arms, enforce_ownership=False):
        self.arms = dict(arms)
        self.enforce_ownership = enforce_ownership
        self.sockets, self.frames, self.stamps = {}, {}, {}
        self.expected = collections.deque(maxlen=300)
        self.events = []
        self.dropped_frames = {}
        self.started = time.monotonic()
        self.writers = {}
        try:
            for name, interface in arms.items():
                s = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                self.sockets[s] = name
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
                # Linux x86_64 socket constants; firmware timestamps do not
                # substitute for host receive age when the reader is delayed.
                s.setsockopt(
                    socket.SOL_SOCKET, getattr(socket, "SO_TIMESTAMPNS", 35), 1
                )
                s.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_RXQ_OVFL", 40), 1)
                s.bind((interface,))
                s.setblocking(False)
                self.frames[name], self.stamps[name] = {}, {}
        except BaseException:
            self.close()
            raise

    def open_writer(self, arm, interface):
        if self.arms.get(arm) != interface:
            raise ValueError("Writer must use the configured arm interface")
        if arm in self.writers:
            raise RuntimeError("Writer already open for " + arm)
        writer = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.writers[arm] = writer
        writer.settimeout(0.1)
        writer.bind((interface,))

    def send(self, cid, data, arm=None):
        if cid not in (0x151, 0x155, 0x156, 0x157, 0x159) or len(data) != 8:
            raise ValueError(
                "Only reviewed mode, joint and gripper frames are supported"
            )
        if arm is None and len(self.writers) == 1:
            arm = next(iter(self.writers))
        if arm not in self.writers:
            raise ValueError("An open, unambiguous arm writer is required")
        self.expected.append((arm, cid, data))
        if self.writers[arm].send(struct.pack("=IB3x8s", cid, 8, data)) != 16:
            raise OSError("Incomplete CAN frame transmission")

    def pump(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                list(self.sockets), [], [], max(0, deadline - time.monotonic())
            )
            for s in ready:
                raw, ancillary, flags, _ = s.recvmsg(
                    16, socket.CMSG_SPACE(16) + socket.CMSG_SPACE(4)
                )
                if len(raw) != 16:
                    raise RuntimeError("Unexpected CAN frame length")
                cid, dlc, data = struct.unpack("=IB3x8s", raw)
                data = data[:dlc]
                arm = self.sockets[s]
                now = time.monotonic()
                received_at = None
                for level, kind, value in ancillary:
                    if level == socket.SOL_SOCKET and kind == getattr(
                        socket, "SO_TIMESTAMPNS", 35
                    ):
                        sec, nsec = struct.unpack("=qq", value[:16])
                        received_at = now - max(0, time.time() - (sec + nsec / 1e9))
                    elif level == socket.SOL_SOCKET and kind == getattr(
                        socket, "SO_RXQ_OVFL", 40
                    ):
                        self.dropped_frames[arm] = struct.unpack("=I", value[:4])[0]
                if received_at is None:
                    raise RuntimeError("Kernel receive timestamp missing")
                self.frames[arm][cid], self.stamps[arm][cid] = data, received_at
                if 0x150 <= cid <= 0x179 or cid in (0x470, 0x471):
                    event = dict(
                        t=now - self.started,
                        arm=arm,
                        id=hex(cid),
                        data=data.hex(),
                        local=bool(flags & socket.MSG_DONTROUTE),
                    )
                    self.events.append(event)
                    if self.enforce_ownership and (
                        not event["local"] or (arm, cid, data) not in self.expected
                    ):
                        raise RuntimeError("Another control source: " + str(event))

    def state(self):
        now = time.monotonic()
        return {
            arm: decode_state(self.frames[arm], self.stamps[arm], now)
            for arm in self.frames
        }

    def close(self):
        for writer in self.writers.values():
            writer.close()
        for s in self.sockets:
            s.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
