"""
The MIT License (MIT)

Copyright (c) 2015-present Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.


Some documentation to refer to:

- Our main web socket (mWS) sends opcode 4 with a guild ID and channel ID.
- The mWS receives VOICE_STATE_UPDATE and VOICE_SERVER_UPDATE.
- We pull the session_id from VOICE_STATE_UPDATE.
- We pull the token, endpoint and server_id from VOICE_SERVER_UPDATE.
- Then we initiate the voice web socket (vWS) pointing to the endpoint.
- We send opcode 0 with the user_id, server_id, session_id and token using the vWS.
- The vWS sends back opcode 2 with an ssrc, port, modes(array) and hearbeat_interval.
- We send a UDP discovery packet to endpoint:port and receive our IP and our port in LE.
- Then we send our IP and port via vWS with opcode 1.
- When that's all done, we receive opcode 4 from the vWS.
- Finally we can transmit data to endpoint:port.
"""

from __future__ import annotations

import asyncio
from asyncio import PriorityQueue, Queue, QueueEmpty
import functools
import socket
import logging
import struct
import threading
import time
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Generator,
    List,
    Optional,
    TYPE_CHECKING,
    Tuple,
    Union,
)

from . import aioudp, opus, utils
from .backoff import ExponentialBackoff
from .gateway import *
from .errors import ClientException, ConnectionClosed
from .player import AudioPlayer, AudioSource
from .utils import MISSING, ModularInt32

if TYPE_CHECKING:
    from .client import Client
    from .guild import Guild, Member
    from .state import ConnectionState
    from .user import ClientUser, User
    from .opus import Encoder
    from .channel import StageChannel, VoiceChannel
    from . import abc

    from .types.voice import (
        GuildVoiceState as GuildVoiceStatePayload,
        VoiceServerUpdate as VoiceServerUpdatePayload,
        SupportedModes,
    )

    VocalGuildChannel = Union[VoiceChannel, StageChannel]


has_nacl: bool

try:
    import nacl.secret  # type: ignore
    import nacl.utils  # type: ignore

    has_nacl = True
except ImportError:
    has_nacl = False

__all__ = (
    'VoiceProtocol',
    'VoiceClient',
)


_log = logging.getLogger(__name__)


class VoiceProtocol:
    """A class that represents the Discord voice protocol.

    This is an abstract class. The library provides a concrete implementation
    under :class:`VoiceClient`.

    This class allows you to implement a protocol to allow for an external
    method of sending voice, such as Lavalink_ or a native library implementation.

    These classes are passed to :meth:`abc.Connectable.connect <VoiceChannel.connect>`.

    .. _Lavalink: https://github.com/freyacodes/Lavalink

    Parameters
    ------------
    client: :class:`Client`
        The client (or its subclasses) that started the connection request.
    channel: :class:`abc.Connectable`
        The voice channel that is being connected to.
    """

    def __init__(self, client: Client, channel: abc.Connectable) -> None:
        self.client: Client = client
        self.channel: abc.Connectable = channel

    async def on_voice_state_update(self, data: GuildVoiceStatePayload) -> None:
        """|coro|

        An abstract method that is called when the client's voice state
        has changed. This corresponds to ``VOICE_STATE_UPDATE``.

        Parameters
        ------------
        data: :class:`dict`
            The raw `voice state payload`__.

            .. _voice_state_update_payload: https://discord.com/developers/docs/resources/voice#voice-state-object

            __ voice_state_update_payload_
        """
        raise NotImplementedError

    async def on_voice_server_update(self, data: VoiceServerUpdatePayload) -> None:
        """|coro|

        An abstract method that is called when initially connecting to voice.
        This corresponds to ``VOICE_SERVER_UPDATE``.

        Parameters
        ------------
        data: :class:`dict`
            The raw `voice server update payload`__.

            .. _voice_server_update_payload: https://discord.com/developers/docs/topics/gateway#voice-server-update-voice-server-update-event-fields

            __ voice_server_update_payload_
        """
        raise NotImplementedError

    async def connect(self, *, timeout: float, reconnect: bool) -> None:
        """|coro|

        An abstract method called when the client initiates the connection request.

        When a connection is requested initially, the library calls the constructor
        under ``__init__`` and then calls :meth:`connect`. If :meth:`connect` fails at
        some point then :meth:`disconnect` is called.

        Within this method, to start the voice connection flow it is recommended to
        use :meth:`Guild.change_voice_state` to start the flow. After which,
        :meth:`on_voice_server_update` and :meth:`on_voice_state_update` will be called.
        The order that these two are called is unspecified.

        Parameters
        ------------
        timeout: :class:`float`
            The timeout for the connection.
        reconnect: :class:`bool`
            Whether reconnection is expected.
        """
        raise NotImplementedError

    async def disconnect(self, *, force: bool) -> None:
        """|coro|

        An abstract method called when the client terminates the connection.

        See :meth:`cleanup`.

        Parameters
        ------------
        force: :class:`bool`
            Whether the disconnection was forced.
        """
        raise NotImplementedError

    def cleanup(self) -> None:
        """This method *must* be called to ensure proper clean-up during a disconnect.

        It is advisable to call this from within :meth:`disconnect` when you are
        completely done with the voice protocol instance.

        This method removes it from the internal state cache that keeps track of
        currently alive voice clients. Failure to clean-up will cause subsequent
        connections to report that it's still connected.
        """
        key_id, _ = self.channel._get_voice_client_key()
        self.client._connection._remove_voice_client(key_id)


class VoiceReceiver:

    FRAME_LENGTH = opus.Decoder.FRAME_LENGTH
    SAMPLING_RATE = opus.Decoder.SAMPLING_RATE

    def __init__(self, vc: VoiceClient, buffer_duration=60, min_buffer=100):
        """`buffer_duration` is the long buffer in approximate seconds.
        `min_buffer` is the jitter buffer in approximate milliseconds.
        """

        self.voice_client = vc
        self.maxsize = (buffer_duration * 1000) // self.FRAME_LENGTH  # Is this an OK assumption?
        self.min_buffer = min_buffer // self.FRAME_LENGTH

        self.sample_size = opus.Decoder.SAMPLE_SIZE

        # Timestamps are in samples

        # Buffer of (timestamp, local timestamp, opus data)
        Buffer = Queue[tuple[ModularInt32, int, bytes]]
        self._Buffer = Buffer
        self._long_buffers: dict[int, Buffer] = {}  # {ssrc: Buffer}
        self._jitterbuffers: dict[int, PriorityQueue[tuple[ModularInt32, int, bytes]]] = {}

        self._decoders: dict[int, opus.Decoder] = {}  # {ssrc: Decoder}
        self._get_user_locks: dict[int, asyncio.Lock] = {}
        self._write_events: dict[int, asyncio.Event] = {}  # Notified per ssrc
        self._write_event = asyncio.Event()  # Notified for every write
        self._last_written: int | None = None  # last written ssrc
        self._new_write_event = asyncio.Event()  # New ssrc
        self._last_timestamp: dict[int, tuple[ModularInt32, int]] = {}  # {ssrc: (timestamp, duration)}
        self._get_lock = asyncio.Lock()
        self.ts_offsets: dict[int, float] = {}  # {ssrc: ts_offset} in seconds
        self.local_epoch: Union[float, None] = None

    def write(self, time_stamp: int, ssrc: int, opusdata: bytes):
        """This should not be called by user code.
        Appends the audio data to the buffer, drops it if buffer is full.
        """
        queue = self._long_buffers.setdefault(ssrc, self._Buffer(self.maxsize))
        heap = self._jitterbuffers.setdefault(ssrc, PriorityQueue(self.min_buffer))

        local_now = time.monotonic_ns() // 10**6
        time_stamp = ModularInt32(time_stamp)
        item = (time_stamp, local_now, opusdata)
        if not heap.full():
            heap.put_nowait(item)
        else:
            if not queue.full():
                queue.put_nowait(item)
            else:
                # The buffer is full. Drop it.
                return

        if ssrc not in self._write_events:
            self._new_write_event.set()
            self._write_events[ssrc] = asyncio.Event()
        ssrc_write_event = self._write_events[ssrc]
        ssrc_write_event.set()
        self._write_event.set()
        self._last_written = ssrc

        if self.local_epoch is None:
            self.local_epoch = time.monotonic()
        if ssrc not in self.ts_offsets:
            self.ts_offsets[ssrc] = time.monotonic() - self.local_epoch
            # TODO: better offset calculation based on some moving average

    _WAIT_SPEAK_PACKET = 0.050

    async def _get_user_id(self, ssrc: int) -> int | None:
        "Return the user id associated with the ssrc. If not known, wait a little in case a SPEAK message arrives."
        async with self._get_user_locks.setdefault(ssrc, asyncio.Lock()):
            user_id = self.voice_client.ws.ssrc_map.get(ssrc)
            if user_id is None:
                # If user is not yet known, it should be known when the SPEAK packet arrives.
                await asyncio.sleep(self._WAIT_SPEAK_PACKET)
                user_id = self.voice_client.ws.ssrc_map.get(ssrc)
            return user_id

    async def _get_ssrc(self, user_id: int) -> int | None:
        try:
            return next(ssrc for ssrc, user in self.voice_client.ws.ssrc_map.items() if user == user_id)
        except StopIteration:
            await asyncio.sleep(self._WAIT_SPEAK_PACKET)
            return next(
                (ssrc for ssrc, user in self.voice_client.ws.ssrc_map.items() if user == user_id),
                None,
            )

    def _get_user(self, user_id: int) -> Member | User | None:
        member = self.voice_client.guild.get_member(user_id)
        if member is not None:
            return member
        else:
            user = self.voice_client.client.get_user(user_id)
            return user

    def _get_audio(self, ssrc) -> tuple[ModularInt32, bytes]:
        "Return decoded and padded audio pcm from the heap."
        queue = self._long_buffers.setdefault(ssrc, Queue(self.maxsize))
        heap = self._jitterbuffers.setdefault(ssrc, PriorityQueue(self.min_buffer))
        timestamp, local_timestamp, enc_audio = heap.get_nowait()
        try:
            heap.put_nowait(queue.get_nowait())
        except QueueEmpty:
            pass
        decoder = self._decoders.setdefault(ssrc, opus.Decoder())
        last_timestamp, last_duration = self._last_timestamp.get(ssrc, (None, None))

        nb_frames = decoder.packet_get_nb_frames(enc_audio)
        samples_per_frame = decoder.packet_get_samples_per_frame(enc_audio)
        nb_samples = nb_frames * samples_per_frame

        if last_timestamp is None or last_duration is None:
            gap = 0
        else:
            gap = timestamp - last_timestamp - last_duration

        # Limit how much silence we can generate.
        # Making this too much can result in segmentation error.
        MAX_GAP = self.SAMPLING_RATE // 2
        gap = min(gap, MAX_GAP)
        if gap >= 0:
            if gap > 0:
                try:
                    missing_pcm = decoder.decode(
                        enc_audio,
                        fec=True,
                        missing_duration=gap,
                    )
                except opus.OpusError as e:
                    _log.error(e)
                    missing_pcm = b"\x00" * gap * decoder.SAMPLE_SIZE
            else:
                missing_pcm = b""
            try:
                real_pcm = decoder.decode(
                    enc_audio,
                    fec=False,
                    nb_frames=nb_frames,
                    samples_per_frame=samples_per_frame,
                )
            except opus.OpusError as e:
                _log.error(e)
                real_pcm = b"\x00" * nb_samples * decoder.SAMPLE_SIZE

            final_ts = timestamp - gap
            self._last_timestamp[ssrc] = (final_ts, nb_samples + gap)

            return final_ts, missing_pcm + real_pcm

        elif gap > -last_duration:  # type: ignore  # last_duration can't be None here.
            # This packet is overlapping with the last one. This doesn't happen if the sender is healthy.
            # What is the best approach here?
            # Cut out the overlapping part of the audio.
            # This way, the total output size cannot be greater than what the returned timestamps imply.
            non_overlap: int = gap + last_duration  # type: ignore
            try:
                real_pcm = decoder.decode(
                    enc_audio,
                    fec=False,
                    nb_frames=nb_frames,
                    samples_per_frame=samples_per_frame,
                )
                real_pcm = real_pcm[:-non_overlap]
            except opus.OpusError as e:
                _log.error(e)
                real_pcm = b"" * non_overlap * decoder.SAMPLE_SIZE

            final_ts = timestamp - gap
            self._last_timestamp[ssrc] = (final_ts, non_overlap)

            return final_ts, real_pcm

        else:
            # This voice packet is before the last one or at the same time.
            return last_timestamp + last_duration, b""  # type: ignore

    @staticmethod
    def _lock(method):
        @functools.wraps(method)
        async def lock(self, *args, **kwargs):
            async with self._get_lock:
                return await method(self, *args, *kwargs)

        return lock

    async def _get_from_user(self, user: User | Member | int) -> tuple[ModularInt32, bytes]:
        """Return the audio data of a user of duration at least FRAME_LENGTH.
        The gaps between the received packets are padded.
        This method should only be called by one consumer.
        """
        while True:
            if isinstance(user, int):
                ssrc = await self._get_ssrc(user)
            else:
                ssrc = await self._get_ssrc(user.id)

            if ssrc is None:
                # Wait until we can actually get the ssrc.
                await self._new_write_event.wait()
                continue
            else:
                ssrc, timestamp, pcm = await self._get_from_ssrc(ssrc)
                return timestamp, pcm

    async def _get_from_ssrc(self, ssrc: int) -> tuple[int, ModularInt32, bytes]:
        heap = self._jitterbuffers.setdefault(ssrc, PriorityQueue(self.min_buffer))

        if heap.full():
            return ssrc, *self._get_audio(ssrc)

        else:
            # Wait until either the heap is filled,
            # or some time has passed since the arrival of the packet at the top of the heap.
            # When a period of silence happens, we need to flush what's left over in the heap without waiting too much.
            # When the packets start arriving again, we don't return immediately to let the heap fill back up.
            write_event = self._write_events.setdefault(ssrc, asyncio.Event())
            while not heap.full():
                if heap.empty():
                    wait_time = None
                else:
                    # The hack of accessing the private member is necessary to get the next item in the heap without
                    # popping.
                    _, local_timestamp, _ = self._jitterbuffers[ssrc]._queue[0]  # type: ignore
                    min_duration = self.min_buffer * self.FRAME_LENGTH / 1000
                    # If the min_duration is 0.100s, and the next packet arrived 0.700s ago, than wait 0.300s or until the
                    # heap is full, than return from the heap.
                    wait_time = min_duration - (time.monotonic() - local_timestamp / 1000)

                done, pending = await asyncio.wait((write_event.wait(),), timeout=wait_time)
                if pending:
                    # We waited enough.
                    # The heap can't be empty.

                    # Cancel the write_event.wait()
                    pending.pop().cancel()

                    break
                else:
                    # Something was pushed to the heap. We have to recalculate the waiting time, or just return if the heap
                    # has filled after this write event.
                    write_event.clear()  # Each resolved write_event.wait() must be followed by .clear()

            return ssrc, *self._get_audio(ssrc)

    async def iterate_user(
        self,
        user: User | Member | int,
        duration: Optional[float] = None,
        silence_timeout: Optional[float] = None,
        fill_silence=True,
    ) -> AsyncIterable[tuple[ModularInt32, bytes]]:
        """Yield timestamp, voice data of the given user. Stop after `duration` if not None.
        Stop after a period of silence of `silence_timeout`, if not None.
        If `fill_silence` is true, gaps in the audio is yielded as silence. This makes the total duration returned
        consistent with how much time has passed.
        If the last part of the audio is silence, it is not yielded.
        """
        if duration is None:
            last_timestamp, last_duration = None, None
            while True:
                get_task = asyncio.create_task(self._get_from_user(user))
                try:
                    await asyncio.wait_for(get_task, silence_timeout)
                except asyncio.TimeoutError:
                    return
                else:
                    timestamp, pcm = get_task.result()

                    # If gap is non-zero, fill the gaps with silence
                    if fill_silence and not (last_timestamp is None or last_duration is None):
                        gap = timestamp - last_timestamp - last_duration

                        for silence in self._silenceiterator(gap):
                            new_timestamp = last_timestamp + last_duration
                            yield last_timestamp + last_duration, silence
                            last_timestamp = new_timestamp
                            last_duration = len(silence) // self.sample_size

                    last_timestamp, last_duration = timestamp, len(pcm) // self.sample_size
                    yield timestamp, pcm

        elif duration is not None:
            time_limit = asyncio.create_task(asyncio.sleep(duration))
            get_task = asyncio.create_task(self._get_from_user(user))
            tasks = {get_task, time_limit}
            last_timestamp, last_duration = None, None
            while True:
                try:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=silence_timeout)
                except asyncio.CancelledError:
                    for t in tasks:
                        t.cancel()
                    raise

                if done:
                    # no silence timeout
                    done_task = done.pop()
                    if done_task is time_limit:
                        # Stop iteration as we hit the duration limit.
                        # If the last part would be silence, it is not generated.
                        pending.pop().cancel()
                        return
                    else:
                        timestamp, pcm = done_task.result()

                        # If gap is non-zero, fill the gaps with silence.
                        if fill_silence and not (last_timestamp is None or last_duration is None):
                            gap = timestamp - last_timestamp - last_duration

                            for silence in self._silenceiterator(gap):
                                new_timestamp = last_timestamp + last_duration
                                yield last_timestamp + last_duration, silence
                                last_timestamp = new_timestamp
                                last_duration = len(silence) // self.sample_size

                        yield timestamp, pcm
                        last_timestamp = timestamp
                        last_duration = len(pcm) // self.sample_size
                        tasks = pending
                        get_task = asyncio.create_task(self._get_from_user(user))
                        tasks.add(get_task)
                else:
                    # silence timeout
                    return

    async def reset_user(self, user: User | Member | int):
        "Reset the state of a user."
        ssrc = await self._get_ssrc(user if isinstance(user, int) else user.id)
        if ssrc is None:
            return
        del self._long_buffers[ssrc]
        del self._jitterbuffers[ssrc]
        del self._decoders[ssrc]
        del self._last_timestamp[ssrc]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.voice_client.stop_receiving()

    def __aiter__(self):
        return self()

    async def _get_any(self) -> AsyncGenerator[tuple[Member | User | int, ModularInt32, bytes], None]:
        """Yield from any user, or wait until one is available."""
        tasks: set[asyncio.Task] = set()
        for ssrc in self._write_events:
            get_cor = self._get_from_ssrc(ssrc)
            get_task = asyncio.create_task(get_cor)
            tasks.add(get_task)

        new_write_wait = asyncio.create_task(self._new_write_event.wait())
        tasks.add(new_write_wait)

        while True:
            done, pending = await asyncio.wait(tasks, timeout=None)

            if done:  # no timeout

                write_event = any(task is new_write_wait for task in done)
                if write_event:  # Add a new get_event to the waited tasks.
                    done.remove(new_write_wait)
                    self._new_write_event.clear()  # Every write_event.wait() is followed by a .clear() .

                    new_get_coro = self._get_from_ssrc(self._last_written)  # type: ignore  # last_written can't be None.
                    new_get_task = asyncio.create_task(new_get_coro)
                    done.add(new_get_task)
                    done.update(pending)
                    tasks = done

                else:  # Audio is returned from at least one ssrc.
                    done_task = done.pop()
                    ssrc, timestamp, pcm = done_task.result()

                    user_id = await self._get_user_id(ssrc)
                    if user_id is None:
                        # Drop it, and reschedule the task.
                        new_get_coro = self._get_from_ssrc(ssrc)
                        new_get_task = asyncio.create_task(new_get_coro)
                        pending.add(new_get_task)
                    else:
                        user = self._get_user(user_id) or user_id
                        yield user, timestamp, pcm

                    done.update(pending)
                    get_cor = self._get_from_ssrc(ssrc)
                    get_task = asyncio.create_task(get_cor)
                    done.add(get_task)
                    tasks = done

    def __call__(self, timeout: Optional[float] = None) -> AsyncIterator[tuple[Member | User | int, ModularInt32, bytes]]:

        if timeout is None:
            return self._get_any()
        else:

            async def timeout_iterator():
                target_time = time.monotonic() + timeout
                get_any_gen = self._get_any()
                while True:
                    get_task = asyncio.create_task(anext(get_any_gen))
                    try:
                        wait_time = target_time - time.monotonic()
                        yield await asyncio.wait_for(get_task, wait_time)
                    except asyncio.TimeoutError:
                        return

            return timeout_iterator()

    def get_duration(self, pcm_audio: bytes) -> float:
        "Return the duration in seconds of the audio returned by this object."
        return len(pcm_audio) / opus.Decoder.SAMPLE_SIZE / self.SAMPLING_RATE

    def generate_silence(self, duration: int) -> bytes:
        "Duration is in samples. Return silence."
        return b"\x00" * opus.Decoder.SAMPLE_SIZE * duration

    def _silenceiterator(self, duration: int) -> Generator[bytes, None, None]:
        "`duration` is in samples."
        if duration == 0:
            yield b""
            return

        SILENCE_CHUNK = self.SAMPLING_RATE // 2  # 0.5 seconds
        for _ in range(duration // SILENCE_CHUNK):
            yield self.generate_silence(SILENCE_CHUNK)
        remainder = SILENCE_CHUNK % duration
        if remainder:
            yield self.generate_silence(remainder)


class VoiceClient(VoiceProtocol):
    """Represents a Discord voice connection.

    You do not create these, you typically get them from
    e.g. :meth:`VoiceChannel.connect`.

    Warning
    --------
    In order to use PCM based AudioSources, you must have the opus library
    installed on your system and loaded through :func:`opus.load_opus`.
    Otherwise, your AudioSources must be opus encoded (e.g. using :class:`FFmpegOpusAudio`)
    or the library will not be able to transmit audio.

    Attributes
    -----------
    session_id: :class:`str`
        The voice connection session ID.
    token: :class:`str`
        The voice connection token.
    endpoint: :class:`str`
        The endpoint we are connecting to.
    channel: Union[:class:`VoiceChannel`, :class:`StageChannel`]
        The voice channel connected to.
    """

    channel: VocalGuildChannel
    endpoint_ip: str
    voice_port: int
    ip: str
    port: int
    secret_key: List[int]
    ssrc: int

    def __init__(self, client: Client, channel: abc.Connectable):
        if not has_nacl:
            raise RuntimeError("PyNaCl library needed in order to use voice")

        super().__init__(client, channel)
        state = client._connection
        self.token: str = MISSING
        self.server_id: int = MISSING
        self.socket = MISSING
        self._local_endpoint: aioudp.LocalEndpoint = MISSING
        self.loop: asyncio.AbstractEventLoop = state.loop
        self._state: ConnectionState = state
        # this will be used in the AudioPlayer thread
        self._connected: threading.Event = threading.Event()

        self._handshaking: bool = False
        self._potentially_reconnecting: bool = False
        self._voice_state_complete: asyncio.Event = asyncio.Event()
        self._voice_server_complete: asyncio.Event = asyncio.Event()

        self.mode: str = MISSING
        self._connections: int = 0
        self.sequence: int = 0
        self.timestamp: int = 0
        self.timeout: float = 0
        self._runner: asyncio.Task = MISSING
        self._player: Optional[AudioPlayer] = None
        self.encoder: Encoder = MISSING
        self._lite_nonce: int = 0
        self.ws: DiscordVoiceWebSocket = MISSING
        self._receiving = False  # Whether the received voice data is buffered rather than discarded.
        # Incremented whenever audio data is recieved, reset when the connection is renewed
        self._recv_sequence: Optional[int] = None
        self._vr: VoiceReceiver = MISSING
        self._receive_loop_task = MISSING

    warn_nacl: bool = not has_nacl
    supported_modes: Tuple[SupportedModes, ...] = (
        'xsalsa20_poly1305_lite',
        'xsalsa20_poly1305_suffix',
        'xsalsa20_poly1305',
    )

    @property
    def guild(self) -> Guild:
        """:class:`Guild`: The guild we're connected to."""
        return self.channel.guild

    @property
    def user(self) -> ClientUser:
        """:class:`ClientUser`: The user connected to voice (i.e. ourselves)."""
        return self._state.user  # type: ignore - user can't be None after login

    def checked_add(self, attr: str, value: int, limit: int) -> None:
        val = getattr(self, attr)
        if val + value > limit:
            setattr(self, attr, 0)
        else:
            setattr(self, attr, val + value)

    # connection related

    async def on_voice_state_update(self, data: GuildVoiceStatePayload) -> None:
        self.session_id: str = data['session_id']
        channel_id = data['channel_id']

        if not self._handshaking or self._potentially_reconnecting:
            # If we're done handshaking then we just need to update ourselves
            # If we're potentially reconnecting due to a 4014, then we need to differentiate
            # a channel move and an actual force disconnect
            if channel_id is None:
                # We're being disconnected so cleanup
                await self.disconnect()
            else:
                self.channel = channel_id and self.guild.get_channel(int(channel_id))  # type: ignore - this won't be None
        else:
            self._voice_state_complete.set()

    async def on_voice_server_update(self, data: VoiceServerUpdatePayload) -> None:
        if self._voice_server_complete.is_set():
            _log.info('Ignoring extraneous voice server update.')
            return

        self.token = data['token']
        self.server_id = int(data['guild_id'])
        endpoint = data.get('endpoint')

        if endpoint is None or self.token is None:
            _log.warning(
                'Awaiting endpoint... This requires waiting. '
                'If timeout occurred considering raising the timeout and reconnecting.'
            )
            return

        self.endpoint, _, _ = endpoint.rpartition(':')
        if self.endpoint.startswith('wss://'):
            # Just in case, strip it off since we're going to add it later
            self.endpoint: str = self.endpoint[6:]

        # This gets set later
        self.endpoint_ip = MISSING

        self.socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)

        if not self._handshaking:
            # If we're not handshaking then we need to terminate our previous connection in the websocket
            await self.ws.close(4000)
            return

        self._voice_server_complete.set()

    async def voice_connect(self) -> None:
        await self.channel.guild.change_voice_state(channel=self.channel)

    async def voice_disconnect(self) -> None:
        _log.info('The voice handshake is being terminated for Channel ID %s (Guild ID %s)', self.channel.id, self.guild.id)
        await self.channel.guild.change_voice_state(channel=None)

    def prepare_handshake(self) -> None:
        self._voice_state_complete.clear()
        self._voice_server_complete.clear()
        self._handshaking = True
        _log.info('Starting voice handshake... (connection attempt %d)', self._connections + 1)
        self._connections += 1

    def finish_handshake(self) -> None:
        _log.info('Voice handshake complete. Endpoint found %s', self.endpoint)
        self._handshaking = False
        self._voice_server_complete.clear()
        self._voice_state_complete.clear()

    async def connect_websocket(self) -> DiscordVoiceWebSocket:
        ws = await DiscordVoiceWebSocket.from_client(self)
        self._connected.clear()
        while ws.secret_key is None:
            await ws.poll_event()
        self._connected.set()
        return ws

    async def connect(self, *, reconnect: bool, timeout: float) -> None:
        _log.info('Connecting to voice...')
        self.timeout = timeout

        for i in range(5):
            self.prepare_handshake()

            # This has to be created before we start the flow.
            futures = [
                self._voice_state_complete.wait(),
                self._voice_server_complete.wait(),
            ]

            # Start the connection flow
            await self.voice_connect()

            try:
                await utils.sane_wait_for(futures, timeout=timeout)
            except asyncio.TimeoutError:
                await self.disconnect(force=True)
                raise

            self.finish_handshake()

            try:
                self.ws = await self.connect_websocket()
                break
            except (ConnectionClosed, asyncio.TimeoutError):
                if reconnect:
                    _log.exception('Failed to connect to voice... Retrying...')
                    await asyncio.sleep(1 + i * 2.0)
                    await self.voice_disconnect()
                    continue
                else:
                    raise

        if self._runner is MISSING:
            self._runner = self.client.loop.create_task(self.poll_voice_ws(reconnect))

    async def potential_reconnect(self) -> bool:
        # Attempt to stop the player thread from playing early
        self._connected.clear()
        self.prepare_handshake()
        self._potentially_reconnecting = True
        try:
            # We only care about VOICE_SERVER_UPDATE since VOICE_STATE_UPDATE can come before we get disconnected
            await asyncio.wait_for(self._voice_server_complete.wait(), timeout=self.timeout)
        except asyncio.TimeoutError:
            self._potentially_reconnecting = False
            await self.disconnect(force=True)
            return False

        self.finish_handshake()
        self._potentially_reconnecting = False
        try:
            self.ws = await self.connect_websocket()
        except (ConnectionClosed, asyncio.TimeoutError):
            return False
        else:
            return True

    @property
    def latency(self) -> float:
        """:class:`float`: Latency between a HEARTBEAT and a HEARTBEAT_ACK in seconds.

        This could be referred to as the Discord Voice WebSocket latency and is
        an analogue of user's voice latencies as seen in the Discord client.

        .. versionadded:: 1.4
        """
        ws = self.ws
        return float("inf") if not ws else ws.latency

    @property
    def average_latency(self) -> float:
        """:class:`float`: Average of most recent 20 HEARTBEAT latencies in seconds.

        .. versionadded:: 1.4
        """
        ws = self.ws
        return float("inf") if not ws else ws.average_latency

    async def poll_voice_ws(self, reconnect: bool) -> None:
        backoff = ExponentialBackoff()
        while True:
            try:
                await self.ws.poll_event()
            except (ConnectionClosed, asyncio.TimeoutError) as exc:
                if isinstance(exc, ConnectionClosed):
                    # The following close codes are undocumented so I will document them here.
                    # 1000 - normal closure (obviously)
                    # 4014 - voice channel has been deleted.
                    # 4015 - voice server has crashed
                    if exc.code in (1000, 4015):
                        _log.info('Disconnecting from voice normally, close code %d.', exc.code)
                        await self.disconnect()
                        break
                    if exc.code == 4014:
                        _log.info('Disconnected from voice by force... potentially reconnecting.')
                        successful = await self.potential_reconnect()
                        if not successful:
                            _log.info('Reconnect was unsuccessful, disconnecting from voice normally...')
                            await self.disconnect()
                            break
                        else:
                            continue

                if not reconnect:
                    await self.disconnect()
                    raise

                retry = backoff.delay()
                _log.exception('Disconnected from voice... Reconnecting in %.2fs.', retry)
                self._connected.clear()
                await asyncio.sleep(retry)
                await self.voice_disconnect()
                try:
                    await self.connect(reconnect=True, timeout=self.timeout)
                except asyncio.TimeoutError:
                    # at this point we've retried 5 times... let's continue the loop.
                    _log.warning('Could not connect to voice... Retrying...')
                    continue

    async def disconnect(self, *, force: bool = False) -> None:
        """|coro|

        Disconnects this voice client from voice.
        """
        if not force and not self.is_connected():
            return

        self.stop()
        self.stop_receiving()
        self._connected.clear()

        try:
            if self.ws:
                await self.ws.close()

            await self.voice_disconnect()
        finally:
            self.cleanup()
            if self.socket:
                self.socket.close()

    async def move_to(self, channel: Optional[abc.Snowflake]) -> None:
        """|coro|

        Moves you to a different voice channel.

        Parameters
        -----------
        channel: Optional[:class:`abc.Snowflake`]
            The channel to move to. Must be a voice channel.
        """
        await self.channel.guild.change_voice_state(channel=channel)

    def is_connected(self) -> bool:
        """Indicates if the voice client is connected to voice."""
        return self._connected.is_set()

    # audio related

    def _get_voice_packet(self, data):
        header = bytearray(12)

        # Formulate rtp header
        header[0] = 0x80
        header[1] = 0x78
        struct.pack_into('>H', header, 2, self.sequence)
        struct.pack_into('>I', header, 4, self.timestamp)
        struct.pack_into('>I', header, 8, self.ssrc)

        encrypt_packet = getattr(self, '_encrypt_' + self.mode)
        return encrypt_packet(header, data)

    def _unpack_voice_packet(self, packet: bytes, mode: str) -> tuple[int, int, bytes]:
        """Takes a voice packet, and returns a tuple of timestamp, SSRC, and unencrypted audio payload.
        mode is the encryption mode. Can raise CryptoError, or ValueError if the packet is not a supported voice packet.
        """
        # TODO: Optimize by reducing copy operations.

        # https://www.rfcreader.com/#rfc3550_line548
        vpxcc, mpt, sequence, timestamp, ssrc = struct.unpack("!BBHII", packet[:12])

        v = vpxcc >> 6
        p = (vpxcc >> 5) & 1
        x = (vpxcc >> 4) & 1
        cc = vpxcc & (2**4 - 1)
        m = mpt >> 7
        pt = mpt & (2**8 - 1)

        # debug_string = (
        #     f"v: {v}, p: {p}, x: {x}, cc: {cc}, m: {m}, pt: {pt}, seq: {sequence}, ts: {timestamp}, ssrc: {ssrc}"
        # )

        if not (v == 2 and p == 0 and cc == 0 and pt == 0x78):
            raise ValueError("Unsupported packet.")

        if self._recv_sequence is not None and (missing_packets := (sequence - self._recv_sequence - 1)):
            # TODO: use this to warn about about dropped packets
            pass
        self._recv_sequence = sequence

        decrypt_fun: Callable[[bytes], bytes] = getattr(self, "_decrypt_" + mode)
        data = decrypt_fun(packet)

        if x:  # Header extension present. It is contained within the encrypted portion.
            x_profile, x_length = struct.unpack("!HH", data[:4])
            header_ext = data[4 : 4 + 4 * x_length]
            payload = data[4 + 4 * x_length :]
            # debug_string_ext = f"x_profile: {x_profile}, x_length: {x_length}, header_ext: {header_ext}"
        else:
            payload = data

        return timestamp, ssrc, payload

    def _encrypt_xsalsa20_poly1305(self, header: bytes, data) -> bytes:
        box = nacl.secret.SecretBox(bytes(self.secret_key))
        nonce = bytearray(24)
        nonce[:12] = header

        return header + box.encrypt(bytes(data), bytes(nonce)).ciphertext

    def _encrypt_xsalsa20_poly1305_suffix(self, header: bytes, data) -> bytes:
        box = nacl.secret.SecretBox(bytes(self.secret_key))
        nonce = nacl.utils.random(nacl.secret.SecretBox.NONCE_SIZE)

        return header + box.encrypt(bytes(data), nonce).ciphertext + nonce

    def _encrypt_xsalsa20_poly1305_lite(self, header: bytes, data) -> bytes:
        box = nacl.secret.SecretBox(bytes(self.secret_key))
        nonce = bytearray(24)

        nonce[:4] = struct.pack('>I', self._lite_nonce)
        self.checked_add('_lite_nonce', 1, 4294967295)

        return header + box.encrypt(bytes(data), bytes(nonce)).ciphertext + nonce[:4]

    def _decrypt_xsalsa20_poly1305(self, packet: bytes) -> bytes:
        packet_view = memoryview(packet)
        header = packet_view[:12]
        enc_payload = packet_view[12:]

        nonce = bytearray(12)
        nonce[:12] = header
        box = nacl.secret.SecretBox(bytes(self.secret_key))

        return box.decrypt(enc_payload, nonce)

    def _decrypt_xsalsa20_poly1305_suffix(self, packet: bytes) -> bytes:
        packet_view = memoryview(packet)
        enc_payload = packet_view[12:-24]

        nonce = packet_view[-24:]
        box = nacl.secret.SecretBox(bytes(self.secret_key))

        return box.decrypt(enc_payload, nonce)

    def _decrypt_xsalsa20_poly1305_lite(self, packet: bytes) -> bytes:
        packet_view = memoryview(packet)
        enc_payload = packet_view[12:-4]

        nonce = bytes(packet_view[-4:]) + b"\00" * 20
        box = nacl.secret.SecretBox(bytes(self.secret_key))

        return box.decrypt(bytes(enc_payload), nonce)

    def play(self, source: AudioSource, *, after: Optional[Callable[[Optional[Exception]], Any]] = None) -> None:
        """Plays an :class:`AudioSource`.

        The finalizer, ``after`` is called after the source has been exhausted
        or an error occurred.

        If an error happens while the audio player is running, the exception is
        caught and the audio player is then stopped.  If no after callback is
        passed, any caught exception will be displayed as if it were raised.

        Parameters
        -----------
        source: :class:`AudioSource`
            The audio source we're reading from.
        after: Callable[[Optional[:class:`Exception`]], Any]
            The finalizer that is called after the stream is exhausted.
            This function must have a single parameter, ``error``, that
            denotes an optional exception that was raised during playing.

        Raises
        -------
        ClientException
            Already playing audio or not connected.
        TypeError
            Source is not a :class:`AudioSource` or after is not a callable.
        OpusNotLoaded
            Source is not opus encoded and opus is not loaded.
        """

        if not self.is_connected():
            raise ClientException('Not connected to voice.')

        if self.is_playing():
            raise ClientException('Already playing audio.')

        if not isinstance(source, AudioSource):
            raise TypeError(f'source must be an AudioSource not {source.__class__.__name__}')

        if not self.encoder and not source.is_opus():
            self.encoder = opus.Encoder()

        self._player = AudioPlayer(source, self, after=after)
        self._player.start()

    def is_playing(self) -> bool:
        """Indicates if we're currently playing audio."""
        return self._player is not None and self._player.is_playing()

    def is_paused(self) -> bool:
        """Indicates if we're playing audio, but if we're paused."""
        return self._player is not None and self._player.is_paused()

    def stop(self) -> None:
        """Stops playing audio."""
        if self._player:
            self._player.stop()
            self._player = None

    def pause(self) -> None:
        """Pauses the audio playing."""
        if self._player:
            self._player.pause()

    def resume(self) -> None:
        """Resumes the audio playing."""
        if self._player:
            self._player.resume()

    @property
    def source(self) -> Optional[AudioSource]:
        """Optional[:class:`AudioSource`]: The audio source being played, if playing.

        This property can also be used to change the audio source currently being played.
        """
        return self._player.source if self._player else None

    @source.setter
    def source(self, value: AudioSource) -> None:
        if not isinstance(value, AudioSource):
            raise TypeError(f'expected AudioSource not {value.__class__.__name__}.')

        if self._player is None:
            raise ValueError('Not playing anything.')

        self._player._set_source(value)

    def send_audio_packet(self, data: bytes, *, encode: bool = True) -> None:
        """Sends an audio packet composed of the data.

        You must be connected to play audio.

        Parameters
        ----------
        data: :class:`bytes`
            The :term:`py:bytes-like object` denoting PCM or Opus voice data.
        encode: :class:`bool`
            Indicates if ``data`` should be encoded into Opus.

        Raises
        -------
        ClientException
            You are not connected.
        opus.OpusError
            Encoding the data failed.
        """

        self.checked_add('sequence', 1, 65535)
        if encode:
            encoded_data = self.encoder.encode(data, self.encoder.SAMPLES_PER_FRAME)
        else:
            encoded_data = data
        packet = self._get_voice_packet(encoded_data)
        try:
            self.socket.sendto(packet, (self.endpoint_ip, self.voice_port))
        except BlockingIOError:
            _log.warning('A packet has been dropped (seq: %s, timestamp: %s)', self.sequence, self.timestamp)

        self.checked_add('timestamp', opus.Encoder.SAMPLES_PER_FRAME, 4294967295)

    async def _receive_audio_packet(self):
        """Receive an audio packet and write it to the VoiceReceive instance.
        Can raise CryptoError or ValueError. `start_receiving` method needs to be called first.
        """
        assert self._receiving
        packet, address = await self._local_endpoint.receive()
        # TODO: Check the address.

        self._vr.write(*self._unpack_voice_packet(packet, self.mode))

    async def _receive_loop(self):
        while True:
            try:
                await self._receive_audio_packet()
            except nacl.secret.exc.CryptoError as e:
                # TODO: Log warning
                _log.error(e)
                pass
            except ValueError as e:
                pass
            except Exception as e:
                _log.exception(e)
                pass

    async def start_receiving(self, min_buffer=100, buffer=60) -> VoiceReceiver:
        """Start receiving audio from the voice channel. Returns a VoiceReceive instance."""
        # TODO: Docstring
        # TODO: Change exception types
        # TODO: More checks, eg. if we are in a channel
        # TODO: Stop at disconnect, don't stop at successful reconnect.
        assert not self._receiving
        assert self._receive_loop_task is MISSING

        loop = asyncio.get_event_loop()
        endpoint = aioudp.LocalEndpoint()
        await loop.create_datagram_endpoint(
            lambda: aioudp.DatagramEndpointProtocol(endpoint),
            sock=self.socket,
        )
        self._local_endpoint = endpoint

        self._vr = VoiceReceiver(self, buffer, min_buffer)
        self._receive_loop_task = asyncio.create_task(self._receive_loop())
        self._receiving = True

        return self._vr

    def stop_receiving(self):
        "Stop receiving audio."
        self._vr = MISSING
        self._receive_loop_task.cancel()
        self._local_endpoint.close()

    @property
    def voice_receiver(self):
        "Return the Voice Receiver object, or None if voice is not currently being recorded."
        return self._vr or None
