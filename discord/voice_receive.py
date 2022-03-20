from __future__ import annotations

import asyncio
from asyncio import transports
from collections import deque
import heapq
import logging
import struct
import time
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Generator,
    Literal,
    Optional,
    TYPE_CHECKING,
    Union,
)

import nacl.secret  # type: ignore

from . import opus
from .utils import ModularInt32

if TYPE_CHECKING:
    from .guild import Member
    from .user import User
    from .voice_client import VoiceClient

__all__ = ('VoiceReceiver',)


_log = logging.getLogger(__name__)


class VoiceReceiver:

    FRAME_LENGTH = opus.Decoder.FRAME_LENGTH
    SAMPLING_RATE = opus.Decoder.SAMPLING_RATE

    def __init__(
        self, vc: VoiceClient, buffer_duration=60, min_buffer=100, output_type: Literal["int16", "float"] = "int16"
    ):
        """`buffer_duration` is the long buffer in approximate seconds.
        `min_buffer` is the jitter buffer in approximate milliseconds.
        """

        self.voice_client = vc
        self.maxsize = (buffer_duration * 1000) // self.FRAME_LENGTH  # Is this an OK assumption?
        self.min_buffer = min_buffer // self.FRAME_LENGTH

        if output_type == "int16":
            self._Decoder = opus.Decoder
        elif output_type == "float":
            self._Decoder = opus.DecoderFloat
        else:
            raise ValueError("Invalid output type.")

        self.sample_size = self._Decoder.SAMPLE_SIZE

        # Timestamps are in samples

        # Buffer of (timestamp, local timestamp, opus data)
        self._long_buffers: dict[int, deque[tuple[ModularInt32, int, bytes]]] = {}  # {ssrc: Buffer}
        self._jitterbuffers: dict[int, list[tuple[ModularInt32, int, bytes]]] = {}

        self._decoders: dict[int, opus.Decoder] = {}  # {ssrc: Decoder}
        self._write_events: dict[int, asyncio.Event] = {}  # Notified per ssrc
        self._write_event = asyncio.Event()  # Notified for every write
        self._last_written: int | None = None  # last written ssrc
        self._new_write_event = asyncio.Event()  # New ssrc
        self._last_timestamp: dict[int, tuple[ModularInt32, int]] = {}  # {ssrc: (timestamp, duration)}
        self._get_lock = asyncio.Lock()
        self.ts_offsets: dict[int, float] = {}  # {ssrc: ts_offset} in seconds
        self.local_epoch: Union[float, None] = None
        self.get_user_lock = asyncio.Lock()
        self._user_last_ssrc: dict[int, int] = {}  # {user_id: ssrc} Last ssrc of the user.
        self._get_any_iterator = None

        # Public field
        # Maximum continous silence that can be generated in seconds, to prevent accidently writing very large files due to
        # a library bug or some user error.
        self.max_silence = 10 * 60

    def write(self, time_stamp: int, ssrc: int, opusdata: bytes):
        """This should not be called by user code.
        Appends the audio data to the buffer, drops it if buffer is full.
        """
        if ssrc not in self._long_buffers:  # First time for this ssrc
            self._new_write_event.set()
            self._write_events.setdefault(ssrc, asyncio.Event())
        ssrc_write_event = self._write_events[ssrc]
        ssrc_write_event.set()
        self._write_event.set()
        self._last_written = ssrc
        queue = self._long_buffers.setdefault(ssrc, deque())
        heap = self._jitterbuffers.setdefault(ssrc, [])

        local_now = time.monotonic_ns() // 10**6
        time_stamp = ModularInt32(time_stamp)
        item = (time_stamp, local_now, opusdata)
        if len(heap) < self.min_buffer:
            heapq.heappush(heap, item)
        else:
            if len(queue) < self.maxsize:
                queue.append(item)
            else:
                # The buffer is full. Drop it.
                return

        if self.local_epoch is None:
            self.local_epoch = time.monotonic()
        if ssrc not in self.ts_offsets:
            self.ts_offsets[ssrc] = time.monotonic() - self.local_epoch
            # TODO: better offset calculation based on some moving average

    def _get_user_id(self, ssrc: int) -> int | None:
        "Return the user id associated with the ssrc."
        return self.voice_client.ws.ssrc_map.get(ssrc)

    def _get_ssrc(self, user_id: int) -> list[int]:
        return [ssrc for ssrc, user in self.voice_client.ws.ssrc_map.items() if user == user_id]

    def _get_user(self, user_id: int) -> Member | User | None:
        member = self.voice_client.guild.get_member(user_id)
        if member is not None:
            return member
        else:
            user = self.voice_client.client.get_user(user_id)
            return user

    def _get_audio(self, ssrc) -> tuple[ModularInt32, bytes]:
        "Return decoded and padded audio pcm from the heap."
        queue = self._long_buffers[ssrc]
        heap = self._jitterbuffers[ssrc]
        timestamp, local_timestamp, enc_audio = heapq.heappop(heap)
        if len(queue) > 0:
            heapq.heappush(heap, queue.popleft())

        decoder = self._decoders.setdefault(ssrc, self._Decoder())
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

    async def _get_from_user(self, user: User | Member | int) -> tuple[ModularInt32, bytes] | None:
        """Return the audio data of a user of duration at least FRAME_LENGTH, or None if the user reconnects.
        The gaps between the received packets are padded.
        This method should only be called by one consumer per user.
        """
        while True:
            user_id = user if isinstance(user, int) else user.id
            ssrcs = self._get_ssrc(user_id)
            if not ssrcs:
                # Wait until we can actually get the ssrc.
                speak_recv = self.voice_client.ws.speak_received
                async with self.get_user_lock:
                    await speak_recv.wait()
                    speak_recv.clear()
                continue
            else:
                # If there are multiple ssrc values, that is because the user has reconnected.
                # In this case, first empty the older buffers. If the buffers but the last one is empty, wait on the last
                # ssrc key. _get_ssrc returns ssrc values in the order of the received SPEAK messages.
                for ssrc in ssrcs:
                    if self._jitterbuffers.get(ssrc) or ssrc == ssrcs[-1]:

                        if user_id in self._user_last_ssrc and ssrc != self._user_last_ssrc.get(user_id):
                            # The user has a new ssrc.
                            del self._user_last_ssrc[user_id]
                            return

                        get_from_ssrc_task = asyncio.create_task(self._get_from_ssrc(ssrc))
                        speak_event_wait = asyncio.create_task(self.voice_client.ws.speak_received.wait())
                        done, pending = await asyncio.wait(
                            (get_from_ssrc_task, speak_event_wait),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if speak_event_wait in done:  # A new SPEAK msg is received.
                            self.voice_client.ws.speak_received.clear()
                        else:  # Get task is completed but .wait() is still pending
                            speak_event_wait.cancel()
                            await asyncio.gather(speak_event_wait, return_exceptions=True)  # Await its cancellation.
                        if get_from_ssrc_task in done:  # Get task is completed
                            ssrc, timestamp, pcm = get_from_ssrc_task.result()
                            self._user_last_ssrc[user_id] = ssrc
                            return timestamp, pcm
                        else:  # A new SPEAK msg is received and get task is not completed.
                            get_task = pending.pop()
                            get_task.cancel()
                            await asyncio.gather(get_task, return_exceptions=True)  # Await its cancellation.
                            break

    async def _get_from_ssrc(self, ssrc: int) -> tuple[int, ModularInt32, bytes]:
        heap = self._jitterbuffers.setdefault(ssrc, [])

        if len(heap) >= self.min_buffer:
            return ssrc, *self._get_audio(ssrc)

        else:
            # Wait until either the heap is filled,
            # or some time has passed since the arrival of the packet at the top of the heap.
            # When a period of silence happens, we need to flush what's left over in the heap without waiting too much.
            # When the packets start arriving again, we don't return immediately to let the heap fill back up.
            write_event = self._write_events.setdefault(ssrc, asyncio.Event())
            while len(heap) < self.min_buffer:
                if len(heap) == 0:
                    wait_time = None
                else:
                    _, local_timestamp, _ = self._jitterbuffers[ssrc][0]
                    min_duration = self.min_buffer * self.FRAME_LENGTH / 1000
                    # If the min_duration is 0.100s, and the next packet arrived 0.700s ago, than wait 0.300s or until the
                    # heap is full, than return from the heap.
                    wait_time = min_duration - (time.monotonic() - local_timestamp / 1000)

                try:
                    await asyncio.wait_for(write_event.wait(), timeout=wait_time)
                except asyncio.TimeoutError:
                    # We waited enough.
                    # The heap can't be empty.
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
        """Yields timestamp, voice data of the given user. Stops after `duration` if not None.
        Stops after a period of silence of `silence_timeout`, if not None.
        Stops if the user reconnects.
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
                    result = get_task.result()
                    if not result:
                        return
                    timestamp, pcm = result

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
                    break_ = False

                    if len(done) == 2:  # Both the time_limit and the get_task completed.
                        done.remove(time_limit)
                        break_ = True

                    if (done_task := done.pop()) is time_limit:
                        # Stop iteration as we hit the duration limit.
                        # If the last part would be silence, it is not generated.
                        pending.pop().cancel()
                        return
                    else:
                        result = done_task.result()
                        if not result:
                            return
                        timestamp, pcm = result

                        # If gap is non-zero, fill the gaps with silence.
                        if fill_silence and not (last_timestamp is None or last_duration is None):
                            gap = timestamp - last_timestamp - last_duration

                            for silence in self._silenceiterator(gap):
                                new_timestamp = last_timestamp + last_duration
                                yield last_timestamp + last_duration, silence
                                last_timestamp = new_timestamp
                                last_duration = len(silence) // self.sample_size

                        yield timestamp, pcm

                        if break_:  # The time_limit task is also completed.
                            return

                        last_timestamp = timestamp
                        last_duration = len(pcm) // self.sample_size
                        tasks = pending
                        get_task = asyncio.create_task(self._get_from_user(user))
                        tasks.add(get_task)
                else:
                    # silence timeout
                    return

    def reset_user(self, user: User | Member | int):
        "Reset the state of a user."
        ssrcs = self._get_ssrc(user if isinstance(user, int) else user.id)
        for ssrc in ssrcs:
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

    def _get_any(self) -> AsyncIterator[tuple[Member | User | int, ModularInt32, bytes]]:
        if self._get_any_iterator is None:
            self._get_any_iterator = self._get_any_gen()
        return self._get_any_iterator

    async def _get_any_gen(self) -> AsyncGenerator[tuple[Member | User | int, ModularInt32, bytes], None]:
        """Yield from any user, or wait until one is available."""
        tasks: set[asyncio.Task] = set()
        for ssrc in self._write_events:
            get_cor = self._get_from_ssrc(ssrc)
            get_task = asyncio.create_task(get_cor)
            tasks.add(get_task)

        new_write_wait = asyncio.create_task(self._new_write_event.wait())
        tasks.add(new_write_wait)

        while True:
            try:
                done, pending = await asyncio.wait(tasks, timeout=None, return_when=asyncio.FIRST_COMPLETED)
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

            if done:  # no timeout

                write_event = any(task is new_write_wait for task in done)
                if write_event:  # Add a new get_event to the waited tasks.
                    done.remove(new_write_wait)
                    self._new_write_event.clear()  # Every write_event.wait() is followed by a .clear() .

                    new_get_coro = self._get_from_ssrc(self._last_written)  # type: ignore  # last_written can't be None.
                    new_get_task = asyncio.create_task(new_get_coro)
                    done.add(new_get_task)
                    new_write_wait = asyncio.create_task(self._new_write_event.wait())
                    done.add(new_write_wait)
                    done.update(pending)
                    tasks = done

                else:  # Audio is returned from at least one ssrc.
                    done_task = done.pop()
                    ssrc, timestamp, pcm = done_task.result()
                    user_id = self._get_user_id(ssrc)
                    if user_id is not None:
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
        return len(pcm_audio) / self._Decoder.SAMPLE_SIZE / self.SAMPLING_RATE

    def generate_silence(self, duration: int) -> bytes:
        "Duration is in samples. Return silence."
        return b"\x00" * self._Decoder.SAMPLE_SIZE * duration

    def _silenceiterator(self, duration: int) -> Generator[bytes, None, None]:
        "`duration` is in samples."
        duration = min(duration, self.max_silence * self._Decoder.SAMPLING_RATE)
        if duration == 0:
            return

        SILENCE_CHUNK = self.SAMPLING_RATE // 2  # 0.5 seconds
        for _ in range(duration // SILENCE_CHUNK):
            yield self.generate_silence(SILENCE_CHUNK)
        remainder = SILENCE_CHUNK % duration
        if remainder:
            yield self.generate_silence(remainder)

    def is_silence(self, pcm: bytes) -> bool:
        return not sum(pcm)

    def full(self, ssrc) -> bool:
        "If the buffer of the given ssrc is full."
        return ssrc in self._long_buffers and len(self._long_buffers[ssrc]) >= self.maxsize


class VoiceReceiveProtocol(asyncio.DatagramProtocol):
    def __init__(self, vr: VoiceReceiver):
        self.vr = vr
        self.vc = vr.voice_client
        self.transport = None
        self._recv_sequence: Optional[ModularInt32] = None

    def connection_made(self, transport: transports.DatagramTransport) -> None:
        self.transport = transport

    def connection_lost(self, exc: Exception | None) -> None:
        if exc is not None:
            raise exc

    def datagram_received(self, data: bytes, addr: tuple[str | Any, int]) -> None:
        vc = self.vr.voice_client
        try:
            ts, ssrc, x, data = self.unpack_voice_packet(data)
            if self.vr.full(ssrc):
                return
            opus_data = self.decrypt(data, x)
        except nacl.secret.exc.CryptoError as e:
            _log.warning(e)
        except ValueError:
            pass
        except Exception as e:
            _log.exception(e)
        else:
            self.vr.write(ts, ssrc, opus_data)

    def error_received(self, exc: OSError) -> None:
        _log.error(exc)

    def unpack_voice_packet(self, packet: bytes) -> tuple[int, int, int, bytes]:
        """Takes a voice packet, and returns a tuple of timestamp, SSRC, extension header flag, and the passed packet back.
        Can raise ValueError if the packet is not a supported voice packet.
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

        return timestamp, ssrc, x, packet

    def decrypt(self, packet: bytes, x, mode=...) -> bytes:
        "Return the decrypted audio data. Can raise CryptoError."
        if mode is ...:
            mode = self.vc.mode
        decrypt_fun: Callable[[bytes], bytes] = getattr(self.vc, "_decrypt_" + mode)
        data = decrypt_fun(packet)

        if x:  # Header extension present. It is contained within the encrypted portion.
            x_profile, x_length = struct.unpack("!HH", data[:4])
            header_ext = data[4 : 4 + 4 * x_length]
            payload = data[4 + 4 * x_length :]
            # debug_string_ext = f"x_profile: {x_profile}, x_length: {x_length}, header_ext: {header_ext}"
        else:
            payload = data

        return payload
