from asyncio import subprocess
import logging
import datetime as dt
import discord as dc
from discord.ext import commands


MEMBERS_INTENT = True


intents = dc.Intents()
intents.voice_states = True
intents.guilds = True
intents.members = MEMBERS_INTENT
intents.messages = True
intents.message_content = True
bot = commands.Bot(";", intents=intents)


class FffmpegWrite:
    ARGS = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-i",
        "pipe:",
    ]

    def __init__(self, filename):
        self.args = self.ARGS + [filename]
        self.proc = None

    async def open(self):
        self.proc = await subprocess.create_subprocess_exec(
            *self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )

    async def write(self, data):
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()

    async def close(self):
        self.proc.stdin.close()
        await self.proc.stdin.wait_closed()
        return await self.proc.wait()


@bot.command()
async def record(ctx: commands.Context, duration: int = 5):
    """Join the voice channel of the author, and record their voice to a file."""
    try:
        voice_channel = ctx.author.voice.channel
    except AttributeError:
        await ctx.send("You need to be in a voice channel.")
        return

    ffmpeg = FffmpegWrite(f"{ctx.author} {dt.datetime.now()}.mp3")

    if ctx.guild.voice_client:
        # Already connected to voice
        return

    try:
        await ffmpeg.open()

        async with await voice_channel.connect() as voice_client:

            voice_receiver = await voice_client.start_receiving()

            async for timestamp, pcm in voice_receiver.iterate_user(ctx.author, duration=duration):
                await ffmpeg.write(pcm)

    finally:
        await ffmpeg.close()


if __name__ == "__main__":
    TOKEN = "token"
    logging.basicConfig(level=logging.INFO)
    bot.run(TOKEN)
