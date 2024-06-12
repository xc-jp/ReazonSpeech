from __future__ import annotations

import json
import os
import typing

if typing.TYPE_CHECKING:
    from src.interface import Segment


class VTTWriter:
    """WebVTT writer.

    WebVTT (Web Video Text Tracks) is a standard caption format defined by W3C in 2010.
    It's supported by major browsers, and can be used with HTML5.

    See also: https://www.w3.org/TR/webvtt1/
    """

    ext = "vtt"

    def __init__(self, fp: typing.IO) -> None:
        self.fp = fp

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds / 3600)
        m = int(seconds / 60) % 60
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return "%02i:%02i:%02i.%03i" % (h, m, s, ms)

    def write_header(self) -> None:
        self.fp.write("WEBVTT\n\n")

    def write(self, segment: Segment) -> None:
        start = self._format_time(segment.start_seconds)
        end = self._format_time(segment.end_seconds)
        self.fp.write(f"{start} --> {end}\n{segment.text}\n\n")


class SRTWriter:
    """SRT writer.

    SRT is a subtitle format commonly used by desktop programs.
    It was originally developed by a Windows program SubRip.

    See also: https://www.matroska.org/technical/subtitles.html#srt-subtitles
    """

    ext = "srt"

    def __init__(self, fp: typing.IO) -> None:
        self.fp = fp
        self.index = 0

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds / 3600)
        m = int(seconds / 60) % 60
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return "%02i:%02i:%02i,%03i" % (h, m, s, ms)

    @staticmethod
    def write_header() -> None:
        return

    def write(self, segment: Segment) -> None:
        self.index += 1
        start = self._format_time(segment.start_seconds)
        end = self._format_time(segment.end_seconds)
        self.fp.write("%i\n%s --> %s\n%s\n\n" % (self.index, start, end, segment.text))


class ASSWriter:
    """ASS writer.

    ASS is another common format among desktop apps.
    It was developed by Advanced Sub Station Alpha, and can be used to burn subtitles using libass.

    See also: https://github.com/libass/libass
    See also: https://trac.ffmpeg.org/wiki/HowToBurnSubtitlesIntoVideo
    """

    ext = "ass"

    def __init__(self, fp: typing.IO) -> None:
        self.fp = fp

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds / 3600)
        m = int(seconds / 60) % 60
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return "%i:%02i:%02i.%02i" % (h, m, s, cs)

    def write_header(self) -> None:
        self.fp.write("""\
[Script Info]
ScriptType: v4.00+
Collisions: Normal
Timer: 100.0000

[V4+ Styles]
Style: Default,Arial,16,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,0

[Events]
""")

    def write(self, segment: Segment) -> None:
        start = self._format_time(segment.start_seconds)
        end = self._format_time(segment.end_seconds)
        self.fp.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{segment.text}\n")


class JSONWriter:
    """JSON (JavaScript Object Notation) writer."""

    ext = "json"

    def __init__(self, fp: typing.IO) -> None:
        self.fp = fp

    @staticmethod
    def write_header() -> None:
        return

    def write(self, ts: Segment) -> None:
        line = json.dumps(
            {"start_seconds": round(ts.start_seconds, 3), "end_seconds": round(ts.end_seconds, 3), "text": ts.text},
            ensure_ascii=False,
        )
        self.fp.write(line + "\n")


class TSVWriter:
    """TSV (Tab-separated values) writer."""

    ext = "tsv"

    def __init__(self, fp: typing.IO) -> None:
        self.fp = fp

    def write_header(self) -> None:
        self.fp.write("start_seconds\tend_seconds\ttext\n")

    def write(self, segment: Segment) -> None:
        self.fp.write(f"{segment.start_seconds:.3f}\t{segment.end_seconds:.3f}\t{segment.text}\n")


class TextWriter:
    ext = "txt"

    def __init__(self, fp: typing.IO) -> None:
        self.fp = fp

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds / 3600)
        m = int(seconds / 60) % 60
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return "%02i:%02i:%02i.%03i" % (h, m, s, ms)

    @staticmethod
    def write_header() -> None:
        return

    def write(self, segment: Segment) -> None:
        start = self._format_time(segment.start_seconds)
        end = self._format_time(segment.end_seconds)
        self.fp.write(f"[{start} --> {end}] {segment.text}\n")


def get_writer(
    fp: typing.IO, ext: str | None = None
) -> VTTWriter | SRTWriter | ASSWriter | JSONWriter | TSVWriter | TextWriter:
    if ext is None:
        name = getattr(fp, "name", "")
        ext = os.path.splitext(name)[-1]  # noqa: PTH122

    for cls in (VTTWriter, SRTWriter, ASSWriter, JSONWriter, TSVWriter):
        if cls.ext == ext:
            return cls(fp)

    return TextWriter(fp)
