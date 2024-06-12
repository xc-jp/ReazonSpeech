"""USAGE.

    reazonspeech-nemo-asr [-h] [--to={vtt,srt,ass,json,tsv}] [-o file] audio

OPTIONS

    audio
        Audio file to transcribe. It can be in any format as long
        as librosa.load() can read.

    -h, --help
        Print this help message.

    --to={vtt,srt,ass,json,tsv}
        Output format for transcription

    -o file, --output=file
        File to write transcription

Examples
    # Transcribe audio file
    $ reazonspeech-nemo-asr sample.wav

    # Output subtitles in VTT format
    $ reazonspeech-nemo-asr -o sample.vtt sample.webm
"""

import getopt
import locale
import sys
import warnings
from typing import Optional

from .audio import audio_from_path
from .transcribe import load_model, transcribe
from .writer import get_writer


def main() -> Optional[int]:
    outpath = None
    outext = None

    opts, args = getopt.getopt(
        sys.argv[1:],
        "ho:",
        (
            "help",
            "output=",
            "to=",
        ),
    )
    for k, v in opts:
        if k in {"-h", "--help"}:
            return None
        elif k in {"-o", "--output"}:
            outpath = v
        elif k == "--to":
            outext = v

    if not args:
        return 1

    outfile = open(outpath, "w", encoding=locale.getpreferredencoding(False)) if outpath is not None else sys.stdout

    # Suppress warnings from ESPnet
    warnings.simplefilter("ignore")

    # Load audio data and model
    audio = audio_from_path(args[0])
    model = load_model()

    # Perform inference
    ret = transcribe(model, audio)

    with outfile:
        writer = get_writer(outfile, outext)
        writer.write_header()
        for ts in ret.segments:
            writer.write(ts)
        return None


if __name__ == "__main__":
    sys.exit(main())
