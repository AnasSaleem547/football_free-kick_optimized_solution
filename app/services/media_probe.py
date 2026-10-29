"""Bounded container/stream inspection before decoding local video files."""
import json
import math
import subprocess
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

VIDEO_CONTAINER_FAMILIES = {
    '.mp4': {'mov', 'mp4'}, '.mov': {'mov', 'mp4'}, '.avi': {'avi'},
    '.webm': {'matroska', 'webm'}, '.mkv': {'matroska', 'webm'},
}


def _positive_rate(value):
    try:
        rate = float(Fraction(str(value)))
        return rate if math.isfinite(rate) and rate > 0 else None
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def probe_video(path, timeout=20):
    path = Path(path).resolve()
    allowed = VIDEO_CONTAINER_FAMILIES.get(path.suffix.lower())
    if allowed is None:
        raise ValueError('Unsupported video extension')
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('Video file is empty or missing')
    stat = path.stat()
    return dict(_probe_video(str(path), stat.st_size, stat.st_mtime_ns, timeout))


def _run_probe(arguments, timeout):
    try:
        response = subprocess.run(
            ['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe', *arguments],
            capture_output=True, text=True, check=False, timeout=timeout,
        )
    except FileNotFoundError as error:
        raise RuntimeError('FFprobe is required; install FFmpeg before starting the app') from error
    except subprocess.TimeoutExpired as error:
        raise ValueError('Video inspection timed out') from error
    if response.returncode or response.stderr.strip():
        raise ValueError('Unrecognized, corrupt or undecodable video container')
    try:
        return json.loads(response.stdout)
    except json.JSONDecodeError as error:
        raise ValueError('Invalid video container metadata') from error


@lru_cache(maxsize=32)
def _probe_video(path, size, modified_ns, timeout):
    """Reuse inspection only while this process sees the same path, size and mtime."""
    allowed = VIDEO_CONTAINER_FAMILIES[Path(path).suffix.lower()]
    data = _run_probe([
        '-show_entries',
        'format=format_name,duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:stream_disposition=attached_pic',
        '-of', 'json', path], timeout)
    try:
        container = data['format']
        if not allowed.intersection(container['format_name'].split(',')):
            raise ValueError('Video container does not match its filename extension')
        streams = [s for s in data.get('streams', [])
                   if s.get('codec_type') == 'video'
                   and not s.get('disposition', {}).get('attached_pic', 0)]
        if not streams:
            raise ValueError('Input has no real video stream')
        if len(streams) != 1:
            raise ValueError('Inputs with multiple video streams are not supported')
        stream = streams[0]
        if (not stream.get('codec_name') or stream['codec_name'] == 'unknown'
                or int(stream.get('width', 0)) <= 0 or int(stream.get('height', 0)) <= 0):
            raise ValueError('Video stream has an unknown codec or invalid dimensions')
        fps = _positive_rate(stream.get('avg_frame_rate')) or _positive_rate(stream.get('r_frame_rate'))
        if fps is None:
            raise ValueError('Video stream has no valid frame rate')
        count = stream.get('nb_frames')
        count = int(count) if str(count).isdigit() and int(count) > 0 else None
        duration = _positive_rate(stream.get('duration')) or _positive_rate(container.get('duration'))
        if count is None:
            # Duration times FPS is only an estimate, especially for variable-rate
            # footage. Decode/count instead, rejecting errors including early EOF.
            counted = _run_probe([
                '-select_streams', str(stream['index']), '-count_frames',
                '-show_entries', 'stream=nb_read_frames', '-of', 'json', path],
                max(timeout, 2 * (duration or 0)))
            value = counted['streams'][0].get('nb_read_frames')
            if not str(value).isdigit() or int(value) <= 0:
                raise ValueError('Cannot decode and count the video frames')
            count = int(value)
        return {'fps': fps, 'total_frames': count, 'duration_s': duration,
                'codec': stream['codec_name'], 'container': container['format_name']}
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError('Invalid video container metadata') from error
