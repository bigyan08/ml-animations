"""Run with python -m shared COMMAND animations/SUBJECT/TOPIC/LESSON."""
import argparse
import asyncio
import os

from .config import LessonConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('narrate', 'render', 'verify', 'finalize'):
        child = sub.add_parser(command)
        child.add_argument('lesson', help='Lesson directory containing lesson.toml')
        child.add_argument('--media-root', help='Alternate build directory, e.g. /tmp/preview-media')
        if command == 'narrate':
            child.add_argument('--offline', action='store_true', help='Require cached speech; never contact TTS')
        if command == 'render':
            child.add_argument('--preview', action='store_true')
            child.add_argument('--captions-only', action='store_true')
        if command == 'finalize':
            child.add_argument('--reviewed', action='store_true', help='Visual and audio review has been completed')
    args = parser.parse_args()
    if args.media_root:
        os.environ['ML_ANIMATIONS_MEDIA_ROOT'] = args.media_root
    try:
        lesson = LessonConfig.load(args.lesson)
        if args.command == 'narrate':
            from .narration import NarrationPipeline
            asyncio.run(NarrationPipeline(lesson).build(offline=args.offline))
        elif args.command == 'render':
            from .rendering import Renderer
            Renderer(lesson).render(args.preview, args.captions_only)
        elif args.command == 'verify':
            from .rendering import verify
            info = verify(lesson)
            print(f'Verified: {lesson.final} ({info["format"]["duration"]}s)')
        else:
            from .rendering import finalize
            _, count = finalize(lesson, reviewed=args.reviewed)
            print(f'Kept {lesson.final}; removed {count} generated files')
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
