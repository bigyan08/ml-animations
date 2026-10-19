"""Caption segmentation, font measurement, and ASS composition in one place."""
from functools import cached_property
import re
import textwrap

from PIL import ImageFont


def normalize(value):
    return re.sub(r'[^a-z0-9]', '', value.lower())


def ass_time(seconds):
    centiseconds = round(seconds * 100)
    hours, rest = divmod(centiseconds, 360000)
    minutes, rest = divmod(rest, 6000)
    seconds, centiseconds = divmod(rest, 100)
    return f'{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}'


class CaptionLayout:
    def __init__(self, lesson):
        self.lesson = lesson
        self.style = lesson.captions

    @cached_property
    def font(self):
        return ImageFont.truetype(str(self.lesson.root / 'assets/fonts/ttf' /
                                      self.style.font_file), self.style.size)

    def lines(self, text):
        if self.style.layout == 'characters':
            return r'\N'.join(textwrap.wrap(text, width=self.style.width,
                              break_long_words=self.style.break_long_words,
                              break_on_hyphens=self.style.break_on_hyphens))
        words = text.split()
        if self.font.getlength(text) <= self.style.width:
            return text
        choices = []
        for cut in range(1, len(words)):
            left, right = ' '.join(words[:cut]), ' '.join(words[cut:])
            widths = self.font.getlength(left), self.font.getlength(right)
            if max(widths) <= self.style.width:
                penalty = 120 if words[cut-1].lower() in {
                    'a', 'an', 'the', 'of', 'to', 'not', 'minus', 'point'} else 0
                choices.append((abs(widths[0]-widths[1])+penalty, left+r'\N'+right))
        if not choices:
            raise ValueError(f'Caption needs a shorter cue: {text}')
        return min(choices)[1]

    def groups(self, tokens):
        normalized = [normalize(t) for t in tokens]
        blocked = set()
        for phrase in self.style.protected_phrases:
            terms = [normalize(t) for t in phrase.split()]
            for i in range(len(tokens)-len(terms)+1):
                if normalized[i:i+len(terms)] == terms:
                    blocked.update(range(i+1, i+len(terms)))
        groups, start = [], 0
        while start < len(tokens):
            stop = min(start+self.style.max_words, len(tokens))
            while stop in blocked:
                stop -= 1
            if stop <= start:
                raise ValueError('A protected phrase exceeds the caption word limit')
            if self.style.grouping == 'phrases':
                for i in range(start, stop):
                    end = i+1
                    if end not in blocked and (tokens[i].endswith(('.', '!', '?')) or
                        (end-start >= self.style.clause_min_words and
                         tokens[i].endswith((',', ';', ':')))):
                        stop = end
                        break
            if self.style.layout == 'measured':
                while True:
                    try:
                        self.lines(' '.join(tokens[start:stop]))
                        break
                    except ValueError:
                        stop -= 1
                        while stop in blocked:
                            stop -= 1
                        if stop <= start:
                            raise ValueError('A protected caption phrase cannot fit')
            groups.append((start, stop))
            start = stop
        if self.style.merge_short_tail and len(groups) > 1 and groups[-1][1]-groups[-1][0] < 3:
            groups[-2:] = [(groups[-2][0], groups[-1][1])]
        return groups

    def ass(self, narration):
        s, r = self.style, self.lesson.render
        lines = [f'''[Script Info]
ScriptType: v4.00+
PlayResX: {r.width}
PlayResY: {r.height}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{s.font},{s.size},&H00FAF2ED,&H00FAF2ED,&H0020100B,&H0020100B,0,0,0,0,100,100,0,0,1,2,0,2,{s.margin_left},{s.margin_right},{s.margin_bottom},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
''']
        elapsed = 0
        for beat in narration['beats']:
            for caption in beat['captions']:
                # Treat authored strings as text, never as ASS override commands.
                text = caption['text'].replace('{', '').replace('}', '').replace('\\', '')
                text = self.lines(text)
                lines.append(f"Dialogue: 0,{ass_time(elapsed+caption['start'])},"
                             f"{ass_time(elapsed+caption['end'])},Default,,0,0,0,,{text}\n")
            elapsed += beat['duration']
        return ''.join(lines)
