"""Narration-aware timing with unrestricted lesson-specific choreography."""
import json

import manimgx as m

from .captions import normalize
from .config import LessonConfig


class NarratedScene(m.Scene):
    pulse_color = '#4DE0C1'
    pulse_radius = .05

    def load_narration(self, scene_file):
        self.lesson = LessonConfig.load(scene_file)
        self.beats = self.lesson.generated()['beats']
        return self.beats

    def begin_beat(self, index):
        if isinstance(index, str):
            index = next(i for i, b in enumerate(self.beats) if b['id'] == index)
        beat = self.beats[index]
        self.current_beat = beat
        self.beat_start = float(self.time)
        self.beat_duration = beat['duration']
        self.add_sound(self.lesson.output/beat['audio'])
        for caption in beat['captions']:
            self.add_subcaption(caption['text'], duration=caption['end']-caption['start'],
                                offset=caption['start'])
        return self.beat_duration

    def cue(self, phrase):
        words = json.loads((self.lesson.output/self.current_beat['audio']).with_suffix('.json').read_text())
        tokens = [normalize(w) for w in self.current_beat['text'].split()]
        wanted = [normalize(w) for w in phrase.split()]
        if not wanted:
            raise ValueError('A spoken cue cannot be empty')
        for i in range(len(tokens)-len(wanted)+1):
            if tokens[i:i+len(wanted)] == wanted:
                target = self.beat_start+words[i]['start']
                if target > float(self.time):
                    self.wait(target-float(self.time))
                return
        raise ValueError(f'Spoken cue not found: {phrase}')

    def finish(self):
        remaining = self.beat_duration-(float(self.time)-self.beat_start)
        if remaining < -1e-7:
            raise ValueError(f'Animation exceeds narration beat by {-remaining:.2f}s')
        self.wait(max(0, remaining))

    def pulse(self, edges, duration, color=None):
        color = color or self.pulse_color
        dots = [m.Dot(e.get_start(), radius=self.pulse_radius, color=color) for e in edges]
        self.add(*dots)
        self.play(*[m.MoveAlongPath(d, e) for d, e in zip(dots, edges)],
                  run_time=duration, rate_func=m.linear)
        self.remove(*dots)
