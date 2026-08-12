import { useCallback, useEffect, useRef, useState } from "react";


// Two states:
//   * OVERVIEW (default) — not stepping; the view highlights everything the request touched.
//   * PLAYBACK — stepping/auto-playing; the view highlights just the current step.
// Any transport interaction leaves overview; `goOverview()` returns to it.
// STARK events are sub-second, so we replay by step, not wall-clock (R17).
export interface Replay {
  index: number;
  playing: boolean;
  overview: boolean;
  speed: number;
  length: number;
  setIndex: (i: number) => void;
  next: () => void;
  prev: () => void;
  toggle: () => void;
  setSpeed: (s: number) => void;
  goOverview: () => void;
  play: () => void;
  loop: boolean;
  setLoop: (v: boolean | ((p: boolean) => boolean)) => void;
}

const BASE_MS = 1100;

// onEnd fires once each time a NON-looping playthrough reaches the end (used for the demo playlist).
// If it returns true the run is being CONTINUED (e.g. playlist advancing) so we do NOT fade to the
// idle step 0 or pause — the next request will drive playback and the last frame stays up meanwhile.
export function useReplay(length: number, initialSpeed = 1, onEnd?: () => boolean | void): Replay {
  const [index, setIndexRaw] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [overview, setOverview] = useState(true);
  const [speed, setSpeed] = useState(initialSpeed);
  const [loop, setLoop] = useState(false);
  const timer = useRef<number | null>(null);
  const onEndRef = useRef(onEnd);
  useEffect(() => { onEndRef.current = onEnd; });

  const clamp = useCallback((i: number) => Math.max(0, Math.min(length - 1, i)), [length]);
  const setIndex = useCallback(
    (i: number) => {
      setOverview(false);
      setIndexRaw(clamp(i));
    },
    [clamp]
  );

  // reset when the trace changes
  useEffect(() => {
    setIndexRaw(0);
    setPlaying(false);
    setOverview(true);
  }, [length]);

  useEffect(() => {
    if (!playing) return;
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      if (index >= length - 1) {
        if (loop) {
          setIndexRaw(0); // wrap to step 0 (empty for the brain -> a clean fade-out between cycles)
          return;
        }
        const continued = onEndRef.current?.(); // playlist returns true -> keep flowing, no fade/pause
        if (!continued) {
          setPlaying(false);
          setIndexRaw(0); // finished -> rewind to the empty step 0 (graph fades out)
        }
        return;
      }
      setIndexRaw(index + 1);
    }, BASE_MS / speed);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [playing, index, speed, length, loop]);

  return {
    index,
    playing,
    overview,
    speed,
    length,
    setIndex,
    next: () => setIndex(index + 1),
    prev: () => setIndex(index - 1),
    toggle: () => {
      setOverview(false);
      setPlaying((p) => {
        if (!p && index >= length - 1) setIndexRaw(0); // paused on last -> restart from the beginning
        return !p;
      });
    },
    setSpeed,
    goOverview: () => {
      setPlaying(false);
      setOverview(true);
    },
    // start auto-playing from the top (used to live-animate a freshly-run request)
    play: () => {
      setOverview(false);
      setIndexRaw(0);
      setPlaying(true);
    },
    loop,
    setLoop,
  };
}
