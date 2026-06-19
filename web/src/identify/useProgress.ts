import { useCallback, useEffect, useRef, useState } from "react";
import { fmtTime } from "../shared/format";

export interface LogLine {
  text: string;
  state: "active" | "done";
}

// Client-side progress simulation. The dashboard uses plain POST /identify (not
// the NDJSON stream — a short request survives the Cloudflare tunnel better), so
// pipeline stages are faked on a timer while the request runs. This hook owns
// the single elapsed counter so an errored request can never leak a timer that
// fights the next request's readout (the bug the original guarded against).
export function useProgress() {
  const [visible, setVisible] = useState(false);
  const [stage, setStage] = useState("Starting…");
  const [elapsedLabel, setElapsedLabel] = useState("0:00 elapsed · usually 8–45s");
  const [pbar, setPbar] = useState(8);
  const [log, setLog] = useState<LogLine[]>([]);

  const timers = useRef<{ elapsed?: number; stages: number[] }>({ stages: [] });
  const count = useRef(0);

  const clearTimers = useCallback(() => {
    const t = timers.current;
    if (t.elapsed) window.clearInterval(t.elapsed);
    t.elapsed = undefined;
    t.stages.forEach(window.clearTimeout);
    t.stages = [];
  }, []);

  useEffect(() => clearTimers, [clearTimers]); // tear down on unmount

  const show = useCallback(
    (on: boolean) => {
      // Always tear down prior timers first, so an errored request can't leave
      // an elapsed counter running.
      clearTimers();
      setVisible(on);
      if (on) {
        count.current = 0;
        setLog([]);
        setPbar(8);
        setStage("Starting…");
        const t0 = Date.now();
        setElapsedLabel("0:00 elapsed · usually 8–45s");
        timers.current.elapsed = window.setInterval(() => {
          setElapsedLabel(fmtTime((Date.now() - t0) / 1000) + " elapsed · usually 8–45s");
        }, 300);
      }
    },
    [clearTimers]
  );

  const pushStage = useCallback((text: string) => {
    count.current += 1;
    setLog((prev) => [...prev.map((l) => ({ ...l, state: "done" as const })), { text, state: "active" as const }]);
    setStage(text.replace(/…$/, ""));
    setPbar(Math.min(92, 8 + count.current * 16));
  }, []);

  const scheduleStages = useCallback(
    (stages: (string | null)[], gap = 600) => {
      stages
        .filter((s): s is string => Boolean(s))
        .forEach((s, i) => timers.current.stages.push(window.setTimeout(() => pushStage(s), i * gap)));
    },
    [pushStage]
  );

  const finish = useCallback(() => {
    setLog((prev) => prev.map((l) => ({ ...l, state: "done" as const })));
    setPbar(100);
    setStage("Done");
  }, []);

  return { visible, stage, elapsedLabel, pbar, log, show, scheduleStages, finish };
}
