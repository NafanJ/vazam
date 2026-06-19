import { useCallback, useRef, useState } from "react";

// MediaRecorder wrapper. Toggles recording, enforces a hard 20s cap, and hands
// the raw blob back via onClip when recording stops. Mirrors the original
// dashboard's toggleRecord / onRecStop.
export function useRecorder(onClip: (blob: Blob, name: string) => void) {
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0); // seconds
  const [micError, setMicError] = useState<string | null>(null);

  const ref = useRef<{
    mr?: MediaRecorder;
    chunks: Blob[];
    stream?: MediaStream;
    timer?: number;
    start: number;
  }>({ chunks: [], start: 0 });

  const stop = useCallback(() => {
    const r = ref.current;
    if (r.mr && r.mr.state === "recording") r.mr.stop();
  }, []);

  const toggle = useCallback(async () => {
    const r = ref.current;
    if (r.mr && r.mr.state === "recording") {
      stop();
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setMicError("🎙 Mic unavailable — use file upload instead.");
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setMicError("🎙 Mic denied or needs HTTPS — use file upload instead.");
      return;
    }
    setMicError(null);
    r.chunks = [];
    r.stream = stream;
    const mr = new MediaRecorder(stream);
    r.mr = mr;
    mr.ondataavailable = (e) => {
      if (e.data.size) r.chunks.push(e.data);
    };
    mr.onstop = () => {
      if (r.timer) window.clearInterval(r.timer);
      setRecording(false);
      setElapsed(0);
      r.stream?.getTracks().forEach((t) => t.stop());
      const raw = new Blob(r.chunks, { type: r.chunks[0]?.type || "audio/webm" });
      onClip(raw, "Recording");
    };
    mr.start();
    r.start = Date.now();
    setRecording(true);
    r.timer = window.setInterval(() => {
      const s = (Date.now() - r.start) / 1000;
      setElapsed(s);
      if (s >= 20) stop(); // hard 20s cap
    }, 100);
  }, [onClip, stop]);

  return { recording, elapsed, micError, setMicError, toggle };
}
