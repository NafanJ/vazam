// Client-side audio → 16 kHz mono 16-bit WAV, matching the existing pipeline
// (api.py expects clips already decoded to 16k mono). Ported verbatim from the
// original dashboard's blobToWav16k / encodeWav.

type AudioCtor = typeof AudioContext;
function audioCtx(): AudioContext {
  const AC = (window.AudioContext || (window as unknown as { webkitAudioContext: AudioCtor }).webkitAudioContext);
  return new AC();
}

export function encodeWav(samples: Float32Array, rate: number): Blob {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buf);
  const wr = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i));
  };
  wr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  wr(8, "WAVE");
  wr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  wr(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    o += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}

// Decode any blob and re-render to 16 kHz mono, capped at 20s.
export async function blobToWav16k(blob: Blob): Promise<Blob> {
  const arrayBuf = await blob.arrayBuffer();
  const tmp = audioCtx();
  const decoded = await tmp.decodeAudioData(arrayBuf.slice(0));
  tmp.close();
  const dur = Math.min(decoded.duration, 20); // cap 20s
  const offline = new OfflineAudioContext(1, Math.ceil(16000 * dur), 16000);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();
  return encodeWav(rendered.getChannelData(0), 16000);
}

export const TRIM_MAX_SECONDS = 30; // cap on the rendered link-clip selection

// Render [start, end] of an already-decoded buffer to a 16 kHz mono WAV,
// capped at TRIM_MAX_SECONDS.
export async function renderSelectionWav(clipBuf: AudioBuffer, start: number, end: number): Promise<Blob> {
  const rate = 16000;
  const dur = Math.max(0.3, Math.min(end - start, TRIM_MAX_SECONDS));
  const off = new OfflineAudioContext(1, Math.ceil(rate * dur), rate);
  const src = off.createBufferSource();
  src.buffer = clipBuf;
  src.connect(off.destination);
  src.start(0, start, dur);
  const rendered = await off.startRendering();
  return encodeWav(rendered.getChannelData(0), rate);
}
