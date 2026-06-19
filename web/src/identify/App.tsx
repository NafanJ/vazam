import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { call } from "../shared/api";
import { fmtTime } from "../shared/format";
import { blobToWav16k, renderSelectionWav, TRIM_MAX_SECONDS } from "../shared/wav";
import type {
  Character,
  EnrollResponse,
  IdentifyResponse,
  Match,
  HistoryItem,
  SceneResponse,
  Show,
} from "../shared/types";
import { useRecorder } from "./useRecorder";
import { useProgress } from "./useProgress";
import { History, Progress, SceneResults, SingleResults } from "./components";

type Mode = "single" | "scene";
type View = { kind: "single"; results: Match[] } | { kind: "scene"; data: SceneResponse } | null;
type StagedClip = { blob: Blob; name: string };
interface CharOption {
  value: string;
  character_id: number;
  actor_id: number;
  voice_label: string;
}

function load<T>(key: string, fallback: T): T {
  try {
    const v = JSON.parse(localStorage.getItem(key) || "null");
    return v ?? fallback;
  } catch {
    return fallback;
  }
}
function save<T>(key: string, value: T): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore quota / private-mode errors */
  }
}

export default function App() {
  const [mode, setMode] = useState<Mode>("single");
  const [isolate, setIsolate] = useState(true);
  const [showId, setShowId] = useState("");
  const [shows, setShows] = useState<Show[]>([]);
  const [characters, setCharacters] = useState<Character[]>([]);

  const [clip, setClip] = useState<StagedClip | null>(null);
  const [running, setRunning] = useState(false);
  const [view, setView] = useState<View>(null);
  const [history, setHistory] = useState<HistoryItem[]>(() => load<HistoryItem[]>("vazam_history", []));
  const [error, setError] = useState<string | null>(null);

  // enroll button states
  const [topEnroll, setTopEnroll] = useState<{ label?: string; done: boolean }>({ done: false });
  const [incorrectActive, setIncorrectActive] = useState(false);
  const [enrollOpen, setEnrollOpen] = useState(false);
  const [enrollValue, setEnrollValue] = useState("");
  const [enrollBtn, setEnrollBtn] = useState<{ label: string; disabled: boolean }>({ label: "Add clip", disabled: true });

  // YouTube link section
  const [urlValue, setUrlValue] = useState("");
  const [urlLoadLabel, setUrlLoadLabel] = useState("Load clip");
  const [clipAudioSrc, setClipAudioSrc] = useState<string>("");
  const [trimPanel, setTrimPanel] = useState(false);
  const [trimMax, setTrimMax] = useState(0);
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState(0);
  const [urlEnrollValue, setUrlEnrollValue] = useState("");
  const [urlEnrollLabel, setUrlEnrollLabel] = useState("Add selection to fingerprint");

  const clipBufRef = useRef<AudioBuffer | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const enrollDetailsRef = useRef<HTMLDetailsElement | null>(null);
  const enrollInputRef = useRef<HTMLInputElement | null>(null);

  const prog = useProgress();

  // ── staging ──────────────────────────────────────────────────────────────
  const stageBlob = useCallback(async (blob: Blob, name: string) => {
    try {
      const wav = await blobToWav16k(blob);
      setClip({ blob: wav, name });
      setEnrollBtn((b) => ({ ...b, disabled: false }));
    } catch (e) {
      setError("Could not decode audio — try a different clip. (" + (e as Error).message + ")");
    }
  }, []);

  const recorder = useRecorder(stageBlob);
  useEffect(() => {
    document.body.classList.toggle("recording", recorder.recording);
  }, [recorder.recording]);

  const clearClip = () => {
    setClip(null);
    setEnrollBtn((b) => ({ ...b, disabled: true }));
  };

  // ── character datalist (shared by both enroll pickers) ─────────────────────
  const charOptions = useMemo<CharOption[]>(() => {
    return characters
      .filter((c) => c.actor_id != null)
      .sort((a, b) => Number(b.samples > 0) - Number(a.samples > 0) || String(a.name).localeCompare(b.name))
      .map((c) => {
        const tail = c.samples ? ` · ${c.samples} clip${c.samples > 1 ? "s" : ""}` : " · no clips";
        const value = `${c.name} — ${c.actor_name || "?"}${c.show_title ? ` (${c.show_title})` : ""}${tail}`;
        return { value, character_id: c.id, actor_id: c.actor_id as number, voice_label: c.name };
      });
  }, [characters]);
  const charMap = useMemo(() => new Map(charOptions.map((o) => [o.value, o])), [charOptions]);
  const resolveChar = (value: string): CharOption | null => charMap.get(value.trim()) || null;

  const loadCharacters = useCallback(async () => {
    try {
      const res = await call("/characters");
      setCharacters(await res.json());
    } catch {
      /* leave datalist empty */
    }
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const r = await call("/shows");
        setShows(await r.json());
      } catch {
        /* shows are optional */
      }
      loadCharacters();
    })();
  }, [loadCharacters]);

  // ── history ────────────────────────────────────────────────────────────────
  const addHistory = useCallback((m: Match) => {
    setHistory((prev) => {
      const next = [
        {
          character_name: m.character_name,
          actor_name: m.actor_name,
          show_title: m.show_title,
          confidence: m.confidence,
          match_level: m.match_level,
          image_url: m.image_url,
          ts: Date.now(),
        },
        ...prev,
      ].slice(0, 8);
      save("vazam_history", next);
      return next;
    });
  }, []);

  // ── identify ────────────────────────────────────────────────────────────────
  const runSingle = useCallback(
    async (fd: FormData) => {
      // Plain POST (not the NDJSON stream): a single short-lived request survives
      // the Cloudflare tunnel / mobile network handoffs. Stages are simulated.
      prog.scheduleStages(
        [
          isolate ? "Isolating music & SFX (Demucs)…" : null,
          "Trimming silence (VAD)…",
          "Embedding voiceprint…",
          "Searching reference voices…",
          "Verifying across sub-windows…",
        ],
        600
      );
      const res = await call("/identify", { method: "POST", body: fd });
      const data = (await res.json()) as IdentifyResponse;
      prog.finish();
      prog.show(false);
      const results = data.results || [];
      setView({ kind: "single", results });
      setTopEnroll({ done: false });
      setIncorrectActive(false);
      if (results.length) addHistory(results[0]);
    },
    [isolate, prog, addHistory]
  );

  const runScene = useCallback(
    async (fd: FormData) => {
      prog.scheduleStages(
        ["Diarizing speakers…", "Isolating music & SFX (Demucs)…", "Embedding voiceprints…", "Searching reference voices…", "Inferring show…"],
        500
      );
      const res = await call("/identify/show", { method: "POST", body: fd });
      const data = (await res.json()) as SceneResponse;
      prog.finish();
      prog.show(false);
      setView({ kind: "scene", data });
      const spk = data.speakers || {};
      const first = Object.keys(spk)[0];
      if (first && spk[first]?.[0]) addHistory(spk[first][0]);
    },
    [prog, addHistory]
  );

  const identify = async () => {
    if (!clip || running) return; // ignore re-entry while one is in flight
    setRunning(true);
    setError(null);
    setView(null);
    prog.show(true);
    const fd = new FormData();
    fd.append("audio", clip.blob, "clip.wav");
    fd.append("isolate", String(isolate));
    if (showId) fd.append("show_id", showId);
    try {
      if (mode === "scene") await runScene(fd);
      else await runSingle(fd);
    } catch (e) {
      prog.show(false);
      setError("Analysis failed — " + (e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  // ── enroll ────────────────────────────────────────────────────────────────
  const enroll = async (
    blob: Blob,
    p: { actor_id: number; voice_label: string; character_id?: number }
  ): Promise<EnrollResponse> => {
    const fd = new FormData();
    fd.append("audio", blob, "clip.wav");
    fd.append("actor_id", String(p.actor_id));
    fd.append("voice_label", p.voice_label);
    if (p.character_id != null) fd.append("character_id", String(p.character_id));
    fd.append("isolate", String(isolate));
    const res = await call("/enroll", { method: "POST", body: fd });
    return res.json() as Promise<EnrollResponse>;
  };

  const enrollTopResult = async () => {
    const top = view?.kind === "single" ? view.results[0] : undefined;
    if (!clip || !top || top.actor_id == null) return;
    setTopEnroll({ label: "Adding…", done: false });
    try {
      const d = await enroll(clip.blob, {
        actor_id: top.actor_id,
        voice_label: top.character_name || "",
        character_id: top.character_id ?? undefined,
      });
      setTopEnroll({ label: `✓ Added — ${d.samples} clips`, done: true });
      loadCharacters();
    } catch (e) {
      setTopEnroll({ label: "✕ " + (e as Error).message, done: false });
    }
  };

  // Wrong match: open the picker so the staged clip routes to the right character.
  const markIncorrect = () => {
    setIncorrectActive(true);
    setEnrollOpen(true);
    enrollDetailsRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    setTimeout(() => enrollInputRef.current?.focus(), 0);
  };

  const enrollStaged = async () => {
    if (!clip) return;
    const c = resolveChar(enrollValue);
    if (!c) {
      setError("Pick a character from the list.");
      return;
    }
    setEnrollBtn({ label: "Adding…", disabled: true });
    try {
      const d = await enroll(clip.blob, { actor_id: c.actor_id, voice_label: c.voice_label, character_id: c.character_id });
      setEnrollBtn({ label: `✓ Added — ${d.samples} clips`, disabled: true });
      loadCharacters();
    } catch {
      setEnrollBtn({ label: "✕ failed", disabled: false });
    }
  };

  // ── from a YouTube/clip link ────────────────────────────────────────────────
  const selRange = (): [number, number] => (trimStart <= trimEnd ? [trimStart, trimEnd] : [trimEnd, trimStart]);
  const selLabel = (() => {
    const [a, b] = selRange();
    const d = b - a;
    return `${a.toFixed(1)} – ${b.toFixed(1)}s (${d.toFixed(1)}s${d > TRIM_MAX_SECONDS ? `, uses first ${TRIM_MAX_SECONDS}s` : ""})`;
  })();

  const loadClip = async () => {
    const url = urlValue.trim();
    if (!url) {
      setError("Paste a link first.");
      return;
    }
    if (running) return;
    setUrlLoadLabel("Downloading…");
    setError(null);
    try {
      const res = await call("/fetch/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const blob = await res.blob();
      setClipAudioSrc(URL.createObjectURL(blob));
      const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const buf = await new AC().decodeAudioData(await blob.arrayBuffer());
      clipBufRef.current = buf;
      const dur = buf.duration;
      setTrimMax(dur);
      setTrimStart(0);
      setTrimEnd(Math.min(dur, TRIM_MAX_SECONDS));
      setTrimPanel(true);
      setUrlLoadLabel("Reload clip");
    } catch (err) {
      setError("Could not load that link — " + (err as Error).message);
      setUrlLoadLabel("Load clip");
    }
  };

  const playSelection = () => {
    const a = audioRef.current;
    if (!a) return;
    const [s, e] = selRange();
    a.currentTime = s;
    a.play();
    const tick = () => {
      if (a.currentTime >= e) {
        a.pause();
        a.removeEventListener("timeupdate", tick);
      }
    };
    a.addEventListener("timeupdate", tick);
  };

  const identifySelection = async () => {
    if (!clipBufRef.current) {
      setError("Load a clip first.");
      return;
    }
    if (running) return;
    setRunning(true);
    setError(null);
    setView(null);
    prog.show(true);
    try {
      const [s, e] = selRange();
      const wav = await renderSelectionWav(clipBufRef.current, s, e);
      const fd = new FormData();
      fd.append("audio", wav, "clip.wav");
      fd.append("isolate", String(isolate));
      if (showId) fd.append("show_id", showId);
      await runSingle(fd); // shared stage log + /identify + render
    } catch (e) {
      prog.show(false);
      setError("Identify failed — " + (e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  const enrollSelection = async () => {
    if (!clipBufRef.current) {
      setError("Load a clip first.");
      return;
    }
    const c = resolveChar(urlEnrollValue);
    if (!c) {
      setError("Pick a character from the list.");
      return;
    }
    if (running) return;
    setUrlEnrollLabel("Adding…");
    setRunning(true);
    try {
      const [s, e] = selRange();
      const wav = await renderSelectionWav(clipBufRef.current, s, e);
      const d = await enroll(wav, { actor_id: c.actor_id, voice_label: c.voice_label, character_id: c.character_id });
      setUrlEnrollLabel(`✓ Added — ${d.samples} clips`);
      loadCharacters();
    } catch (e) {
      setUrlEnrollLabel("Add selection to fingerprint");
      setError("Enroll failed — " + (e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  // ── derived ─────────────────────────────────────────────────────────────────
  const canIdentify = !!clip && !running;
  const topName = view?.kind === "single" ? view.results[0]?.character_name : undefined;
  const topEnrollLabel =
    topEnroll.label ?? `➕ Correct? Add this clip to ${topName ?? "this character"}'s fingerprint`;

  return (
    <div className="wrap">
      {/* HEADER */}
      <header className="top">
        <div className="brand">
          <span className="dot" />
          <div>
            <b>Vazam</b>
            <small>who's that voice?</small>
          </div>
        </div>
        <a className="nav-link" href="characters.html">
          Characters
        </a>
      </header>

      {/* MODE */}
      <div className="seg">
        <button className={mode === "single" ? "on" : ""} onClick={() => setMode("single")}>
          Single voice
        </button>
        <button className={mode === "scene" ? "on" : ""} onClick={() => setMode("scene")}>
          Scene · who's in it
        </button>
      </div>

      {/* FILTERS */}
      <div className="filters">
        <div className="sel">
          <select value={showId} onChange={(e) => setShowId(e.target.value)}>
            <option value="">All shows (open search)</option>
            {shows.map((s) => (
              <option key={s.id} value={s.id}>
                {s.title}
              </option>
            ))}
          </select>
        </div>
        <button className={`toggle${isolate ? " on" : ""}`} aria-pressed={isolate} onClick={() => setIsolate((v) => !v)}>
          <span className="sw" />
          Strip SFX
        </button>
      </div>

      {/* BANNERS */}
      {recorder.micError && <div className="banner warn show">{recorder.micError}</div>}
      {error && <div className="banner err show">⚠ {error}</div>}

      {/* HERO */}
      <div className="hero">
        <div className="recwrap">
          <span className="ring r1" />
          <span className="ring r2" />
          <span className="pulse p1" />
          <span className="pulse p2" />
          <button className="recbtn" aria-label="Record" onClick={recorder.toggle}>
            <svg viewBox="0 0 24 24" fill="none">
              <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Z" fill="#fff" />
              <path d="M6 11a6 6 0 0 0 12 0M12 17v3" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
            <span className="bar" />
            <span className="bar" />
            <span className="bar" />
            <span className="bar" />
            <span className="bar" />
            <span className="bar" />
            <span className="bar" />
          </button>
        </div>
        <div className="timer">{fmtTime(recorder.elapsed)}</div>
        <div className="hero-label">Tap to record</div>
        <div className="hero-sub">point at the screen · a few seconds is enough</div>
        <div className="or">or</div>
        <label className="upload">
          ⬆ Upload m4a / mp3 / wav
          <input
            type="file"
            accept="audio/*,.m4a,.mp3,.wav"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) stageBlob(f, f.name.replace(/\.[^.]+$/, ""));
            }}
          />
        </label>
      </div>

      {/* STAGED */}
      <div className={`staged${clip ? " show" : ""}`}>
        <span className="play">▶</span>
        <span>{clip ? <><b>Ready: {clip.name}</b> — tap Identify.</> : "Ready — tap Identify."}</span>
        <span className="x" onClick={clearClip}>
          ✕
        </span>
      </div>

      {/* CTA */}
      <button className={`cta${canIdentify ? " ready" : ""}`} disabled={!canIdentify} onClick={identify}>
        {running ? "Identifying…" : "Identify"}
      </button>

      {/* ENROLLMENT */}
      <details
        ref={enrollDetailsRef}
        open={enrollOpen}
        onToggle={(e) => setEnrollOpen((e.target as HTMLDetailsElement).open)}
      >
        <summary>
          ➕ Add staged clip to a character's fingerprint <span>⌄</span>
        </summary>
        <div className="body">
          <div className="field">
            <label>Character</label>
            <input
              ref={enrollInputRef}
              list="charlist"
              autoComplete="off"
              placeholder="search any character…"
              value={enrollValue}
              onChange={(e) => setEnrollValue(e.target.value)}
            />
          </div>
          <button className="mini-btn" disabled={enrollBtn.disabled} onClick={enrollStaged}>
            {enrollBtn.label}
          </button>
        </div>
      </details>
      <datalist id="charlist">
        {charOptions.map((o) => (
          <option key={o.value} value={o.value} />
        ))}
      </datalist>

      {/* FROM A LINK */}
      <details>
        <summary>
          🔗 From a YouTube link <span>⌄</span>
        </summary>
        <div className="body">
          <div className="field">
            <label>Video / clip URL</label>
            <input
              type="text"
              inputMode="url"
              placeholder="https://youtu.be/…"
              value={urlValue}
              onChange={(e) => setUrlValue(e.target.value)}
            />
          </div>
          <button className="mini-btn" disabled={running} onClick={loadClip}>
            {urlLoadLabel}
          </button>
          {trimPanel && (
            <div>
              <audio ref={audioRef} src={clipAudioSrc} controls style={{ width: "100%", marginTop: 10 }} />
              <div className="trim">
                <div className="trimrow">
                  <label>Start</label>
                  <input
                    type="range"
                    min={0}
                    max={trimMax}
                    step={0.1}
                    value={trimStart}
                    onChange={(e) => setTrimStart(+e.target.value)}
                  />
                  <button className="tnow" title="set to playhead" onClick={() => setTrimStart(audioRef.current?.currentTime ?? 0)}>
                    ⏱
                  </button>
                </div>
                <div className="trimrow">
                  <label>End</label>
                  <input
                    type="range"
                    min={0}
                    max={trimMax}
                    step={0.1}
                    value={trimEnd}
                    onChange={(e) => setTrimEnd(+e.target.value)}
                  />
                  <button className="tnow" title="set to playhead" onClick={() => setTrimEnd(audioRef.current?.currentTime ?? 0)}>
                    ⏱
                  </button>
                </div>
                <div className="trimsel">
                  <span>{selLabel}</span>
                  <button className="tplay" onClick={playSelection}>
                    ▶ selection
                  </button>
                </div>
              </div>
              <button className="mini-btn" disabled={running} onClick={identifySelection}>
                Identify selection
              </button>
              <div className="field" style={{ marginTop: 8 }}>
                <label>…or add the selection to a character's fingerprint</label>
                <input
                  list="charlist"
                  autoComplete="off"
                  placeholder="search any character…"
                  value={urlEnrollValue}
                  onChange={(e) => setUrlEnrollValue(e.target.value)}
                />
              </div>
              <button className="mini-btn ghost" disabled={running} onClick={enrollSelection}>
                {urlEnrollLabel}
              </button>
            </div>
          )}
          <div className="hero-sub" style={{ marginTop: 6 }}>
            Downloads up to ~4 min server-side — play it, drag the handles to the voice you want, then identify/enroll just
            that selection (uses up to 30s of it).
          </div>
        </div>
      </details>

      {/* PROGRESS */}
      {prog.visible && <Progress stage={prog.stage} elapsedLabel={prog.elapsedLabel} pbar={prog.pbar} log={prog.log} />}

      {/* RESULTS */}
      <div className="results">
        {view?.kind === "single" && (
          <SingleResults
            results={view.results}
            onCorrect={enrollTopResult}
            onIncorrect={markIncorrect}
            enrollLabel={topEnrollLabel}
            enrollDone={topEnroll.done}
            incorrectActive={incorrectActive}
          />
        )}
        {view?.kind === "scene" && <SceneResults data={view.data} />}
      </div>

      {/* HISTORY */}
      <History items={history} />
    </div>
  );
}
