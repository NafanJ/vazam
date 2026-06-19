import type { CSSProperties } from "react";
import type { Match, HistoryItem, SceneResponse } from "../shared/types";
import { initial, lvlOf, pct } from "../shared/format";
import type { LogLine } from "./useProgress";

// CSS custom property (--lvl) needs a cast through CSSProperties.
const lvlVar = (color: string): CSSProperties => ({ ["--lvl" as keyof CSSProperties]: color } as CSSProperties);

export function MatchCard({ m }: { m: Match }) {
  const lv = lvlOf(m.match_level);
  const p = pct(m.confidence);
  return (
    <div className={`card ${lv.cls}`} style={lvlVar(lv.color)}>
      <div className="art">
        {m.image_url ? <img src={m.image_url} alt="" /> : <span className="mono">{initial(m.character_name)}</span>}
        <div className="scrim" />
        {m.show_title && <div className="show-chip">{m.show_title}</div>}
        <div className={`pill ${lv.cls}`}>
          <span className="pd" />
          {lv.pill}
        </div>
      </div>
      <div className="meta">
        <div className="name">{m.character_name || "No confident match"}</div>
        <div className="actor">
          voiced by <b>{m.actor_name || "—"}</b>
        </div>
        <div className="conf-row">
          <span className="lab">Confidence</span>
          <span className="pct" style={{ color: lv.color }}>
            {p}%
          </span>
        </div>
        <div className="confbar">
          <i style={{ width: `${p}%`, background: lv.color }} />
        </div>
        {m.window_agreement != null && (
          <div className="winrow">
            <span>window agreement</span>
            <span className="wb">
              <i style={{ width: `${Math.round(m.window_agreement * 100)}%` }} />
            </span>
            <span>{Math.round(m.window_agreement * 100)}%</span>
          </div>
        )}
      </div>
    </div>
  );
}

export function CompactRow({ m }: { m: Match | HistoryItem }) {
  const lv = lvlOf(m.match_level);
  const p = pct(m.confidence);
  return (
    <div className="crow" style={lvlVar(lv.color)}>
      <div className="av">{m.image_url ? <img src={m.image_url} alt="" /> : initial(m.character_name)}</div>
      <div className="info">
        <div className="nm">
          <b>{m.character_name || "No match"}</b>
          <span className="badge" style={{ background: lv.color }}>
            {p}%
          </span>
        </div>
        <div className="ac">
          {m.actor_name || "—"}
          {m.show_title ? ` · ${m.show_title}` : ""}
        </div>
        <div className="b">
          <i style={{ width: `${p}%`, background: lv.color }} />
        </div>
      </div>
    </div>
  );
}

export function SingleResults({
  results,
  onCorrect,
  onIncorrect,
  enrollLabel,
  enrollDone,
  incorrectActive,
}: {
  results: Match[];
  onCorrect: () => void;
  onIncorrect: () => void;
  enrollLabel: string;
  enrollDone: boolean;
  incorrectActive: boolean;
}) {
  if (!results.length) {
    return (
      <div className="empty">
        No character matches for this clip.
        <br />
        Try a longer or cleaner recording.
      </div>
    );
  }
  const [top, ...rest] = results;
  const considered = rest.slice(0, 4);
  return (
    <>
      <div className="section-label">Top match</div>
      <MatchCard m={top} />
      <button className={`enroll-cta${enrollDone ? " done" : ""}`} onClick={onCorrect}>
        {enrollLabel}
      </button>
      <button className={`incorrect-cta${incorrectActive ? " active" : ""}`} onClick={onIncorrect}>
        {incorrectActive ? "↓ Pick the correct character below, then Add clip" : "✗ Incorrect? Assign this clip to the right character"}
      </button>
      {considered.length > 0 && (
        <div className="considered">
          <div className="ttl">ALSO CONSIDERED</div>
          {considered.map((m, i) => (
            <div className="row" key={i}>
              <div>
                <b>{m.character_name}</b> <span>· {m.actor_name || ""}</span>
              </div>
              <span className="p">{pct(m.confidence)}%</span>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

export function SceneResults({ data }: { data: SceneResponse }) {
  const spk = data.speakers || {};
  const ids = Object.keys(spk);
  return (
    <>
      {data.show ? (
        <div className="show-banner">
          <div className="k">Likely show</div>
          <div className="t">{data.show.title}</div>
          <div className="m">
            {data.show.speakers_matched} of {data.show.speakers_total} speakers matched
          </div>
        </div>
      ) : (
        <div className="banner warn show" style={{ position: "static" }}>
          No single show inferred — showing best per-speaker matches.
        </div>
      )}
      {!ids.length ? (
        <div className="empty">No speakers detected.</div>
      ) : (
        ids.map((id, i) => {
          const best = (spk[id] || [])[0];
          return (
            <div key={id}>
              <div className="spk-label">Speaker {i + 1}</div>
              {best ? <CompactRow m={best} /> : <div className="empty">no match</div>}
            </div>
          );
        })
      )}
    </>
  );
}

export function History({ items }: { items: HistoryItem[] }) {
  if (!items.length) return null;
  return (
    <>
      <div className="section-label" style={{ marginTop: 10 }}>
        Previous recordings
      </div>
      <div className="history">
        {items.map((m, i) => (
          <CompactRow m={m} key={i} />
        ))}
      </div>
    </>
  );
}

export function Progress({
  stage,
  elapsedLabel,
  pbar,
  log,
}: {
  stage: string;
  elapsedLabel: string;
  pbar: number;
  log: LogLine[];
}) {
  return (
    <div className="progress show">
      <div className="prog-head">
        <span className="spinner" />
        <div>
          <div className="stage">{stage}</div>
          <div className="elapsed">{elapsedLabel}</div>
        </div>
      </div>
      <div className="pbar">
        <i style={{ width: `${pbar}%` }} />
      </div>
      <div className="log">
        {log.map((l, i) => (
          <div className={`line ${l.state}`} key={i}>
            <span className="ic">{l.state === "done" ? "✓" : "▸"}</span>
            {l.text}
          </div>
        ))}
      </div>
    </div>
  );
}
