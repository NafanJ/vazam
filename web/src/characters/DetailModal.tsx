import { useCallback, useEffect, useState } from "react";
import { apiBase, call } from "../shared/api";
import { initial } from "../shared/format";
import type { CharacterDetail } from "../shared/types";

export function DetailModal({
  id,
  onClose,
  onPatched,
  onSampleDeleted,
}: {
  id: number;
  onClose: () => void;
  onPatched: (id: number, patch: { image_url: string; occupation: string }) => void;
  onSampleDeleted: (id: number) => void;
}) {
  const [c, setC] = useState<CharacterDetail | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [img, setImg] = useState("");
  const [occ, setOcc] = useState("");
  const [saveLabel, setSaveLabel] = useState("Save changes");
  const [saveDone, setSaveDone] = useState(false);
  const [deleting, setDeleting] = useState<Record<number, boolean>>({});

  const reload = useCallback(async () => {
    try {
      const r = await call("/characters/" + id);
      const data = (await r.json()) as CharacterDetail;
      setC(data);
      setImg(data.image_url || "");
      setOcc(data.occupation || "");
    } catch (e) {
      setLoadError((e as Error).message);
    }
  }, [id]);

  useEffect(() => {
    reload();
  }, [reload]);

  const save = async () => {
    setSaveLabel("Saving…");
    const patch = { image_url: img, occupation: occ };
    try {
      await call("/characters/" + id, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
    } catch (e) {
      setSaveDone(false);
      setSaveLabel("✕ " + (e as Error).message);
      return;
    }
    onPatched(id, patch);
    setSaveDone(true);
    setSaveLabel("✓ Saved");
  };

  const delSource = async (eid: number) => {
    if (!c) return;
    if (!confirm(`Remove this clip from ${c.name}'s fingerprint? This can't be undone.`)) return;
    setDeleting((d) => ({ ...d, [eid]: true }));
    try {
      await call("/embeddings/" + eid, { method: "DELETE" });
    } catch (err) {
      setDeleting((d) => ({ ...d, [eid]: false }));
      alert("Delete failed: " + (err as Error).message);
      return;
    }
    onSampleDeleted(id);
    reload(); // refresh sources list + clip count
  };

  const onOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) onClose();
  };

  let body;
  if (loadError) {
    body = (
      <div style={{ padding: 60, textAlign: "center", color: "var(--muted2)" }}>Failed to load ({loadError}).</div>
    );
  } else if (!c) {
    body = <div style={{ padding: 60, textAlign: "center", color: "var(--muted2)" }}>Loading…</div>;
  } else {
    const emb = c.embeddings || [];
    const sub = [c.actor_name, c.show_title, c.occupation].filter(Boolean).join(" · ");
    body = (
      <>
        <div className="m-art">
          {c.image_url ? <img src={c.image_url} alt="" /> : <span className="mono">{initial(c.name)}</span>}
          <div className="scrim" />
          <button className="close" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="m-body">
          <div className="m-name">{c.name}</div>
          <div className="m-sub">{sub || "—"}</div>
          <div className="field">
            <label>Image URL</label>
            <input value={img} placeholder="https://…" onChange={(e) => setImg(e.target.value)} />
          </div>
          <div className="field">
            <label>Occupation</label>
            <input value={occ} placeholder="e.g. bounty hunter" onChange={(e) => setOcc(e.target.value)} />
          </div>
          <button className={`save${saveDone ? " done" : ""}`} onClick={save}>
            {saveLabel}
          </button>

          <div className="src-ttl">
            Voice sources · {emb.length} clip{emb.length === 1 ? "" : "s"}
          </div>
          {emb.length ? (
            <>
              {emb.map((e) => (
                <div key={e.id}>
                  <div className="src">
                    <div className="l">
                      <span className={e.verified ? "v" : "uv"}>{e.verified ? "✓" : "○"}</span>
                      {e.source_url ? (
                        <a href={e.source_url} target="_blank" rel="noopener">
                          ▶ {e.voice_label || e.audio_source || "clip"}
                        </a>
                      ) : (
                        <span>{e.voice_label || e.audio_source || "clip"}</span>
                      )}
                      <span style={{ color: "var(--muted2)" }}>· {e.audio_source || ""}</span>
                    </div>
                    <div className="r">
                      <span className="meta">
                        {e.duration_s != null ? (+e.duration_s).toFixed(1) + "s" : ""}
                        {e.quality_score != null ? " · q " + (+e.quality_score).toFixed(2) : ""}
                      </span>
                      <button
                        className="srcdel"
                        title="Remove this clip from the fingerprint"
                        disabled={deleting[e.id]}
                        onClick={() => delSource(e.id)}
                      >
                        {deleting[e.id] ? "…" : "🗑"}
                      </button>
                    </div>
                  </div>
                  {e.audio_path && (
                    <audio className="srcaudio" controls preload="none" src={`${apiBase()}/embeddings/${e.id}/audio`} />
                  )}
                </div>
              ))}
              <div className="src-note">
                Enrolled clips have an inline player; ▶ links open the original source (those aren't stored). Deleting a
                clip updates the fingerprint immediately.
              </div>
            </>
          ) : (
            <div className="src-empty">No voice fingerprint yet — record a clip on the dashboard and enroll it.</div>
          )}
        </div>
      </>
    );
  }

  return (
    <div className="overlay show" onClick={onOverlayClick}>
      <div className="modal">{body}</div>
    </div>
  );
}
