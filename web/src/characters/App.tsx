import { useEffect, useMemo, useState } from "react";
import { call } from "../shared/api";
import { initial } from "../shared/format";
import type { Character } from "../shared/types";
import { DetailModal } from "./DetailModal";

const PAGE_SIZE = 50;

type SortMode = "voiced" | "clips";
const SORTS: Record<SortMode, (a: Character, b: Character) => number> = {
  voiced: (a, b) => Number(b.samples > 0) - Number(a.samples > 0) || String(a.name).localeCompare(b.name),
  clips: (a, b) => (b.samples || 0) - (a.samples || 0) || String(a.name).localeCompare(b.name),
};

export default function App() {
  const [all, setAll] = useState<Character[]>([]);
  const [countText, setCountText] = useState("Loading…");
  const [loadError, setLoadError] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [voiceOnly, setVoiceOnly] = useState(false);
  const [sortMode, setSortMode] = useState<SortMode>("voiced");
  const [page, setPage] = useState(1);

  const [openId, setOpenId] = useState<number | null>(null);

  useEffect(() => {
    (async () => {
      let r: Response;
      try {
        r = await call("/characters");
      } catch (e) {
        setCountText("Failed to load characters.");
        setLoadError(`Could not reach the server (${(e as Error).message}).`);
        return;
      }
      const data = (await r.json()) as Character[];
      setAll(data);
      const withVoice = data.filter((c) => c.samples > 0).length;
      setCountText(`${data.length} characters · ${withVoice} with a voice fingerprint`);
    })();
  }, []);

  // Esc closes the modal.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpenId(null);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return all
      .filter((c) => {
        if (voiceOnly && !(c.samples > 0)) return false;
        if (!q) return true;
        const hay = `${c.name} ${c.actor_name || ""} ${c.show_title || ""} ${c.occupation || ""}`.toLowerCase();
        return hay.includes(q);
      })
      .sort(SORTS[sortMode] || SORTS.voiced);
  }, [all, query, voiceOnly, sortMode]);

  const total = filtered.length;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const curPage = Math.min(Math.max(page, 1), pages);
  const start = (curPage - 1) * PAGE_SIZE;
  const slice = filtered.slice(start, start + PAGE_SIZE);

  const resetPage = () => setPage(1);
  const goPage = (p: number) => {
    setPage(p);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const onPatched = (id: number, patch: { image_url: string; occupation: string }) =>
    setAll((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  const onSampleDeleted = (id: number) =>
    setAll((prev) => prev.map((c) => (c.id === id && c.samples > 0 ? { ...c, samples: c.samples - 1 } : c)));

  return (
    <div className="wrap">
      <div className="top">
        <a className="back" href="/">
          ‹ Dashboard
        </a>
      </div>
      <div style={{ marginTop: 12 }}>
        <div className="ttl">Characters</div>
        <div className="count">{countText}</div>
      </div>

      <div className="search">
        <span className="ic">⌕</span>
        <input
          placeholder="name, actor, show, occupation…"
          autoComplete="off"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            resetPage();
          }}
        />
      </div>
      <div className="filters">
        <div className="fgroup">
          <button
            className={`toggle${voiceOnly ? " on" : ""}`}
            aria-pressed={voiceOnly}
            onClick={() => {
              setVoiceOnly((v) => !v);
              resetPage();
            }}
          >
            <span className="sw" />
            Only with a voice
          </button>
          <select
            className="sortsel"
            aria-label="Sort"
            value={sortMode}
            onChange={(e) => {
              setSortMode(e.target.value as SortMode);
              resetPage();
            }}
          >
            <option value="voiced">Voiced first</option>
            <option value="clips">Most clips</option>
          </select>
        </div>
        <span className="shown">{total ? `${start + 1}–${start + slice.length} of ${total}` : ""}</span>
      </div>

      <div className="list">
        {loadError ? (
          <div className="empty">{loadError}</div>
        ) : !total ? (
          <div className="empty">
            No characters match “{query}”.
            <br />
            Try a different name, actor, or show.
          </div>
        ) : (
          slice.map((c) => {
            const sub = [c.actor_name, c.show_title, c.occupation].filter(Boolean).join(" · ");
            return (
              <button className="row" key={c.id} onClick={() => setOpenId(c.id)}>
                <div className="av">
                  {c.image_url ? <img src={c.image_url} alt="" loading="lazy" /> : initial(c.name)}
                </div>
                <div className="info">
                  <div className="nm">{c.name}</div>
                  <div className="sub">{sub || "—"}</div>
                </div>
                {c.samples > 0 ? (
                  <span className="badge has">
                    {c.samples} clip{c.samples > 1 ? "s" : ""}
                  </span>
                ) : (
                  <span className="badge no">no voice</span>
                )}
              </button>
            );
          })
        )}
      </div>

      {total > PAGE_SIZE && (
        <div className="pager">
          <button disabled={curPage <= 1} onClick={() => goPage(curPage - 1)}>
            ← Prev
          </button>
          <span className="pginfo">
            Page {curPage} of {pages}
          </span>
          <button disabled={curPage >= pages} onClick={() => goPage(curPage + 1)}>
            Next →
          </button>
        </div>
      )}

      {openId != null && (
        <DetailModal
          id={openId}
          onClose={() => setOpenId(null)}
          onPatched={onPatched}
          onSampleDeleted={onSampleDeleted}
        />
      )}
    </div>
  );
}
