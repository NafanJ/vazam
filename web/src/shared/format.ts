// Small formatting helpers shared by both pages.

import type { MatchLevel } from "./types";

export const fmtTime = (s: number): string =>
  `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export const initial = (n?: string | null): string =>
  (n || "?").trim().charAt(0).toUpperCase() || "?";

export interface LevelStyle {
  cls: MatchLevel;
  color: string;
  pill: string;
  hex: string;
}

export const LVL: Record<MatchLevel, LevelStyle> = {
  confident: { cls: "confident", color: "var(--good)", pill: "CONFIDENT", hex: "#2bd576" },
  possible: { cls: "possible", color: "var(--maybe)", pill: "POSSIBLE", hex: "#ffb020" },
  none: { cls: "none", color: "var(--none)", pill: "NO MATCH", hex: "#6b6b78" },
};

export const lvlOf = (level?: MatchLevel | null): LevelStyle => LVL[level ?? "none"] ?? LVL.none;

export const pct = (confidence?: number | null): number => Math.round((confidence || 0) * 100);
