/**
 * vazam.ts — typed API client for the Vazam backend
 *
 * All methods return the parsed response body on success and throw an
 * AxiosError on HTTP errors. Callers should wrap in try/catch.
 */

import axios from "axios";
import type {
  Actor,
  ActorProfile,
  HealthResponse,
  IdentifyResponse,
  MultiIdentifyResponse,
  Show,
} from "../types";

// Override at build time via env or settings screen
const DEFAULT_BASE_URL = "http://localhost:8000";

const client = axios.create({
  baseURL: DEFAULT_BASE_URL,
  timeout: 30_000,
});

/** Update the base URL at runtime (e.g. from settings). */
export function setBaseUrl(url: string): void {
  client.defaults.baseURL = url.replace(/\/$/, "");
}

// ── Identification ────────────────────────────────────────────────────────────

export interface IdentifyOptions {
  /** Local file path or URI to the recorded audio. */
  audioPath: string;
  /** Run Demucs vocal isolation server-side (slower but more accurate on TV audio). */
  isolate?: boolean;
  /** Restrict search to this show's cast. */
  showId?: number;
  /** Max candidates to return (1-20). */
  topK?: number;
}

export async function identify(opts: IdentifyOptions): Promise<IdentifyResponse> {
  const form = new FormData();
  form.append("audio", {
    uri:  opts.audioPath,
    name: "recording.wav",
    type: "audio/wav",
  } as any);
  form.append("isolate", String(opts.isolate ?? false));
  form.append("top_k",   String(opts.topK ?? 5));
  if (opts.showId != null) {
    form.append("show_id", String(opts.showId));
  }

  const { data } = await client.post<IdentifyResponse>("/identify", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function identifyMulti(opts: IdentifyOptions): Promise<MultiIdentifyResponse> {
  const form = new FormData();
  form.append("audio", {
    uri:  opts.audioPath,
    name: "recording.wav",
    type: "audio/wav",
  } as any);
  form.append("isolate", String(opts.isolate ?? true));
  form.append("top_k",   String(opts.topK ?? 3));

  const { data } = await client.post<MultiIdentifyResponse>("/identify/multi", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

// ── Actors ────────────────────────────────────────────────────────────────────

export async function listActors(limit = 100, offset = 0): Promise<Actor[]> {
  const { data } = await client.get<Actor[]>("/actors", { params: { limit, offset } });
  return data;
}

export async function getActorProfile(actorId: number): Promise<ActorProfile> {
  const { data } = await client.get<ActorProfile>(`/actors/${actorId}`);
  return data;
}

// ── Shows ─────────────────────────────────────────────────────────────────────

export async function listShows(limit = 100, offset = 0): Promise<Show[]> {
  const { data } = await client.get<Show[]>("/shows", { params: { limit, offset } });
  return data;
}

export async function searchShows(query: string): Promise<Show[]> {
  const { data } = await client.get<Show[]>("/shows/search", { params: { q: query } });
  return data;
}

export async function getShow(showId: number): Promise<Show> {
  const { data } = await client.get<Show>(`/shows/${showId}`);
  return data;
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function health(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>("/health");
  return data;
}
