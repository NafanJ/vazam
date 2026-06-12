// ── API response types ────────────────────────────────────────────────────────

export type MatchLevel = "confident" | "possible" | "none";

export interface IdentificationMatch {
  actor_id: number;
  actor_name: string;
  character_name: string;
  confidence: number;
  /** Fraction of verification windows this actor won; null when the clip was too short to verify. */
  window_agreement: number | null;
  match_level: MatchLevel;
}

export interface IdentifyResponse {
  results: IdentificationMatch[];
}

export interface MultiIdentifyResponse {
  speakers: Record<string, IdentificationMatch[]>;
}

export interface InferredShow {
  show_id: number;
  title: string;
  /** Distinct detected speakers with a candidate in this show's cast. */
  speakers_matched: number;
  /** Distinct speakers detected in the clip. */
  speakers_total: number;
  score: number;
}

export interface ShowIdentifyResponse {
  /** null when no cast consensus was found (single speaker or no agreement). */
  show: InferredShow | null;
  speakers: Record<string, IdentificationMatch[]>;
}

export interface Actor {
  id: number;
  name: string;
  bio: string | null;
  image_url: string | null;
}

export interface FilmographyEntry {
  character_name: string;
  show_title: string | null;
  media_type: string | null;
  year: number | null;
  image_url: string | null;
}

export interface ActorProfile extends Actor {
  anilist_id: number | null;
  filmography: FilmographyEntry[];
}

export interface Show {
  id: number;
  title: string;
  media_type: string;
  year: number | null;
  image_url: string | null;
}

export interface HealthResponse {
  status: string;
  index_size: number;
  db_embeddings: number;
  vad_enabled: boolean;
  diarization: boolean;
  device: string;
}

// ── Navigation types ──────────────────────────────────────────────────────────

export type RootStackParamList = {
  Home: undefined;
  Results: { results: IdentificationMatch[]; multiSpeaker?: boolean };
  ActorProfile: { actorId: number; actorName: string };
  ShowSearch: undefined;
  Settings: undefined;
};

// ── App state ─────────────────────────────────────────────────────────────────

export type RecordingState = "idle" | "recording" | "processing" | "done" | "error";
