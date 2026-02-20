// ── API response types ────────────────────────────────────────────────────────

export type MatchLevel = "confident" | "possible" | "none";

export interface IdentificationMatch {
  actor_id: number;
  actor_name: string;
  character_name: string;
  confidence: number;
  match_level: MatchLevel;
}

export interface IdentifyResponse {
  results: IdentificationMatch[];
}

export interface MultiIdentifyResponse {
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
